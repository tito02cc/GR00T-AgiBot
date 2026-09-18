"""Deterministic runner regression; no GDK, sockets, GPU, or checkpoints."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import threading
import types
import unittest
from unittest import mock

from agibot.rtc.action_queue import ActionQueue
from agibot.scripts import run_g2_groot_rtc_place_inference as runner
import numpy as np


class FakeClock:
    def __init__(self):
        self.now_ns = 100_000_000_000

    def monotonic(self):
        return self.now_ns / 1e9

    def time_ns(self):
        return self.now_ns + 1_700_000_000_000_000_000

    def sleep(self, seconds):
        self.now_ns += round(seconds * 1e9)
        if self.now_ns > 110_000_000_000:
            raise AssertionError("simulated run exceeded ten seconds")


def targets_at(start, *, predicted=False):
    targets = np.zeros((16, 8), dtype=np.float64)
    targets[:, 1:3] = [-0.2, 1.05]
    targets[:, 6] = 1.0
    for offset, target in enumerate(targets):
        index = start + offset
        target[0] = 0.5 + index * 0.009 if index < 16 else 0.65 - min((index - 16) * 0.06, 0.24)
        # Distinguish replaceable future predictions from the original chunk.
        target[2] += 0.001 if predicted else 0
        target[7] = [0.0, -0.123, -0.312][index % 3] if index < 16 else -0.785
    return targets


class FakeActionClient:
    """One fake socket and serial 100 ms execution, with real receipt schema."""

    def __init__(
        self,
        clock,
        *,
        fatal_after=None,
        reject_index=None,
        fail_index=None,
        status_delays=(0.0,),
    ):
        self.clock = clock
        self.fatal_after = fatal_after
        self.reject_index = reject_index
        self.fail_index = fail_index
        self.owner_thread = threading.get_ident()
        self.calls = []
        self.commands = []
        self.release_event = None
        self.status_delays = status_delays
        self.status_replies = []

    def request(self, payload):
        thread_id = threading.get_ident()
        if thread_id != self.owner_thread:
            raise AssertionError("action socket used outside its owning thread")
        self.calls.append((self.clock.now_ns, thread_id, deepcopy(payload)))
        if payload["op"] == "execute_h1_gripper":
            index = len(self.commands)
            if index == self.reject_index:
                return {"ok": False, "accepted": False, "error": "fake rejection"}
            start = max(
                self.clock.now_ns,
                self.commands[-1]["end_ns"] if self.commands else self.clock.now_ns,
            )
            self.commands.append(
                {"payload": deepcopy(payload), "start_ns": start, "end_ns": start + 100_000_000}
            )
            return {"ok": True, "accepted": True, "command_id": payload["command_id"]}
        if payload["op"] != "status":
            raise AssertionError(f"unexpected operation: {payload['op']}")
        status_payload = payload
        delay = self.status_delays[min(len(self.status_replies), len(self.status_delays) - 1)]
        self.clock.sleep(delay)
        completed, active = [], None
        live, gripper = targets_at(0)[0, :7].tolist(), 0.0
        for index, command in enumerate(self.commands):
            payload = command["payload"]
            if command["end_ns"] > self.clock.now_ns:
                if command["start_ns"] <= self.clock.now_ns:
                    active = command
                continue
            live, gripper = payload["target_pose"], payload["target_gripper"]
            if gripper <= -0.72 and self.release_event is None:
                self.release_event = {
                    "command_id": payload["command_id"],
                    "feedback_position": gripper,
                    "pose": live,
                    "monotonic_s": command["end_ns"] / 1e9,
                }
            completed.append(
                {
                    "command_id": payload["command_id"],
                    "accepted": True,
                    "ok": index != self.fail_index,
                    "execution_started_monotonic_ns": command["start_ns"],
                    "execution_finished_monotonic_ns": command["end_ns"],
                    "gripper_release_event": self.release_event,
                }
            )
        fatal = self.fatal_after is not None and self.clock.monotonic() >= self.fatal_after
        queue_depth = len(self.commands) - len(completed)
        if (
            status_payload.get("compact") is True
            and status_payload.get("after_command_id") is not None
        ):
            for index, receipt in enumerate(completed):
                if receipt["command_id"] == status_payload["after_command_id"]:
                    completed = completed[index + 1 :]
                    break
        response = {
            "ok": True,
            "ready": not fatal,
            "fatal_error": "fake bridge fault" if fatal else None,
            "queue_depth": queue_depth,
            "recent_results": completed,
            "active_command_id": None if active is None else active["payload"]["command_id"],
            "active_command_started_monotonic_ns": None if active is None else active["start_ns"],
            "live_pose": list(live),
            "live_pose_stale": False,
            "right_gripper": {
                "last_completed": gripper,
                "last_observation": {"raw_position": gripper},
            },
        }
        self.status_replies.append(deepcopy(response))
        return response


class FakeObserver:
    def __init__(self, clock, *, available=True, available_after_ns=0):
        self.clock, self.available = clock, available
        self.available_after_ns = available_after_ns
        self.calls = []

    def latest(self):
        self.calls.append(self.clock.now_ns)
        if not self.available or self.clock.now_ns < self.available_after_ns:
            return None
        metadata = {
            "snapshot_index": self.clock.now_ns,
            "capture_started_monotonic_ns": self.clock.now_ns,
            "capture_started_wall_ns": self.clock.time_ns(),
            "right_eef_tf_timestamp_ns": self.clock.time_ns(),
        }
        snapshot = types.SimpleNamespace(metadata=metadata)
        return snapshot, self.clock.monotonic() - 0.01, self.clock.monotonic()


class FakePlanner:
    """Deterministic model latency, retaining the production queue merge rules."""

    def __init__(self, clock, actions, *, delays=(0.3,), fail=False):
        self.clock, self.actions = clock, actions
        self.delays, self.fail = delays, fail
        self.token = None
        self.requests, self.results = [], []
        self.failure_ns = None

    @property
    def pending(self):
        return self.token is not None

    def request(self, observation, *, observation_step_index, frozen_steps, ramp_rate):
        self.token = self.actions.snapshot(frozen_steps, start_index=observation_step_index)
        delay = self.delays[min(len(self.requests), len(self.delays) - 1)]
        self.due_ns = self.clock.now_ns + round(delay * 1e9)
        self.request_ns = self.clock.now_ns
        request = {
            "start_index": self.token.start_index,
            "overlap_steps": self.token.overlap_steps,
            "frozen_steps": frozen_steps,
            "generation": self.token.generation,
        }
        self.requests.append(request)
        return request

    def poll(self):
        if self.token is None or self.clock.now_ns < self.due_ns:
            return None
        token, self.token = self.token, None
        if self.fail:
            self.actions.cancel(token)
            self.failure_ns = self.clock.now_ns
            raise RuntimeError("RTC prediction failed: fake model failure")
        metrics = self.actions.merge(token, targets_at(token.start_index, predicted=True))
        result = {
            "inference_s": (self.clock.now_ns - self.request_ns) / 1e9,
            "merge": metrics.to_dict(),
        }
        self.results.append(result)
        return result


class RtcPlaceStreamTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.actions = ActionQueue()
        self.initial_targets = targets_at(0)
        self.actions.initialize(self.initial_targets)
        self.client = FakeActionClient(self.clock)
        self.observer = FakeObserver(self.clock)
        self.planner = FakePlanner(self.clock, self.actions)
        self.progress = runner.PlacementProgress()
        self.report = {"executions": [], "predictions": [], "requests": [], "underruns": []}

    def run_stream(self, **options):
        with (
            mock.patch.object(runner, "time", self.clock),
            mock.patch.object(runner, "validate_snapshot") as validation,
            mock.patch.object(runner, "model_observation", return_value={}),
            redirect_stdout(io.StringIO()),
        ):
            runner.run_stream(
                self.client,
                self.actions,
                self.planner,
                self.observer,
                self.progress,
                self.report,
                42,
                prompt="fake place task",
                **options,
            )
        return validation

    def execution_calls(self):
        return [call for call in self.client.calls if call[2]["op"] == "execute_h1_gripper"]

    def assert_unique_ordered_rows(self):
        rows = self.report["executions"]
        self.assertEqual([row["index"] for row in rows], list(range(len(rows))))
        ids = [row["command_id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(self.execution_calls()), len(rows))

    def test_full_task_releases_and_retracts_with_rtc_without_replaying_rows(self):
        validation = self.run_stream()
        self.assertTrue(validation.called)
        self.assertEqual(self.report["status"], "PASS_PLACED_RELEASED_AND_RETRACTED")
        self.assertTrue(self.report["task_progress"]["passed"])
        self.assertGreaterEqual(self.report["task_progress"]["retraction_m"], 0.13)
        self.assertEqual(self.report["final_bridge_status"]["queue_depth"], 0)
        self.assertTrue(self.report["predictions"])
        self.assert_unique_ordered_rows()
        merge = self.report["predictions"][0]["merge"]
        replacement_start = merge["replacement_range"][0]
        sent = np.asarray([row["pose"] + [row["gripper"]] for row in self.report["executions"]])
        np.testing.assert_array_equal(
            sent[:replacement_start], self.initial_targets[:replacement_start]
        )
        expected = targets_at(replacement_start, predicted=True)
        np.testing.assert_array_equal(sent[replacement_start], expected[0])

    def test_dispatch_is_10hz_and_arm_and_gripper_share_one_socket_call(self):
        self.run_stream()
        calls = self.execution_calls()
        intervals = np.diff([call[0] for call in calls]) / 1e9
        self.assertTrue(np.all(intervals >= 0.098), intervals)
        self.assertTrue(np.all(intervals <= 0.102), intervals)
        self.assertGreater(len(calls), 16)
        self.assertEqual({call[1] for call in self.client.calls}, {threading.get_ident()})
        self.assertEqual(
            {call[2]["op"] for call in self.client.calls}, {"status", "execute_h1_gripper"}
        )
        for row, (sent_ns, _, payload) in zip(self.report["executions"], calls, strict=True):
            self.assertEqual(payload["target_pose"], row["pose"])
            self.assertEqual(payload["target_gripper"], row["gripper"])
            self.assertEqual(payload["timestamp_ns"], sent_ns + 1_700_000_000_000_000_000 + 42)
        grippers = [call[2]["target_gripper"] for call in calls]
        self.assertIn(0.0, grippers)
        self.assertIn(-0.123, grippers)
        self.assertIn(-0.312, grippers)
        self.assertIn(-0.785, grippers)

    def test_model_failure_stops_without_an_additional_submission(self):
        self.planner.fail = True
        with self.assertRaisesRegex(RuntimeError, "RTC prediction failed"):
            self.run_stream()
        self.assertTrue(self.execution_calls())
        self.assertTrue(all(call[0] < self.planner.failure_ns for call in self.execution_calls()))
        self.assertEqual(len(self.execution_calls()), self.actions.next_index)
        self.assertNotIn("status", self.report)

    def test_stale_prediction_keeps_original_future_then_retries_without_replay(self):
        self.planner.delays = (0.46, 0.1)
        self.run_stream(frozen_steps=4, request_remaining=8)
        self.assertEqual(self.report["status"], "PASS_PLACED_RELEASED_AND_RETRACTED")
        first, second = self.report["predictions"][:2]
        self.assertFalse(first["merge"]["accepted"])
        self.assertIsNone(first["merge"]["replacement_range"])
        self.assertGreater(first["merge"]["remaining"], 0)
        self.assertTrue(second["merge"]["accepted"])
        replace_from = second["merge"]["replacement_range"][0]
        sent = np.asarray([row["pose"] + [row["gripper"]] for row in self.report["executions"]])
        np.testing.assert_array_equal(sent[:replace_from], self.initial_targets[:replace_from])
        self.assert_unique_ordered_rows()

    def test_faulted_bridge_stops_before_next_submission(self):
        self.client.fatal_after = 100.55
        with self.assertRaisesRegex(RuntimeError, "bridge is not healthy"):
            self.run_stream()
        self.assertTrue(all(call[0] < 100_550_000_000 for call in self.execution_calls()))

    def test_rejected_ack_stops_and_marks_only_attempted_row_failed(self):
        self.client.reject_index = 4
        with self.assertRaisesRegex(RuntimeError, "command rejected"):
            self.run_stream()
        self.assertEqual(len(self.execution_calls()), 5)
        self.assertEqual(self.report["executions"][-1]["status"], "FAILED")

    def test_failed_completion_stops_before_further_dispatch(self):
        self.client.fail_index = 3
        with self.assertRaisesRegex(RuntimeError, "execution failed"):
            self.run_stream()
        failure_end = self.client.commands[3]["end_ns"]
        self.assertTrue(all(call[0] < failure_end + 40_000_000 for call in self.execution_calls()))
        self.assertNotIn("status", self.report)

    def test_missing_observations_exhausts_seed_without_replay(self):
        self.observer.available = False
        with self.assertRaisesRegex(RuntimeError, "queue exhausted"):
            self.run_stream()
        self.assertEqual(len(self.execution_calls()), 16)
        self.assertEqual(self.report["predictions"], [])
        self.assert_unique_ordered_rows()

    def test_result_arriving_after_full_queue_exhaustion_is_not_replayed(self):
        self.planner.delays = (1.5,)
        with self.assertRaisesRegex(RuntimeError, "queue exhausted"):
            self.run_stream()
        self.assertEqual(len(self.execution_calls()), 16)
        self.assertTrue(self.report["underruns"])
        self.assertFalse(self.report["predictions"][0]["merge"]["accepted"])
        self.assertIsNone(self.report["predictions"][0]["merge"]["replacement_range"])
        self.assert_unique_ordered_rows()

    def test_stale_result_recovers_with_fresh_short_prefix_and_no_replay(self):
        self.planner.delays = (1.0, 0.1)
        self.run_stream()
        self.assertEqual(self.report["status"], "PASS_PLACED_RELEASED_AND_RETRACTED")
        first, second = self.report["predictions"][:2]
        self.assertFalse(first["merge"]["accepted"])
        self.assertIsNone(first["merge"]["replacement_range"])
        self.assertTrue(second["merge"]["accepted"])
        self.assertLess(self.report["requests"][1]["frozen_steps"], 8)
        self.assertGreater(
            self.report["requests"][1]["snapshot_index"],
            self.report["requests"][0]["snapshot_index"],
        )
        self.assert_unique_ordered_rows()

    def test_first_observation_with_less_than_eight_overlap_rows_still_replans(self):
        self.observer.available_after_ns = 101_000_000_000
        self.run_stream()
        first_request = self.report["requests"][0]
        self.assertLess(first_request["overlap_steps"], 8)
        self.assertGreater(first_request["overlap_steps"], 0)
        self.assertEqual(first_request["frozen_steps"], first_request["overlap_steps"])
        self.assertGreaterEqual(first_request["snapshot_index"], self.observer.available_after_ns)
        self.assertEqual(self.report["status"], "PASS_PLACED_RELEASED_AND_RETRACTED")
        self.assertTrue(self.report["predictions"][0]["merge"]["accepted"])
        self.assert_unique_ordered_rows()

    def test_status_rpc_delay_uses_pre_rpc_frame_and_does_not_starve_predictions(self):
        self.client.status_delays = (0.08,)
        self.run_stream()
        self.assertEqual(self.report["status"], "PASS_PLACED_RELEASED_AND_RETRACTED")
        self.assertTrue(self.report["requests"])
        self.assertTrue(self.report["predictions"])
        status_calls = [call for call in self.client.calls if call[2]["op"] == "status"]
        requested_ns = {call[0] for call in status_calls}
        for request in self.report["requests"]:
            # The latest() fake returns a newly captured frame on every call;
            # fetching after the RPC would produce a strictly newer timestamp.
            self.assertIn(request["snapshot_index"], requested_ns)
            self.assertGreaterEqual(request["snapshot_received_age_s"], 0.08 - 1e-9)
        self.assertTrue(
            all(abs(item["roundtrip_s"] - 0.08) < 1e-9 for item in self.report["status_requests"])
        )
        self.assert_unique_ordered_rows()

    def test_slow_first_status_does_not_compress_first_two_action_submissions(self):
        self.client.status_delays = (0.25, 0.0)
        self.run_stream()
        calls = self.execution_calls()
        self.assertEqual(calls[0][0], 100_250_000_000)
        self.assertGreaterEqual(calls[1][0] - calls[0][0], 100_000_000)
        self.assertTrue(np.all(np.diff([call[0] for call in calls]) >= 100_000_000))
        self.assertEqual(self.report["status"], "PASS_PLACED_RELEASED_AND_RETRACTED")

    def test_compact_cursor_advances_to_last_consumed_receipt(self):
        self.run_stream()
        status_calls = [call for call in self.client.calls if call[2]["op"] == "status"]
        self.assertEqual(len(status_calls), len(self.client.status_replies))
        expected_cursor = None
        delivered = []
        for (_, _, payload), reply in zip(status_calls, self.client.status_replies, strict=True):
            self.assertIs(payload.get("compact"), True)
            self.assertEqual(payload.get("after_command_id"), expected_cursor)
            results = reply["recent_results"]
            delivered.extend(result["command_id"] for result in results)
            if results:
                expected_cursor = results[-1]["command_id"]
        self.assertIsNotNone(expected_cursor)
        self.assertEqual(len(delivered), len(set(delivered)))
        self.assertEqual(delivered, [row["command_id"] for row in self.report["executions"]])
        self.assertTrue(all(row["status"] == "COMPLETED" for row in self.report["executions"]))


if __name__ == "__main__":
    unittest.main()
