#!/usr/bin/env python3
"""Single-process lifecycle/queue checks; no GDK, GPU, or physical commands."""

from contextlib import ExitStack
import importlib
import json
import socket
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np


sys.modules.setdefault("agibot_gdk", types.ModuleType("agibot_gdk"))
bridge = importlib.import_module("g2_groot_continuous_action_bridge")
from g2_groot_gripper_state_machine import GripperObservation  # noqa: E402


class FakeRobot:
    def __init__(self):
        self.pose = np.asarray([0.5, -0.2, 1.05, 0, 0, 0, 1.0])
        self.gripper = 0.0
        self.events = []
        self.fail_publication = False
        self.fail_feedback = False
        self.fail_calibration = False
        self.force_n = 0.0
        self.control_mode = 3
        self.recovery_ms = 500

    def get_collision_detection_config(self):
        return types.SimpleNamespace(is_enabled=True, sensitivity=3, checkout_timeout_ms=self.recovery_ms)

    def get_motion_control_status(self):
        from test_g2_groot_contact_guard import motion

        state = motion(force=self.force_n)
        state.mode, state.control_mode = 1, self.control_mode
        return state

    def get_tf_from_base_link(self, frame):
        return types.SimpleNamespace(rotation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))

    def set_reference_frame_poses(self, *quaternions):
        self.events.append(("references", quaternions, threading.get_ident()))
        return 0

    def trajectory_tracking_control(self, timestamp, states, actions, **kwargs):
        # Instantaneous test plant, not a model of actual robot dynamics.
        action = actions[0]
        self.pose = np.asarray(action["right_arm"]["action_data"]).copy()
        self.gripper = action["right_effector"]["action_data"][0]
        self.events.append(("publish", self.pose.tolist(), self.gripper, threading.get_ident()))
        return 0


class FakeSender:
    def __init__(self, initial_gripper, before_publish=None, pose_guard=None):
        self.before_publish = before_publish
        self.pose_guard = pose_guard
        self.target = self.current = self.start = initial_gripper
        self.index = self.ticks = 1
        self.robot = None

    def initialize_references(self, robot, tf):
        self.robot = robot
        robot.events.append(("references", threading.get_ident()))

    def set_target(self, target, ticks=5):
        self.start, self.target = self.current, target
        self.index, self.ticks = 0, ticks
        self.robot.events.append(("gripper_row", target, threading.get_ident()))

    def __call__(self, robot, pose):
        self.before_publish()
        if self.pose_guard is not None:
            self.pose_guard(pose)
        if robot.fail_publication:
            raise RuntimeError("simulated GDK publication failure")
        self.index = min(self.index + 1, self.ticks)
        self.current = (
            self.target
            if self.index == self.ticks
            else self.start + (self.target - self.start) * self.index / self.ticks
        )
        robot.pose = np.asarray(pose).copy()
        robot.gripper = self.current
        robot.events.append(("publish", robot.pose.tolist(), self.current, threading.get_ident()))
        return 0

    def snapshot(self):
        return {"requested_target": self.target, "accepted_target": self.current, "fault": None}


class FakeController:
    def __init__(self, robot, tf, *, sender, workspace_min, workspace_max, required_motion_mode):
        self.robot, self.sender = robot, sender
        self.workspace_min, self.workspace_max = workspace_min, workspace_max
        self.desired = robot.pose.copy()
        self.calibrated = False

    def tick(self, pose):
        self.sender(self.robot, pose)
        self.desired = np.asarray(pose).copy()
        time.sleep(0.001)

    def calibrate(self, duration_s):
        if self.robot.fail_calibration:
            raise RuntimeError("simulated calibration failure")
        self.robot.events.append(("calibrate", threading.get_ident()))
        self.tick(self.desired)
        self.calibrated = True
        return {"test": True}

    def move_to(self, target, duration_s):
        self.robot.events.append(("row", target.tolist(), threading.get_ident()))
        start = self.desired.copy()
        for index in range(1, 6):
            self.tick(start + (target - start) * index / 5)
        return {"ticks": 5}


def read_feedback(robot, joint_name):
    if robot.fail_feedback:
        raise RuntimeError("simulated motor feedback fault")
    return GripperObservation(time.monotonic(), robot.gripper, 0, 0, 0, 0.0)


class ContinuousServiceTest(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot()
        self.patches = [
            mock.patch.object(bridge, "pose_values", side_effect=lambda tf, frame: tf.pose.copy()),
            mock.patch.object(bridge, "read_right_gripper_observation", side_effect=read_feedback),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.service = bridge.ContinuousActionService(
            self.robot,
            self.robot,
            initial_gripper_command=0.0,
            workspace_min=[0.1, -0.5, 0.7],
            workspace_max=[0.9, 0.1, 1.5],
            enable_control=True,
            controller_factory=FakeController,
            sender_factory=FakeSender,
        )
        self.addCleanup(self.service.stop)
        self.owner = object()

    def activate(self):
        return self.service.activate({"confirm": bridge.ACTIVATION_CONFIRMATION}, self.owner)

    def row(self, index=0, gripper=-0.3):
        return {
            "op": "execute_h1_gripper",
            "command_id": f"test-row-{index}",
            "timestamp_ns": time.time_ns(),
            "target_pose": [0.5 + index * 0.0002, -0.2, 1.05, 0, 0, 0, 1.0],
            "target_gripper": gripper,
        }

    def wait_for(self, condition):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.002)
        self.fail("test condition timed out")

    def test_pending_contact_limits_prevent_worker_start(self):
        from g2_groot_contact_guard import ContactGuard
        from test_g2_groot_contact_guard import synthetic_config

        self.service.contact_guard = ContactGuard({**synthetic_config(), "confirmed": False})
        with self.assertRaisesRegex(ValueError, "site confirmation"):
            self.activate()
        self.assertIsNone(self.service.thread)
        self.assertEqual(self.robot.events, [])

    def test_native_collision_short_window_rejects_before_worker_start(self):
        self.service.native_collision_latch = bridge.NativeCollisionLatch()
        self.robot.recovery_ms = 5
        with self.assertRaisesRegex(RuntimeError, "at least 500"):
            self.activate()
        self.assertIsNone(self.service.thread)
        self.assertEqual(self.robot.events, [])

    def test_native_collision_during_chunk_discards_queue_and_never_resumes(self):
        self.service.native_collision_latch = bridge.NativeCollisionLatch()
        self.activate()
        original_status = self.robot.get_motion_control_status

        def collide_on_first_row():
            state = original_status()
            if any(event[0] == "row" for event in self.robot.events):
                state.control_mode = 1
            return state

        with mock.patch.object(self.robot, "get_motion_control_status", side_effect=collide_on_first_row):
            self.service.execute_chunk({"commands": [self.row(i) for i in range(16)]}, self.owner)
            self.wait_for(lambda: self.service.fatal_error is not None)
            self.service.stop()
        self.robot.control_mode = 3
        results = self.service.status()["recent_results"]
        self.assertEqual(len(results), 16)
        self.assertTrue(all(result["ok"] is False for result in results))
        self.assertEqual(self.service.status()["queue_depth"], 0)
        self.assertEqual(self.robot.gripper, 0.0)
        event_count = len(self.robot.events)
        with self.assertRaisesRegex(RuntimeError, "cannot activate"):
            self.activate()
        self.assertEqual(len(self.robot.events), event_count)
        self.assertTrue(self.service.native_collision_latch.snapshot()["fault"])

    def test_unsafe_last_z_rejects_entire_chunk_without_any_row_execution(self):
        from g2_groot_contact_guard import ContactGuard
        from test_g2_groot_contact_guard import synthetic_config

        self.service.contact_guard = ContactGuard(synthetic_config())
        self.activate()
        rows = [self.row(i) for i in range(16)]
        rows[-1]["target_pose"][2] = 0.99
        with self.assertRaisesRegex(RuntimeError, "below confirmed EEF floor"):
            self.service.execute_chunk({"commands": rows}, self.owner)
        self.service.stop()
        self.assertEqual([event for event in self.robot.events if event[0] == "row"], [])
        self.assertTrue(self.service.commands.empty())
        self.assertTrue(self.service.contact_guard.snapshot()["fault"])

    def test_contact_trip_stops_further_publication_without_auto_release(self):
        from g2_groot_contact_guard import ContactGuard
        from test_g2_groot_contact_guard import synthetic_config

        self.service.contact_guard = ContactGuard(synthetic_config())
        self.activate()
        self.robot.force_n = 100.0
        self.wait_for(lambda: self.service.fatal_error is not None)
        self.service.stop()
        count = len(self.robot.events)
        self.assertEqual(self.service.activation_state, "fault")
        self.assertEqual(self.robot.gripper, 0.0)
        with self.assertRaises(RuntimeError):
            self.service.execute(self.row(gripper=-0.785), self.owner)
        self.assertEqual(len(self.robot.events), count)

    def test_batch_preserves_sixteen_rows_and_gripper_targets(self):
        self.activate()
        rows = [self.row(i, -0.785 * i / 15) for i in range(16)]
        ack = self.service.execute_chunk({"commands": rows}, self.owner)
        self.assertEqual(ack["command_ids"], [row["command_id"] for row in rows])
        self.wait_for(lambda: self.service.last_command_id == rows[-1]["command_id"])
        receipts = self.service.status()["recent_results"]
        self.assertEqual(len(receipts), 16)
        for command, receipt in zip(rows, receipts, strict=True):
            self.assertTrue(receipt["ok"])
            self.assertEqual(receipt["ticks"], 5)
            np.testing.assert_array_equal(receipt["target_pose"], command["target_pose"])
            self.assertEqual(receipt["gripper_target"], command["target_gripper"])

    def test_contact_trip_marks_pending_rows_failed_without_executing_them(self):
        from g2_groot_contact_guard import ContactGuard
        from test_g2_groot_contact_guard import motion, synthetic_config

        self.service.contact_guard = ContactGuard(synthetic_config())
        self.activate()

        def contact_on_first_row():
            in_row = any(event[0] == "row" for event in self.robot.events)
            return motion(force=100.0 if in_row else 0.0)

        with mock.patch.object(self.robot, "get_motion_control_status", side_effect=contact_on_first_row):
            self.service.execute_chunk({"commands": [self.row(i) for i in range(16)]}, self.owner)
            self.wait_for(lambda: self.service.fatal_error is not None)
            self.service.stop()
        results = self.service.status()["recent_results"]
        self.assertEqual(len(results), 16)
        self.assertTrue(all(result["ok"] is False for result in results))
        self.assertEqual(len([event for event in self.robot.events if event[0] == "row"]), 1)
        self.assertEqual(self.robot.gripper, 0.0)
        self.assertEqual(self.service.status()["queue_depth"], 0)

    def test_batch_invalid_last_row_enqueues_nothing(self):
        self.activate()
        rows = [self.row(i) for i in range(16)]
        rows[-1]["target_gripper"] = float("nan")
        before = self.service.last_accepted_pose.copy()
        with self.assertRaises(ValueError):
            self.service.execute_chunk({"commands": rows}, self.owner)
        self.service.stop()
        self.assertTrue(self.service.commands.empty())
        self.assertEqual(self.service.seen_command_ids, set())
        np.testing.assert_array_equal(self.service.last_accepted_pose, before)
        self.assertEqual([event for event in self.robot.events if event[0] == "row"], [])

    def test_batch_checks_steps_against_previous_row(self):
        self.activate()
        rows = [self.row(i) for i in range(16)]
        rows[5]["target_pose"][0] = 0.89
        with self.assertRaisesRegex(ValueError, "translation step"):
            self.service.execute_chunk({"commands": rows}, self.owner)
        self.assertTrue(self.service.commands.empty())

    def test_batch_duplicate_ids_enqueues_nothing(self):
        self.activate()
        rows = [self.row(i) for i in range(16)]
        rows[-1]["command_id"] = rows[0]["command_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.service.execute_chunk({"commands": rows}, self.owner)
        self.assertTrue(self.service.commands.empty())

    def test_batch_partial_chunk_is_rejected(self):
        self.activate()
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            self.service.execute_chunk({"commands": [self.row()]}, self.owner)
        self.assertTrue(self.service.commands.empty())

    def test_batch_requires_active_owner(self):
        rows = [self.row(i) for i in range(16)]
        with self.assertRaisesRegex(RuntimeError, "active owner"):
            self.service.execute_chunk({"commands": rows}, self.owner)
        self.assertEqual(self.robot.events, [])
        self.activate()
        with self.assertRaisesRegex(RuntimeError, "active owner"):
            self.service.execute_chunk({"commands": rows}, object())
        self.assertTrue(self.service.commands.empty())

    def test_batch_cannot_append_while_busy(self):
        self.activate()
        # Hold the service lock so the worker cannot drain the first batch.
        with self.service.lock:
            self.service.execute_chunk({"commands": [self.row(i) for i in range(16)]}, self.owner)
            with self.assertRaisesRegex(RuntimeError, "idle command queue"):
                self.service.execute_chunk(
                    {"commands": [self.row(i + 16) for i in range(16)]}, self.owner
                )
            self.assertEqual(len(self.service.seen_command_ids), 16)

    def test_standby_is_read_only_and_short_connection_does_not_stop_it(self):
        reader = object()
        self.assertEqual(self.service.info()["activation_state"], "standby")
        self.assertFalse(self.service.status()["arm_owner_active"])
        self.assertFalse(self.service.status()["ready"])
        self.service.disconnect(reader)
        self.assertEqual(self.robot.events, [])
        self.assertIsNone(self.service.sender)
        self.assertFalse(self.service.stop_event.is_set())

    def test_control_flag_and_confirmation_required(self):
        with self.assertRaises(ValueError):
            self.service.activate({}, self.owner)
        self.service.enable_control = False
        with self.assertRaisesRegex(RuntimeError, "read-only"):
            self.activate()
        self.assertEqual(self.robot.events, [])

    def seed_diagnostic_results(self, count=3):
        with self.service.lock:
            for index in range(count):
                result = {
                    "command_id": f"receipt-{index}",
                    "accepted": True,
                    "ok": True,
                    "ticks": 5,
                    "execution_started_monotonic_ns": 1_000_000_000 + index * 100_000_000,
                    "execution_finished_monotonic_ns": 1_100_000_000 + index * 100_000_000,
                    "gripper_release_event": None,
                    "live_pose_at_100ms": self.robot.pose.tolist(),
                    "gripper_target": -0.3,
                    "native_trajectory": {"reference": [list(range(64))] * 50},
                    "gripper_status": {"diagnostic": "full"},
                }
                self.service.recent_results.append(result)
                self.service.last_result = result
                self.service.last_command_id = result["command_id"]
            self.service.calibration = {"reference": [list(range(64))] * 50}

    def test_compact_status_removes_heavy_payload_and_filters_after_cursor(self):
        self.seed_diagnostic_results()
        compact = self.service.status(compact=True, after_command_id="receipt-0")
        self.assertNotIn("calibration", compact)
        self.assertNotIn("last_result", compact)
        self.assertEqual(
            [r["command_id"] for r in compact["recent_results"]], ["receipt-1", "receipt-2"]
        )
        for result in compact["recent_results"]:
            self.assertEqual(set(result), set(bridge.COMPACT_RESULT_FIELDS) - {"error"})
            self.assertNotIn("native_trajectory", result)
            self.assertNotIn("gripper_status", result)
        for required in (
            "ready",
            "fatal_error",
            "live_pose",
            "live_pose_stale",
            "active_command_id",
            "active_command_started_monotonic_ns",
            "queue_depth",
            "right_gripper",
        ):
            self.assertIn(required, compact)
        full = self.service.status()
        self.assertGreater(len(json.dumps(full)), 10 * len(json.dumps(compact)))
        self.assertIn("native_trajectory", full["recent_results"][0])
        self.assertEqual(len(full["recent_results"]), 3)

    def test_default_status_ignores_cursor_and_preserves_full_diagnostics(self):
        self.seed_diagnostic_results()
        now = time.monotonic()
        with mock.patch.object(bridge.time, "monotonic", return_value=now):
            full = self.service.status()
            self.assertEqual(full, self.service.status(after_command_id="receipt-1"))
            self.assertEqual(full, self.service.status(compact=False, after_command_id="receipt-1"))
        self.assertEqual(full["calibration"], self.service.calibration)
        self.assertEqual(full["last_result"], self.service.last_result)
        self.assertIn("native_trajectory", full["recent_results"][-1])

    def test_compact_unknown_cursor_returns_retained_32_and_latest_returns_empty(self):
        self.seed_diagnostic_results(40)
        for cursor in (None, "unknown", "receipt-0"):
            compact = self.service.status(compact=True, after_command_id=cursor)
            self.assertEqual(len(compact["recent_results"]), 32)
            self.assertEqual(compact["recent_results"][0]["command_id"], "receipt-8")
        self.assertEqual(
            self.service.status(compact=True, after_command_id="receipt-39")["recent_results"],
            [],
        )
        with self.service.lock:
            self.service.recent_results[-1].update(ok=False, error="example execution failure")
        failure = self.service.status(compact=True, after_command_id="receipt-38")[
            "recent_results"
        ][0]
        self.assertFalse(failure["ok"])
        self.assertEqual(failure["error"], "example execution failure")

    def test_execution_telemetry_covers_the_entire_row_and_clears_when_idle(self):
        entered, release = threading.Event(), threading.Event()
        move_times = {}

        class BlockingController(FakeController):
            def move_to(self, target, duration_s):
                move_times["started"] = time.monotonic_ns()
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("test release timed out")
                outcome = super().move_to(target, duration_s)
                move_times["finished"] = time.monotonic_ns()
                return outcome

        self.service.controller_factory = BlockingController
        before = self.service.status()
        self.assertIsNone(before["active_command_id"])
        self.assertIsNone(before["active_command_started_monotonic_ns"])
        self.activate()
        self.service.execute(self.row(), self.owner)
        try:
            self.assertTrue(entered.wait(2))
            active = self.service.status()
            self.assertEqual(active["active_command_id"], "test-row-0")
            started_ns = active["active_command_started_monotonic_ns"]
            self.assertIsInstance(started_ns, int)
            self.assertGreater(started_ns, 0)
            self.assertLessEqual(started_ns, move_times["started"])
        finally:
            release.set()
        self.wait_for(lambda: self.service.status()["last_command_id"] == "test-row-0")
        completed = self.service.status()
        self.assertIsNone(completed["active_command_id"])
        self.assertIsNone(completed["active_command_started_monotonic_ns"])
        result = completed["last_result"]
        self.assertTrue(result["ok"])
        self.assertEqual(result["ticks"], 5)
        self.assertEqual(result["execution_started_monotonic_ns"], started_ns)
        self.assertGreaterEqual(result["execution_finished_monotonic_ns"], move_times["finished"])
        self.assertLessEqual(result["execution_finished_monotonic_ns"], time.monotonic_ns())

    def test_failed_active_row_has_times_but_unexecuted_pending_row_does_not(self):
        entered, release = threading.Event(), threading.Event()

        class FailingController(FakeController):
            def move_to(self, target, duration_s):
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("test release timed out")
                raise RuntimeError("simulated row failure")

        self.service.controller_factory = FailingController
        self.activate()
        self.service.execute(self.row(0), self.owner)
        try:
            self.assertTrue(entered.wait(2))
            self.service.execute(self.row(1), self.owner)
        finally:
            release.set()
        self.wait_for(lambda: len(self.service.status()["recent_results"]) == 2)
        completed = self.service.status()
        active_result, pending_result = completed["recent_results"]
        self.assertFalse(active_result["ok"])
        self.assertGreater(active_result["execution_started_monotonic_ns"], 0)
        self.assertGreaterEqual(
            active_result["execution_finished_monotonic_ns"],
            active_result["execution_started_monotonic_ns"],
        )
        self.assertIsNone(pending_result["execution_started_monotonic_ns"])
        self.assertIsNone(pending_result["execution_finished_monotonic_ns"])
        self.assertIsNone(completed["active_command_id"])
        self.assertIsNone(completed["active_command_started_monotonic_ns"])

    def test_activate_then_shutdown_uses_one_worker_and_no_extra_tool_command(self):
        self.assertEqual(self.activate()["activation_state"], "active")
        self.assertTrue(self.service.status()["ready"])
        self.assertTrue(self.service.status()["arm_owner_active"])
        self.activate()  # Same-connection idempotence, not a restart.
        self.service.disconnect(self.owner, shutdown=True)
        count = len(self.robot.events)
        time.sleep(0.01)
        self.assertEqual(len(self.robot.events), count)
        self.assertEqual(self.service.activation_state, "stopped")
        self.assertIsNone(self.service.fatal_error)
        self.assertEqual(len([e for e in self.robot.events if e[0] == "calibrate"]), 1)
        ids = {e[-1] for e in self.robot.events}
        self.assertEqual(len(ids), 1)
        self.assertNotIn(threading.get_ident(), ids)
        self.assertEqual(self.robot.gripper, 0.0)

    def test_disconnect_fault_latches_and_cannot_reactivate(self):
        self.activate()
        self.service.disconnect(self.owner)
        self.assertEqual(self.service.activation_state, "fault")
        self.assertFalse(self.service.status()["arm_owner_active"])
        with self.assertRaises(RuntimeError):
            self.activate()
        with self.assertRaises(RuntimeError):
            self.service.execute(self.row(), self.owner)

    def test_foreign_owner_cannot_execute_or_take_over(self):
        self.activate()
        with self.assertRaises(RuntimeError):
            self.service.activate({"confirm": bridge.ACTIVATION_CONFIRMATION}, object())
        with self.assertRaises(RuntimeError):
            self.service.execute(self.row(), object())
        self.service.disconnect(object())
        self.assertEqual(self.service.activation_state, "active")

    def test_sixteen_rows_preserve_partial_targets_and_actual_feedback_receipts(self):
        self.activate()
        targets = [float(x) for x in np.linspace(-0.03, -0.785, 16)]
        for index, target in enumerate(targets):
            ack = self.service.execute(self.row(index, target), self.owner)
            self.assertTrue(ack["queued"])
        self.wait_for(lambda: self.service.status()["last_command_id"] == "test-row-15")
        status = self.service.status()
        self.assertEqual(status["queue_depth"], 0)
        self.assertEqual(len(status["recent_results"]), 16)
        self.assertEqual([r["gripper_target"] for r in status["recent_results"]], targets)
        self.assertTrue(all(r["ticks"] == 5 and r["ok"] for r in status["recent_results"]))
        self.assertEqual([e[1] for e in self.robot.events if e[0] == "gripper_row"], targets)
        self.assertEqual(len([e for e in self.robot.events if e[0] == "row"]), 16)
        self.assertAlmostEqual(status["right_gripper"]["last_observation"]["raw_position"], -0.785)
        event = status["gripper_release_event"]
        self.assertLessEqual(event["feedback_position"], bridge.OPEN_THRESHOLD)
        self.assertTrue(event["command_id"].startswith("test-row-"))
        self.assertEqual(event["time_basis"], "local_feedback_read_not_sensor_timestamp")
        self.assertIsNotNone(status["recent_results"][-1]["gripper_release_event"])

    def test_idle_keeps_last_gripper_target_without_replaying_policy_row(self):
        self.activate()
        self.service.execute(self.row(gripper=-0.312), self.owner)
        self.wait_for(lambda: self.service.status()["last_command_id"] == "test-row-0")
        publishes_before = len([e for e in self.robot.events if e[0] == "publish"])
        self.wait_for(
            lambda: len([e for e in self.robot.events if e[0] == "publish"]) > publishes_before + 3
        )
        self.assertAlmostEqual(self.robot.gripper, -0.312)
        self.assertEqual(len([e for e in self.robot.events if e[0] == "row"]), 1)

    def test_release_in_idle_is_associated_with_last_gripper_command(self):
        self.activate()
        self.service.execute(self.row(gripper=-0.75), self.owner)
        self.wait_for(lambda: self.service.status()["last_command_id"] == "test-row-0")
        self.service.stop()
        with self.service.lock:
            self.service.release_event = None
            self.service.last_observation = GripperObservation(time.monotonic(), -0.6, 0, 0, 0, 0)
        self.robot.gripper = -0.75
        self.service._poll_feedback()
        self.assertEqual(self.service.release_event["command_id"], "test-row-0")

    def test_calibration_failure_never_becomes_ready(self):
        self.robot.fail_calibration = True
        with self.assertRaisesRegex(RuntimeError, "calibration failure"):
            self.activate()
        self.service.stop()
        self.assertEqual(self.service.activation_state, "fault")
        self.assertFalse(self.service.status()["ready"])
        self.assertFalse(any(e[0] == "publish" for e in self.robot.events))

    def test_publication_failure_stops_and_rejects_following_commands(self):
        self.activate()
        self.robot.fail_publication = True
        self.wait_for(lambda: self.service.fatal_error is not None)
        self.service.stop()
        self.assertFalse(self.service.status()["ready"])
        with self.assertRaises(RuntimeError):
            self.service.execute(self.row(), self.owner)

    def test_feedback_failure_stops_before_next_publication(self):
        self.activate()
        self.robot.fail_feedback = True
        self.wait_for(lambda: self.service.fatal_error is not None)
        self.service.stop()
        count = len(self.robot.events)
        time.sleep(0.01)
        self.assertEqual(len(self.robot.events), count)
        self.assertIn("feedback fault", self.service.fatal_error)

    def test_invalid_or_duplicate_row_faults_without_a_second_execution(self):
        self.activate()
        row = self.row()
        self.service.execute(row, self.owner)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.service.execute(row, self.owner)
        self.service.stop()
        self.assertEqual(self.service.activation_state, "fault")
        self.assertLessEqual(len([e for e in self.robot.events if e[0] == "row"]), 1)

    def test_tcp_short_probe_then_persistent_activation_and_owner_eof(self):
        def run_client(messages):
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            errors = []

            def server():
                try:
                    connection, _ = listener.accept()
                    bridge.serve_connection(self.service, connection, time.monotonic() + 3)
                except Exception as error:
                    errors.append(error)

            thread = threading.Thread(target=server)
            thread.start()
            replies = []
            with socket.create_connection(listener.getsockname(), timeout=3) as client:
                with client.makefile("rb") as stream:
                    for payload in messages:
                        client.sendall((json.dumps(payload) + "\n").encode())
                        replies.append(json.loads(stream.readline()))
            thread.join(timeout=3)
            listener.close()
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            return replies

        self.seed_diagnostic_results()
        probe = run_client(
            [
                {"op": "info"},
                {"op": "status"},
                {"op": "status", "compact": True, "after_command_id": "receipt-1"},
            ]
        )
        self.assertTrue(all(r["activation_state"] == "standby" for r in probe))
        self.assertIn("native_trajectory", probe[1]["recent_results"][0])
        self.assertEqual([r["command_id"] for r in probe[2]["recent_results"]], ["receipt-2"])
        self.assertNotIn("native_trajectory", probe[2]["recent_results"][0])
        self.assertNotIn("calibration", probe[2])
        self.assertEqual(self.robot.events, [])
        replies = run_client(
            [
                {"op": "activate", "confirm": bridge.ACTIVATION_CONFIRMATION},
                {"op": "status"},
            ]
        )
        self.assertTrue(replies[0]["ok"])
        self.assertTrue(replies[1]["ready"])
        self.assertEqual(self.service.activation_state, "fault")
        self.assertFalse(self.service.status()["arm_owner_active"])

    def test_production_controller_sender_tcp_and_runner_four_h16_chunks(self):
        self._production_controller_sender_tcp_and_runner_four_h16_chunks(batch=False)

    def test_production_controller_sender_tcp_and_runner_batch_four_h16_chunks(self):
        self._production_controller_sender_tcp_and_runner_four_h16_chunks(batch=True)

    def _production_controller_sender_tcp_and_runner_four_h16_chunks(self, *, batch):
        from agibot.scripts import run_g2_groot_full_protected_inference as runner
        from agibot.scripts.run_g2_groot_full_place_inference import (
            PlacementBridgeSession,
            PlacementProgress,
            inspect_standby_bridge,
            validate_activation_state,
        )
        import g2_groot_continuous_controller as controller_module
        import g2_groot_continuous_sender as sender_module
        import g2_groot_persistent_right_arm_controller as legacy

        initial = self.robot.pose.copy()
        self.service = bridge.ContinuousActionService(
            self.robot,
            self.robot,
            initial_gripper_command=0.0,
            workspace_min=[0.1, -0.5, 0.7],
            workspace_max=[0.9, 0.1, 1.5],
            enable_control=True,
            calibration_duration_s=0.2,
            native_collision_latch=bridge.NativeCollisionLatch() if batch else None,
            freeze_compensation_after_calibration=batch,
        )
        self.addCleanup(self.service.stop)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(2)
        listener.settimeout(10)
        port = listener.getsockname()[1]
        server_errors = []

        def server():
            try:
                stopping = False
                while not stopping:
                    connection, _ = listener.accept()
                    stopping = bridge.serve_connection(
                        self.service,
                        connection,
                        time.monotonic() + 15,
                    )
            except Exception as error:
                server_errors.append(error)

        targets = np.tile(np.append(initial, 0.0), (64, 1))
        targets[:, 0] += np.arange(64) * 0.0002
        targets[:, 7] = np.concatenate(
            [
                np.linspace(-0.01, -0.3, 16),
                np.linspace(-0.3, -0.785, 16),
                np.linspace(-0.785, -0.2, 16),
                np.linspace(-0.2, -0.785, 16),
            ]
        )
        row_calls, gripper_calls = [], []
        actual_move = controller_module.ContinuousRightArmController.move_to
        actual_target = sender_module.NativeTrajectorySender.set_target

        def move(controller, target, duration_s):
            before = controller.tick_count
            result = actual_move(controller, target, duration_s)
            row_calls.append((target.copy(), controller.tick_count - before))
            return result

        def set_target(sender, target, ticks=5):
            gripper_calls.append(target)
            return actual_target(sender, target, ticks=ticks)

        def pose(tf, frame):
            return (initial if frame == legacy.LEFT_FRAME else self.robot.pose).copy()

        thread = threading.Thread(target=server, daemon=True)
        rows = []
        progress = PlacementProgress(minimum_retraction_m=0.001)
        with ExitStack() as stack:
            # Hot-path stubs need no mock call history. Recording every 50 Hz
            # read retains many cyclic mock objects across these real-clock
            # tests and can cause collection pauses longer than a control tick.
            for module in (controller_module, legacy):
                stack.enter_context(mock.patch.object(module, "pose_values", new=pose))
            stack.enter_context(mock.patch.object(controller_module, "snapshot", new=lambda *a, **k: {}))
            stack.enter_context(mock.patch.object(controller_module, "require_safe_state", new=lambda *a, **k: None))
            stack.enter_context(mock.patch.object(legacy, "emit", new=lambda *a, **k: None))
            stack.enter_context(
                mock.patch.object(
                    controller_module.ContinuousRightArmController,
                    "move_to",
                    autospec=True,
                    side_effect=move,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    sender_module.NativeTrajectorySender,
                    "set_target",
                    autospec=True,
                    side_effect=set_target,
                )
            )
            thread.start()
            try:
                self.assertEqual(
                    inspect_standby_bridge("127.0.0.1", port)["activation_state"], "standby"
                )
                self.assertEqual(self.robot.events, [])
                with PlacementBridgeSession("127.0.0.1", port) as client:
                    response = client.request(
                        {
                            "op": "activate",
                            "confirm": bridge.ACTIVATION_CONFIRMATION,
                        }
                    )
                    self.assertTrue(response["ok"], response)
                    validate_activation_state(
                        client.request({"op": "info"}),
                        client.request({"op": "status"}),
                        "active",
                    )
                    for cycle in range(4):
                        completion_status = {}
                        try:
                            _, completed = runner.execute_action_chunk(
                                client,
                                targets[cycle * 16 : (cycle + 1) * 16],
                                f"tcp-cycle-{cycle}",
                                0,
                                native_chunk_submission=batch,
                                completion_status_out=completion_status,
                            )
                        except Exception as error:
                            raise AssertionError(
                                f"TCP rollout failed: worker_fault={self.service.fatal_error!r}, "
                                f"state={self.service.activation_state}, batch={batch}"
                            ) from error
                        self.assertEqual(completion_status["queue_depth"], 0)
                        self.assertTrue(completion_status["ready"])
                        rows.extend(completed)
                        progress.observe_executions(completed, cycle)
                    status = client.request({"op": "status"})
                    native = self.service.sender.snapshot()
            finally:
                self.service.stop()
                thread.join(timeout=5)
                listener.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(server_errors, [])
        self.assertEqual(len(rows), 64)
        self.assertTrue(all(row["status"] == "COMPLETED" for row in rows))
        self.assertEqual(len({row["command_id"] for row in rows}), 64)
        self.assertEqual(len(row_calls), 64)
        self.assertTrue(all(ticks == 5 for _, ticks in row_calls))
        np.testing.assert_array_equal([pose for pose, _ in row_calls], targets[:, :7])
        np.testing.assert_array_equal(gripper_calls, targets[:, 7])
        np.testing.assert_array_equal(
            [row["completion"]["gripper_target"] for row in rows], targets[:, 7]
        )
        self.assertEqual(native["call_count"], native["success_count"])
        self.assertIsNone(status["fatal_error"])
        self.assertEqual(len(status["recent_results"]), 32)
        self.assertTrue(status["ready"])
        self.assertTrue(progress.status(status)["passed"])
        self.assertEqual(self.service.activation_state, "stopped")
        self.assertEqual(len({event[-1] for event in self.robot.events}), 1)
        self.assertEqual(len([event for event in self.robot.events if event[0] == "references"]), 1)
        self.assertAlmostEqual(self.robot.gripper, -0.785)


if __name__ == "__main__":
    unittest.main()
