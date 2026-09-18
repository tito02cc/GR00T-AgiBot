"""No hardware: prove predictions overlap consumption without replaying actions."""

import threading
import time
import unittest

from agibot.rtc.action_queue import ActionQueue
from agibot.rtc.adapter import targets_to_prefix
from agibot.rtc.contract import RTC_SCHEMA
from agibot.rtc.planner import AsyncRtcPlanner
from agibot.tools.g2_gr00t_shadow_adapter import decode_action_chunk, make_right_eef_state
import numpy as np
from scipy.spatial.transform import Rotation


def targets():
    values = np.zeros((16, 8))
    values[:, 0] = np.arange(16) * 0.001 + 0.5
    values[:, 1:3] = [-0.18, 1.05]
    values[:, 6] = 1
    values[:, 7] = np.linspace(0, -0.785, 16)
    return values


def as_policy(values):
    return {
        "right_eef": np.stack([make_right_eef_state(r[:7]) for r in values])[None],
        "right_gripper": values[None, :, 7:8].astype(np.float32),
    }


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.actions = ActionQueue()
        self.original = targets()
        self.actions.initialize(self.original)
        for _ in range(8):
            self.actions.pop()
        self.entered, self.finish = threading.Event(), threading.Event()
        self.created_thread = None
        self.action_thread = None
        self.closed_thread = None
        self.received = None
        self.no_rtc = False
        self.receipt_override = {}
        self.fail_prediction = False
        outer = self

        class Policy:
            def get_action(self, observation, options=None):
                outer.action_thread = threading.get_ident()
                outer.received = (observation, options)
                outer.entered.set()
                if not outer.finish.wait(3):
                    raise RuntimeError("test did not release model")
                if outer.fail_prediction:
                    raise ValueError("model failure")
                generated = targets()
                generated[:, 0] += 0.1
                request = options["rtc"]
                receipt = {
                    "schema": RTC_SCHEMA,
                    "enabled": True,
                    "overlap_steps": request["previous_actions"]["right_eef"].shape[1],
                    "frozen_steps": request["frozen_steps"],
                    "ramp_rate": request["ramp_rate"],
                    "frozen_return_restored_exactly": True,
                    **outer.receipt_override,
                }
                return as_policy(generated), {} if outer.no_rtc else {"rtc": receipt}

            def close(self):
                outer.closed_thread = threading.get_ident()

        def factory():
            self.created_thread = threading.get_ident()
            return Policy()

        self.planner = AsyncRtcPlanner(self.actions, factory)

    def tearDown(self):
        self.finish.set()
        self.planner.close()

    def completed(self):
        self.finish.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = self.planner.poll()
            if result is not None:
                return result
            time.sleep(0.005)
        self.fail("prediction did not finish")

    def start(self):
        self.planner.request({"state": [8]}, observation_step_index=8, frozen_steps=4)
        self.assertTrue(self.entered.wait(2))

    def test_async_consumption_preserves_committed_and_frozen(self):
        self.start()
        a, old_a = self.actions.pop()
        b, old_b = self.actions.pop()
        self.assertEqual((a, b), (8, 9))
        np.testing.assert_array_equal(old_a, self.original[8])
        np.testing.assert_array_equal(old_b, self.original[9])
        result = self.completed()
        self.assertTrue(result["merge"]["accepted"])
        self.assertEqual(result["merge"]["consumed_steps"], 2)
        for expected in (10, 11):
            index, target = self.actions.pop()
            self.assertEqual(index, expected)
            np.testing.assert_array_equal(target, self.original[expected])
        index, target = self.actions.pop()
        self.assertEqual(index, 12)
        self.assertAlmostEqual(target[0], 0.604, places=6)
        self.assertNotEqual(self.created_thread, threading.get_ident())
        self.assertEqual(self.created_thread, self.action_thread)

    def test_late_prediction_does_not_replay_or_replace(self):
        self.start()
        for _ in range(5):
            self.actions.pop()
        result = self.completed()
        self.assertFalse(result["merge"]["accepted"])
        self.assertEqual(self.actions.pop()[0], 13)

    def test_wrong_observation_alignment_leaves_queue_untouched(self):
        with self.assertRaises(ValueError):
            self.planner.request({}, observation_step_index=9)
        self.assertFalse(self.actions.pending)
        self.assertEqual(self.actions.next_index, 8)

    def test_historical_observation_freezes_already_committed_row(self):
        request = self.planner.request({}, observation_step_index=7, frozen_steps=4)
        self.assertEqual(request["overlap_steps"], 9)
        self.assertTrue(self.entered.wait(2))
        self.actions.pop()
        result = self.completed()
        self.assertEqual(result["merge"]["consumed_steps"], 2)
        self.assertTrue(result["merge"]["accepted"])
        for index in (9, 10):
            actual, row = self.actions.pop()
            self.assertEqual(actual, index)
            np.testing.assert_array_equal(row, self.original[index])

    def test_model_failure_or_silent_baseline_never_merges(self):
        for attr in ("fail_prediction", "no_rtc"):
            with self.subTest(attr=attr):
                setattr(self, attr, True)
                self.start()
                with self.assertRaises(RuntimeError):
                    self.completed()
                self.assertFalse(self.actions.pending)
                self.assertEqual(self.actions.remaining, 8)
                setattr(self, attr, False)

    def test_close_invalidates_pending_result(self):
        self.start()
        self.assertFalse(self.planner.close(0))
        self.finish.set()
        self.assertTrue(self.planner.close(2))
        self.assertIsNone(self.planner.poll())
        self.assertFalse(self.actions.pending)
        self.assertEqual(self.actions.remaining, 8)
        self.assertEqual(self.created_thread, self.closed_thread)

    def test_disabled_or_mismatched_acknowledgement_never_merges(self):
        for override in (
            {"enabled": False},
            {"schema": "another-implementation"},
            {"overlap_steps": 7},
            {"frozen_steps": 3},
            {"ramp_rate": 1.0},
            {"frozen_return_restored_exactly": False},
        ):
            with self.subTest(override=override):
                self.receipt_override = override
                self.start()
                with self.assertRaises(RuntimeError):
                    self.completed()
                self.assertFalse(self.actions.pending)
                self.assertEqual(self.actions.remaining, 8)

    def test_exhaustion_rejects_late_result_without_replaying(self):
        self.start()
        for _ in range(8):
            self.actions.pop()
        self.assertIsNone(self.actions.pop())
        result = self.completed()
        self.assertFalse(result["merge"]["accepted"])
        self.assertEqual(self.actions.next_index, 16)
        self.assertIsNone(self.actions.pop())

    def test_completion_at_frozen_end_can_extend_exhausted_queue(self):
        self.planner.request({}, observation_step_index=8, frozen_steps=8)
        self.assertTrue(self.entered.wait(2))
        for _ in range(8):
            self.actions.pop()
        self.assertIsNone(self.actions.pop())
        result = self.completed()
        self.assertTrue(result["merge"]["accepted"])
        self.assertEqual(self.actions.pop()[0], 16)

    def test_observation_copy_and_duplicate_request(self):
        observation = {"state": np.array([8.0])}
        self.planner.request(observation, observation_step_index=8)
        observation["state"][0] = 99
        self.assertTrue(self.entered.wait(2))
        np.testing.assert_array_equal(self.received[0]["state"], [8])
        with self.assertRaises(RuntimeError):
            self.planner.request({}, observation_step_index=8)

    def test_target_prefix_preserves_native_jaw_and_rotation(self):
        prefix = targets_to_prefix(self.original[8:])
        np.testing.assert_array_equal(
            prefix["right_gripper"][0, :, 0], self.original[8:, 7].astype(np.float32)
        )
        self.assertEqual(prefix["right_eef"].shape, (1, 8, 9))

    def test_nonidentity_rotation_roundtrip(self):
        original = self.original[8:].copy()
        rotations = Rotation.from_euler(
            "xyz", [[10 + i, -15, 80 - i] for i in range(8)], degrees=True
        )
        original[:, 3:7] = rotations.as_quat()
        decoded = decode_action_chunk(targets_to_prefix(original))
        np.testing.assert_allclose(decoded[:, :3], original[:, :3], atol=1e-7)
        rotation_error = (rotations.inv() * Rotation.from_quat(decoded[:, 3:7])).magnitude()
        self.assertLess(float(rotation_error.max()), 1e-6)
        np.testing.assert_allclose(decoded[:, 7], original[:, 7], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
