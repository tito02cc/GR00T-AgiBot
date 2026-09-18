"""Check candidate feedback behavior without a robot or SDK runtime."""

import importlib
import sys
import types
import unittest
from unittest import mock

import numpy as np


sys.modules.setdefault("agibot_gdk", types.ModuleType("agibot_gdk"))
candidate = importlib.import_module("g2_groot_continuous_controller")
legacy = importlib.import_module("g2_groot_persistent_right_arm_controller")


POSE = np.asarray([0.5, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0])


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class CandidateControllerTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.right = POSE.copy()
        self.calls = []
        self.times = []

        def pose(tf, frame):
            return (self.right if frame == legacy.RIGHT_FRAME else POSE).copy()

        for patcher in (
            mock.patch.object(candidate, "time", self.clock),
            mock.patch.object(legacy, "time", self.clock),
            mock.patch.object(candidate, "pose_values", side_effect=pose),
            mock.patch.object(legacy, "pose_values", side_effect=pose),
            mock.patch.object(candidate, "snapshot", return_value={}),
            mock.patch.object(candidate, "require_safe_state"),
            mock.patch.object(legacy, "emit"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def sender(self, robot, target, **kwargs):
        self.calls.append(target.copy())
        self.times.append(self.clock.now)
        return 0

    def controller(self, sender=None):
        return candidate.ContinuousRightArmController(
            object(), object(), sender=sender or self.sender,
            workspace_min=[0.3, -0.5, 0.7], workspace_max=[0.9, 0.1, 1.4],
        )

    def test_zero_initial_correction_and_legacy_defaults_unchanged(self):
        old_seed = legacy.SEED_TRANSLATION_COMPENSATION.copy()
        controller = self.controller()
        np.testing.assert_array_equal(controller.translation_compensation, [0, 0, 0])
        controller.tick(POSE)
        np.testing.assert_array_equal(self.calls[0], POSE)
        np.testing.assert_array_equal(legacy.SEED_TRANSLATION_COMPENSATION, old_seed)

    def test_simulated_bias_converges_with_existing_feedback_law(self):
        def biased_sender(robot, target, **kwargs):
            self.sender(robot, target)
            self.right = target.copy()
            self.right[2] -= 0.002
            return 0

        controller = self.controller(biased_sender)
        result = controller.calibrate(2.0)
        self.assertTrue(controller.calibrated)
        self.assertLess(result["final_position_error_m"], 1e-6)
        self.assertAlmostEqual(controller.translation_compensation[2], 0.002, places=6)
        np.testing.assert_allclose(np.diff(self.times), 0.02, atol=1e-12)
        # An instantaneous synthetic plant does not validate hardware startup.

    def test_all_sixteen_waypoints_get_five_ticks(self):
        controller = self.controller()
        controller.calibrated = True
        with mock.patch.object(controller, "_adapt"):
            for index in range(16):
                target = POSE.copy()
                target[0] += (index + 1) * 0.001
                result = controller.move_to(target, 0.1)
                self.assertEqual(result["ticks"], 5)
                np.testing.assert_allclose(self.calls[-1], target)
        self.assertEqual(len(self.calls), 80)
        np.testing.assert_allclose(np.diff(self.times), 0.02, atol=1e-12)

    def test_calibration_only_bias_is_learned_then_frozen_during_motion_and_hold(self):
        def biased_sender(robot, target, **kwargs):
            self.sender(robot, target)
            self.right = target.copy()
            self.right[2] -= 0.002
            return 0

        controller = candidate.ContinuousRightArmController(
            object(), object(), sender=biased_sender,
            workspace_min=[0.3, -0.5, 0.7], workspace_max=[0.9, 0.1, 1.4],
            freeze_compensation_after_calibration=True,
        )
        result = controller.calibrate(2.0)
        self.assertEqual(result["compensation_mode"], "calibration_only")
        self.assertAlmostEqual(controller.translation_compensation[2], 0.002, places=6)
        translation = controller.translation_compensation.copy()
        rotation = controller.rotation_compensation.copy()
        before = controller.tick_count
        for index in range(16):
            target = POSE.copy()
            target[0] += (index + 1) * 0.002
            self.assertEqual(controller.move_to(target, 0.1)["ticks"], 5)
            np.testing.assert_array_equal(controller.translation_compensation, translation)
            np.testing.assert_array_equal(controller.rotation_compensation, rotation)
            np.testing.assert_allclose(self.calls[-1][:3], target[:3] + translation)
        self.assertEqual(controller.tick_count - before, 80)
        controller.tick(target)
        np.testing.assert_array_equal(controller.translation_compensation, translation)

    def test_official_fault_or_expired_tick_does_not_publish(self):
        controller = self.controller()
        with mock.patch.object(candidate, "require_safe_state", side_effect=RuntimeError("GDK")):
            with self.assertRaisesRegex(RuntimeError, "GDK"):
                controller.tick(POSE)
        with self.assertRaisesRegex(RuntimeError, "latched"):
            controller.tick(POSE)
        other = self.controller()
        self.clock.now += 0.1
        with self.assertRaisesRegex(RuntimeError, "before publication"):
            other.tick(POSE)
        self.assertEqual(self.calls, [])

    def test_failed_send_latches_and_invalid_pose_never_sent(self):
        controller = self.controller(mock.Mock(return_value=1))
        with self.assertRaisesRegex(RuntimeError, "returned 1"):
            controller.tick(POSE)
        with self.assertRaisesRegex(RuntimeError, "latched"):
            controller.tick(POSE)
        self.assertEqual(controller.command_sender.call_count, 1)
        for target in ([0] * 7, [float("nan")] * 7, [1] * 6):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    self.controller().tick(np.asarray(target))
        self.assertEqual(self.calls, [])

    def test_failed_calibration_is_not_ready(self):
        controller = self.controller()
        with mock.patch.object(legacy.PersistentRightArmController, "calibrate",
                               side_effect=RuntimeError("convergence")):
            with self.assertRaisesRegex(RuntimeError, "convergence"):
                controller.calibrate()
        self.assertFalse(controller.calibrated)
        with self.assertRaisesRegex(RuntimeError, "latched"):
            controller.tick(POSE)


if __name__ == "__main__":
    unittest.main()
