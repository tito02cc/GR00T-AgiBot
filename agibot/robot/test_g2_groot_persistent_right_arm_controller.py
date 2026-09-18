#!/usr/bin/env python3
"""Restored legacy algorithm regressions; fake clock, TF and sender, no hardware."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
import unittest
from unittest import mock

import numpy as np


sys.modules.setdefault("agibot_gdk", types.ModuleType("agibot_gdk"))
m = importlib.import_module("g2_groot_persistent_right_arm_controller")
POSE = np.asarray([0.5, -0.25, 1.05, 0.0, 0.0, 0.0, 1.0])
SEED_T = np.asarray([-0.0009471253, 0.0004121550, 0.0040800230])
SEED_Q = m.normalize([0.0005071716, 0.0065928102, -0.0006921396, 0.9999778990])


class FakeClock:
    def __init__(self):
        self.now = 10.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        if duration < 0:
            raise AssertionError("negative sleep")
        self.now += duration


class LegacyControllerRegressionTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.tf = types.SimpleNamespace(left=POSE.copy(), right=POSE.copy())
        self.robot = object()
        self.sent, self.times = [], []

        def pose_values(tf, frame):
            return (tf.left if frame == m.LEFT_FRAME else tf.right).copy()

        for patcher in (
            mock.patch.object(m, "time", self.clock),
            mock.patch.object(m, "pose_values", side_effect=pose_values),
            mock.patch.object(m, "snapshot", return_value={"fake": True}),
            mock.patch.object(m, "require_safe_state"),
            mock.patch.object(m, "send", side_effect=self.record),
            mock.patch.object(m, "emit"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def build(self, **options):
        return m.PersistentRightArmController(self.robot, self.tf, **options)

    def record(self, robot, target):
        self.assertIs(robot, self.robot)
        self.sent.append(np.asarray(target).copy())
        self.times.append(self.clock.now)
        return 0

    def test_original_constructor_and_default_seeds_unchanged(self):
        self.assertEqual(
            list(inspect.signature(m.PersistentRightArmController).parameters),
            [
                "robot",
                "tf",
                "right_session_max_translation_m",
                "right_session_max_rotation_rad",
                "initial_translation_compensation",
                "initial_rotation_compensation",
                "workspace_min",
                "workspace_max",
                "required_motion_mode",
            ],
        )
        controller = self.build()
        self.assertEqual(controller.required_motion_mode, 5)
        self.assertFalse(controller.calibrated)
        self.assertEqual(controller.tick_count, 0)
        np.testing.assert_array_equal(controller.translation_compensation, SEED_T)
        np.testing.assert_allclose(controller.rotation_compensation, SEED_Q, atol=1e-15)
        np.testing.assert_array_equal(controller.desired, POSE)
        other = self.build()
        controller.translation_compensation[:] = 0.0
        np.testing.assert_array_equal(other.translation_compensation, SEED_T)
        self.assertEqual(self.sent, [])

    def test_explicit_seeds_copied_and_applied(self):
        translation = np.asarray([0.001, -0.0005, 0.002])
        rotation = m.rv_to_q(np.asarray([0.0, 0.008, 0.0]))
        expected_t, expected_q = translation.copy(), rotation.copy()
        controller = self.build(
            initial_translation_compensation=translation,
            initial_rotation_compensation=rotation,
            required_motion_mode=1,
        )
        translation[:] = 0.0
        rotation[:] = [0, 0, 0, 1]
        controller.tick(POSE)
        np.testing.assert_allclose(self.sent[0][:3], POSE[:3] + expected_t)
        np.testing.assert_allclose(self.sent[0][3:], expected_q, atol=1e-15)
        self.assertEqual(controller.required_motion_mode, 1)

    def test_original_seed_limits_and_roundoff_projection(self):
        for options in (
            {"initial_translation_compensation": [0, 0, 0.007]},
            {"initial_rotation_compensation": m.rv_to_q(np.asarray([0.06, 0, 0]))},
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "compensation exceeds limit"):
                    self.build(**options)
        limit = m.MAX_TOTAL_TRANSLATION_COMPENSATION_M
        controller = self.build(
            initial_translation_compensation=[np.nextafter(limit, np.inf), 0, 0]
        )
        self.assertLessEqual(np.linalg.norm(controller.translation_compensation), limit)
        self.assertEqual(self.sent, [])

    def test_calibration_keeps_default_bias_and_fifty_hz_schedule(self):
        controller = self.build()
        result = controller.calibrate(0.2)
        self.assertTrue(controller.calibrated)
        self.assertEqual(controller.tick_count, 10)
        self.assertEqual(result["final_position_error_m"], 0.0)
        expected = np.concatenate((POSE[:3] + SEED_T, SEED_Q))
        np.testing.assert_allclose(self.sent, np.tile(expected, (10, 1)), atol=1e-12)
        np.testing.assert_allclose(np.diff(self.times), np.full(9, 0.02), atol=1e-12)
        self.assertAlmostEqual(self.clock.now, 10.2)

    def test_ten_hz_waypoint_has_five_interpolated_ticks(self):
        controller = self.build(
            initial_translation_compensation=[0, 0, 0], initial_rotation_compensation=[0, 0, 0, 1]
        )
        controller.calibrated = True
        target = POSE.copy()
        target[0] += 0.01
        target[3:] = m.rv_to_q(np.asarray([0, 0.02, 0]))
        # Isolate interpolation; real tick/sender/scheduler still run.
        with mock.patch.object(controller, "_adapt"):
            result = controller.move_to(target, m.MODEL_PERIOD_S)
        expected = [
            np.concatenate(
                (
                    POSE[:3] * (1 - i / 5) + target[:3] * (i / 5),
                    m.interpolate_quaternion(POSE[3:], target[3:], i / 5),
                )
            )
            for i in range(1, 6)
        ]
        self.assertEqual(result["ticks"], 5)
        np.testing.assert_allclose(self.sent, expected, atol=1e-12)
        np.testing.assert_allclose(result["emitted_50hz_targets"], expected, atol=1e-12)
        np.testing.assert_allclose(controller.desired, target, atol=1e-12)
        np.testing.assert_allclose(np.diff(self.times), np.full(4, 0.02), atol=1e-12)
        self.assertAlmostEqual(self.clock.now, 10.1)

    def test_hold_remains_adaptive_as_original(self):
        controller = self.build(
            initial_translation_compensation=[0, 0, 0], initial_rotation_compensation=[0, 0, 0, 1]
        )
        self.tf.right[2] -= 0.001
        self.tf.right[3:] = m.rv_to_q(np.asarray([0, -0.001, 0]))
        controller.hold_ticks(2)
        np.testing.assert_allclose(controller.translation_compensation, [0, 0, 0.00024], atol=1e-12)
        np.testing.assert_allclose(
            controller.rotation_compensation, m.rv_to_q(np.asarray([0, 0.0002, 0])), atol=1e-12
        )
        np.testing.assert_array_equal(controller.desired, POSE)
        self.assertEqual(len(self.sent), 2)

    def test_official_gdk_fault_stops_before_next_adaptation_and_publish(self):
        controller = self.build(required_motion_mode=1)
        controller.tick(POSE)
        before = controller.translation_compensation.copy()
        with mock.patch.object(
            m, "require_safe_state", side_effect=RuntimeError("collision imminent")
        ) as guard:
            with mock.patch.object(controller, "_adapt") as adapt:
                with self.assertRaisesRegex(RuntimeError, "collision imminent"):
                    controller.tick(POSE)
                adapt.assert_not_called()
        guard.assert_called_once_with({"fake": True}, 1)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(controller.tick_count, 1)
        np.testing.assert_array_equal(controller.translation_compensation, before)

    def test_existing_post_dispatch_schedule_error(self):
        controller = self.build()

        def slow_send(robot, target):
            self.record(robot, target)
            self.clock.now += 0.041

        with mock.patch.object(m, "send", side_effect=slow_send):
            with self.assertRaisesRegex(RuntimeError, "50 Hz control deadline missed by"):
                controller.tick(POSE)
        # Historical check is after dispatch; do not invent an early gate.
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(controller.tick_count, 1)

    def test_workspace_validation_and_compensated_boundary(self):
        for options in (
            {"workspace_min": [0.4, -0.3]},
            {"workspace_min": [float("nan"), -0.3, 0.9]},
            {"workspace_min": [0.9, -0.3, 0.9], "workspace_max": [0.8, -0.1, 1.3]},
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "workspace"):
                    self.build(**options)
        controller = self.build()
        target = POSE.copy()
        target[0] = controller.workspace_max[0] + 0.02
        with self.assertRaisesRegex(RuntimeError, "compensated command outside extended workspace"):
            controller.tick(target)
        self.assertEqual(self.sent, [])

    def test_original_calibration_requirement_and_target_step_limit(self):
        controller = self.build()
        with self.assertRaisesRegex(RuntimeError, "not calibrated"):
            controller.move_to(POSE, 0.1)
        controller.calibrated = True
        target = POSE.copy()
        target[0] += m.MAX_TARGET_STEP_M + 0.001
        with self.assertRaisesRegex(RuntimeError, "target translation step exceeds limit"):
            controller.move_to(target, 0.1)
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
