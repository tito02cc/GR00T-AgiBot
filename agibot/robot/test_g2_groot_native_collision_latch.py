from types import SimpleNamespace as NS
import unittest

from g2_groot_native_collision_latch import NativeCollisionLatch


def robot():
    state = NS(mode=1, control_mode=3, error_code=0, collision_pairs_1=[], collision_pairs_2=[])
    config = NS(is_enabled=True, sensitivity=3, checkout_timeout_ms=500)
    return NS(
        state=state, config=config,
        get_motion_control_status=lambda: state,
        get_collision_detection_config=lambda: config,
    )


class NativeCollisionLatchTest(unittest.TestCase):
    def test_arm_reads_native_config_without_setters(self):
        guard = NativeCollisionLatch()
        guard.arm(robot())
        self.assertTrue(guard.snapshot()["armed"])
        self.assertEqual(guard.snapshot()["configuration_at_arm"]["checkout_timeout_ms"], 500)

    def test_disabled_or_too_short_recovery_rejected(self):
        for changes in ({"is_enabled": False}, {"checkout_timeout_ms": 5}):
            fake = robot()
            for key, value in changes.items():
                setattr(fake.config, key, value)
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                NativeCollisionLatch().arm(fake)

    def test_mode_change_latches_even_after_firmware_recovers(self):
        fake = robot()
        guard = NativeCollisionLatch()
        guard.arm(fake)
        fake.state.control_mode = 1
        with self.assertRaisesRegex(RuntimeError, "interlock"):
            guard.check_robot(fake)
        fake.state.control_mode = 3
        with self.assertRaisesRegex(RuntimeError, "fault latched"):
            guard.check_robot(fake)
        with self.assertRaisesRegex(RuntimeError, "fault latched"):
            guard.arm(fake)

    def test_errors_collision_and_missing_feedback_fail_closed(self):
        for changes in ({"error_code": 1}, {"collision_pairs_2": ["link"]}, {"mode": 0}):
            fake = robot()
            guard = NativeCollisionLatch()
            guard.arm(fake)
            for key, value in changes.items():
                setattr(fake.state, key, value)
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                guard.check_robot(fake)
        fake = robot()
        guard = NativeCollisionLatch()
        guard.arm(fake)
        fake.get_motion_control_status = lambda: NS()
        with self.assertRaises(RuntimeError):
            guard.check_robot(fake)

    def test_unarmed_cannot_publish(self):
        with self.assertRaisesRegex(RuntimeError, "not armed"):
            NativeCollisionLatch().check_robot(robot())


if __name__ == "__main__":
    unittest.main()
