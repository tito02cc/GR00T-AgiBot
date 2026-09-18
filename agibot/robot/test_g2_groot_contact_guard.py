"""Synthetic limits only: these numbers must never be copied to the robot."""

from types import SimpleNamespace as NS
import unittest

from g2_groot_contact_guard import ContactGuard, ContactGuardTrip


def synthetic_config():
    return {
        "schema": "g2_right_contact_guard_v1", "robot_ip": "fake-robot",
        "base_frame": "base_link", "eef_frame": "arm_r_end_link",
        "confirmed": True, "eef_min_z_m": 1.0,
        "max_force_norm_n": 10.0, "max_torque_norm_nm": 2.0,
        "calibration_note": "UNIT TEST ONLY, NOT HARDWARE LIMITS",
    }


def wrench(force=0.0, torque=0.0):
    return NS(force=NS(x=force, y=0.0, z=0.0), torque=NS(x=0.0, y=torque, z=0.0))


def motion(force=0.0, torque=0.0):
    return NS(
        error_code=0, frame_names=["arm_l_end_link", "arm_r_end_link"],
        wrenches=[wrench(1000), wrench(force, torque)], collision_pairs_1=[], collision_pairs_2=[],
    )


POSE = [0.5, -0.2, 1.05, 0.0, 0.0, 0.0, 1.0]


class ContactGuardTest(unittest.TestCase):
    def test_pending_config_cannot_activate(self):
        config = synthetic_config()
        config.update(confirmed=False, eef_min_z_m=None, calibration_note="")
        guard = ContactGuard(config)
        self.assertFalse(guard.snapshot()["configured"])
        with self.assertRaisesRegex(ValueError, "site confirmation"):
            guard.ensure_ready()

    def test_confirmed_but_missing_limits_still_cannot_activate(self):
        for key in ("eef_min_z_m", "max_force_norm_n", "max_torque_norm_nm"):
            config = synthetic_config()
            config[key] = None
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "site confirmation"):
                ContactGuard(config).ensure_ready()

    def test_bad_limits_and_unknown_fields_fail(self):
        for changes in (
            {"max_force_norm_n": -1}, {"max_torque_norm_nm": float("nan")},
            {"eef_min_z_m": True}, {"unexpected": 1}, {"confirmed": "true"},
            {"eef_frame": "arm_l_end_link"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ContactGuard({**synthetic_config(), **changes})

    def test_named_right_frame_not_fixed_index(self):
        guard = ContactGuard(synthetic_config())
        state = motion(force=3, torque=0.2)
        state.frame_names.reverse()
        state.wrenches.reverse()
        guard.check_feedback(state, POSE)
        self.assertEqual(guard.snapshot()["last_feedback"]["force_norm_n"], 3)

    def test_missing_or_ambiguous_or_nonfinite_feedback_fails(self):
        for state in (
            NS(error_code=0),
            NS(**{**vars(motion()), "frame_names": ["arm_r_end_link", "arm_r_end_link"]}),
            motion(force=float("nan")),
        ):
            with self.subTest(state=state), self.assertRaises(ContactGuardTrip):
                ContactGuard(synthetic_config()).check_feedback(state, POSE)

    def test_force_torque_collision_and_error_trip(self):
        for state in (
            motion(force=10.01), motion(torque=2.01),
            NS(**{**vars(motion()), "collision_pairs_1": ["link1"]}),
            NS(**{**vars(motion()), "error_code": 1}),
        ):
            with self.subTest(state=state), self.assertRaises(ContactGuardTrip):
                ContactGuard(synthetic_config()).check_feedback(state, POSE)

    def test_norm_uses_all_three_axes(self):
        state = motion()
        state.wrenches[1].force = NS(x=6, y=6, z=6)
        with self.assertRaisesRegex(ContactGuardTrip, "force norm"):
            ContactGuard(synthetic_config()).check_feedback(state, POSE)

    def test_pose_below_floor_is_rejected_not_clipped_and_fault_latches(self):
        guard = ContactGuard(synthetic_config())
        pose = POSE.copy()
        pose[2] = 0.99
        with self.assertRaisesRegex(ContactGuardTrip, "below confirmed EEF floor"):
            guard.check_pose(pose)
        self.assertEqual(pose[2], 0.99)
        with self.assertRaisesRegex(ContactGuardTrip, "fault latched"):
            guard.check_pose(POSE)

    def test_live_below_floor_rejected(self):
        with self.assertRaisesRegex(ContactGuardTrip, "live_feedback"):
            ContactGuard(synthetic_config()).check_feedback(motion(), [0, 0, 0.9, 0, 0, 0, 1])

    def test_limit_equality_passes(self):
        guard = ContactGuard(synthetic_config())
        guard.check_feedback(motion(force=10, torque=2), [0, 0, 1, 0, 0, 0, 1])
        self.assertIsNone(guard.snapshot()["fault"])

    def test_config_cannot_be_changed_through_snapshot(self):
        config = synthetic_config()
        guard = ContactGuard(config)
        config["eef_min_z_m"] = 0.0
        guard.snapshot()["config"]["eef_min_z_m"] = 0.0
        self.assertEqual(guard.snapshot()["config"]["eef_min_z_m"], 1.0)


if __name__ == "__main__":
    unittest.main()
