"""Check that the hardware probe fails closed and retains either jaw endpoint."""

import copy
import importlib
import sys
import types
import unittest


sys.modules.setdefault("agibot_gdk", types.ModuleType("agibot_gdk"))
probe = importlib.import_module("g2_groot_continuous_probe")


def sample(jaw=-0.785):
    return {
        "pose": [0.5, -0.2, 1.0, 0, 0, 0, 1],
        "read_wall_ns": 2_000_000_000, "tf_timestamp_ns": 1_999_000_000,
        "joint_timestamp_ns": 1_999_000_000,
        "mode": 1, "motion_error": 0, "motion_error_msg": "",
        "whole": {"right_end_model": "omnipicker", "left_arm_error": 0,
                  "right_arm_error": 0, "left_end_error": 0, "right_end_error": 0},
        "tool_names": ["idx71_gripper_r_inner_joint1"],
        "jaw_enabled": True, "jaw_error": 0, "jaw": jaw,
        "right_joints": [{"name": f"idx{i}_arm_r_joint{i-60}", "error_code": 0}
                         for i in range(61, 68)],
    }


class ProbeTest(unittest.TestCase):
    def test_accepts_explicitly_retained_open_or_closed_jaw(self):
        for jaw in (-0.785, 0.0):
            value = sample(jaw)
            probe.validate_sample(value, None, jaw)
            probe.validate_sample(value, value, jaw)

    def test_jaw_change_and_faults_rejected(self):
        for mutate in (
            lambda s: s.update(jaw=-0.7),
            lambda s: s.update(jaw_enabled=False),
            lambda s: s.update(jaw_error=1),
            lambda s: s.update(motion_error=2),
            lambda s: s["whole"].update(right_arm_estop=True),
            lambda s: s["whole"].update(right_end_error=1),
            lambda s: s["right_joints"][0].update(error_code=1),
        ):
            with self.subTest(mutate=mutate):
                value = sample()
                mutate(value)
                with self.assertRaises(RuntimeError):
                    probe.validate_sample(value, None, -0.785)

    def test_original_downward_bound_and_small_upward_path(self):
        initial = sample()
        current = copy.deepcopy(initial)
        current["pose"][2] += 0.002
        probe.validate_sample(current, initial, -0.785)
        current["pose"][2] = initial["pose"][2] - 0.0016
        with self.assertRaisesRegex(RuntimeError, "envelope"):
            probe.validate_sample(current, initial, -0.785)

    def test_source_times_must_be_real_and_recent(self):
        for key in ("tf_timestamp_ns", "joint_timestamp_ns"):
            for stamp in (0, 1_000_000_000, 3_000_000_000):
                with self.subTest(key=key, stamp=stamp):
                    value = sample()
                    value[key] = stamp
                    with self.assertRaisesRegex(RuntimeError, "source time"):
                        probe.validate_sample(value, None, -0.785)

    def test_startup_allowance_is_limited_to_calibration(self):
        initial = sample(0.0)
        current = copy.deepcopy(initial)
        current["pose"][2] -= 0.002
        probe.validate_sample(current, initial, 0.0, calibration=True)
        with self.assertRaisesRegex(RuntimeError, "envelope"):
            probe.validate_sample(current, initial, 0.0)
        current["pose"][2] = initial["pose"][2] - 0.0031
        with self.assertRaisesRegex(RuntimeError, "envelope"):
            probe.validate_sample(current, initial, 0.0, calibration=True)

    def test_read_sample_uses_official_tf_timestamp_getter(self):
        # GDK Transform does not have a timestamp_ns field. Its timestamp is
        # returned by TF.get_latest_timestamp(frame), as in observation code.
        vector = types.SimpleNamespace(x=0.0, y=0.0, z=1.0)
        rotation = types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        tf = types.SimpleNamespace(
            get_tf_from_base_link=lambda frame: types.SimpleNamespace(
                translation=vector, rotation=rotation),
            get_latest_timestamp=lambda frame: 123456,
        )
        robot = types.SimpleNamespace(
            get_motion_control_status=lambda: types.SimpleNamespace(mode=1, error_code=0,
                                                                     error_msg=""),
            get_whole_body_status=lambda: {},
            get_end_state=lambda: {"right_end_state": {
                "names": ["idx71_gripper_r_inner_joint1"], "end_states": [{
                    "position": -0.785, "err_code": 0, "effort": 1, "enable": True}]}},
            get_joint_states=lambda: {"timestamp": 123400, "states": []},
        )
        result = probe.read_sample(robot, tf)
        self.assertEqual(result["tf_timestamp_ns"], 123456)
        self.assertEqual(result["joint_timestamp_ns"], 123400)
        self.assertEqual(result["jaw"], -0.785)


if __name__ == "__main__":
    unittest.main()
