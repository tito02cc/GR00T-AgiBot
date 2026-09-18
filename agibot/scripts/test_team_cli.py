from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from agibot.scripts import team_cli


class TeamCliConfigurationTest(unittest.TestCase):
    def calibrated_robot_text(self) -> str:
        template = (
            team_cli.AGIBOT_ROOT / "configs/g2_right_arm.template.toml"
        ).read_text(encoding="utf-8")
        replacements = {
            'robot_id = "CHANGE_ME"': 'robot_id = "test_robot"',
            'gdk_version = "CHANGE_ME"': 'gdk_version = "test_gdk"',
            "calibration_confirmed = false": "calibration_confirmed = true",
            'calibrated_at = "CHANGE_ME"': 'calibrated_at = "2026-09-01"',
            'robot_host = "CHANGE_ME"': 'robot_host = "192.0.2.10"',
            "raw_open_nominal = -1.0": "raw_open_nominal = 0.0",
            "raw_closed_nominal = -1.0": "raw_closed_nominal = 120.0",
            "accepted_settled_statuses = []": "accepted_settled_statuses = [0, 2, 3]",
            "hardware_minimum_xyz_m = [0.0, 0.0, 0.0]": (
                "hardware_minimum_xyz_m = [0.1, -0.4, 0.8]"
            ),
            "hardware_maximum_xyz_m = [0.0, 0.0, 0.0]": (
                "hardware_maximum_xyz_m = [0.8, 0.2, 1.4]"
            ),
            "policy_minimum_xyz_m = [0.0, 0.0, 0.0]": (
                "policy_minimum_xyz_m = [0.2, -0.3, 0.9]"
            ),
            "policy_maximum_xyz_m = [0.0, 0.0, 0.0]": (
                "policy_maximum_xyz_m = [0.7, 0.1, 1.3]"
            ),
        }
        for old, new in replacements.items():
            template = template.replace(old, new)
        return template

    def test_released_task_contract_passes(self):
        self.assertEqual(team_cli.validate_task(team_cli.CASE_TASK), [])

    def test_reference_robot_contract_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.toml"
            path.write_text(self.calibrated_robot_text(), encoding="utf-8")
            self.assertEqual(team_cli.validate_robot(path), [])

    def test_left_robot_contract_uses_left_runtime_keys(self):
        config = self.calibrated_robot_text()
        replacements = {
            'arm = "right"': 'arm = "left"',
            "right_eef_frame": "left_eef_frame",
            "arm_r_end_link": "arm_l_end_link",
            "right_wrist_key": "left_wrist_key",
            "hand_right": "hand_left",
            "right_gripper_joint1": "left_gripper_joint1",
        }
        for old, new in replacements.items():
            config = config.replace(old, new)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.toml"
            path.write_text(config, encoding="utf-8")
            self.assertEqual(team_cli.validate_robot(path), [])

    def test_left_task_contract_uses_left_modality_keys(self):
        config = team_cli.CASE_TASK.read_text(encoding="utf-8")
        config = config.replace('arm = "right"', 'arm = "left"')
        config = config.replace("hand_right", "hand_left")
        config = config.replace("state.right_", "state.left_")
        config = config.replace("action.right_", "action.left_")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.toml"
            path.write_text(config, encoding="utf-8")
            self.assertEqual(team_cli.validate_task(path), [])

    def test_generic_task_template_requires_project_values(self):
        template = team_cli.AGIBOT_ROOT / "templates/task_profile/task.toml"
        errors = team_cli.validate_task(template)
        self.assertTrue(any("CHANGE_ME" in error or "replace" in error for error in errors))

    def test_template_cannot_be_used_for_physical_execution(self):
        template = team_cli.AGIBOT_ROOT / "configs/g2_right_arm.template.toml"
        errors = team_cli.validate_robot(template)
        self.assertTrue(any("placeholder" in error for error in errors))
        self.assertTrue(any("calibration_confirmed" in error for error in errors))
        self.assertTrue(any("gripper" in error for error in errors))

    def test_unconfirmed_config_is_rejected(self):
        reference = self.calibrated_robot_text()
        changed = reference.replace("calibration_confirmed = true", "calibration_confirmed = false")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.toml"
            path.write_text(changed, encoding="utf-8")
            errors = team_cli.validate_robot(path)
        self.assertIn("robot calibration_confirmed must be true", errors)

    def test_artifact_lock_passes(self):
        self.assertEqual(
            team_cli.validate_artifacts_lock(team_cli.EXAMPLE_ARTIFACTS_LOCK), []
        )

    def test_artifact_template_is_rejected_until_filled(self):
        template = team_cli.AGIBOT_ROOT / "templates/task_profile/artifacts.lock.json"
        errors = team_cli.validate_artifacts_lock(template)
        self.assertTrue(any("CHANGE_ME" in error for error in errors))

    def test_sha256_verification_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.bin"
            payload.write_bytes(b"released artifact\n")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            manifest = root / "SHA256SUMS"
            manifest.write_text(f"{digest}  payload.bin\n", encoding="utf-8")
            self.assertIsNone(team_cli.sha256_error(manifest, root))
            payload.write_bytes(b"modified artifact\n")
            self.assertIn("SHA256 verification failed", team_cli.sha256_error(manifest, root))

    def test_cli_exposes_supported_commands(self):
        parser = team_cli.build_parser()
        commands = (
            ["doctor"],
            [
                "prepare",
                "--task-config", str(team_cli.CASE_TASK),
                "--raw-root", "/raw",
                "--full-dataset", "/full",
                "--selected-dataset", "/selected",
                "--report-dir", "/reports",
            ],
            [
                "train", "preflight", "--ct-root", "/ct", "--dataset", "/dataset",
                "--task-config", str(team_cli.CASE_TASK),
            ],
            ["serve", "--task-config", str(team_cli.CASE_TASK), "--model", "/model"],
            [
                "robot-preflight", "--task-config", str(team_cli.CASE_TASK),
                "--robot-config", "/robot.toml", "--dataset", "/data",
            ],
            [
                "infer", "--task-config", str(team_cli.CASE_TASK),
                "--robot-config", "/robot.toml",
            ],
            [
                "recover", "--task-config", str(team_cli.CASE_TASK),
                "--robot-config", "/robot.toml",
            ],
        )
        for argv in commands:
            with self.subTest(command=argv[0]):
                self.assertTrue(hasattr(parser.parse_args(argv), "function"))


if __name__ == "__main__":
    unittest.main()
