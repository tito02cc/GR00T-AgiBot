#!/usr/bin/env python3

import unittest
from unittest.mock import Mock, patch

from agibot.configs.xichong_right_single_grasp_config import XICHONG_RIGHT_SINGLE_GRASP_CONFIG
from agibot.scripts import run_g2_groot_full_protected_inference as runner
from agibot.scripts.run_g2_groot_full_protected_inference import (
    BRIDGE_SCHEMA,
    HORIZON,
    OPEN,
    TRAINING_REFERENCE,
    GripperCommandLatch,
    execute_action_chunk,
    validate_complete_chunk,
    validate_model_modality_config,
    validate_preflight,
    wait_result,
)
from agibot.tools.g2_gr00t_shadow_adapter import (
    decode_action_chunk,
    quaternion_xyzw_to_rot6d,
    rot6d_to_quaternion_xyzw,
)
import numpy as np
from scipy.spatial.transform import Rotation


class FullProtectedInferenceTest(unittest.TestCase):
    def test_batch_submits_once_and_checks_all_completion_receipts(self):
        targets = np.tile(np.append(TRAINING_REFERENCE, OPEN), (HORIZON, 1))
        client = Mock()
        commands = []

        def request(payload):
            if payload["op"] == "execute_h16_gripper":
                commands.extend(payload["commands"])
                return {
                    "ok": True, "accepted": True,
                    "chunk_submission": "atomic_h16_native_v1",
                    "command_ids": [command["command_id"] for command in commands],
                }
            self.assertEqual(payload["op"], "status")
            return {
                "ok": True, "queue_depth": 0,
                "recent_results": [
                    {"command_id": command["command_id"], "ok": True, "accepted": True}
                    for command in commands
                ],
            }

        client.request.side_effect = request
        completion_status = {"old": "must be replaced"}
        with patch.object(runner.time, "sleep") as sleep:
            _, rows = execute_action_chunk(
                client, targets, "batch", 0, native_chunk_submission=True,
                completion_status_out=completion_status,
            )
        sleep.assert_not_called()
        self.assertEqual(client.request.call_count, 2)
        self.assertNotIn("old", completion_status)
        self.assertEqual(completion_status["queue_depth"], 0)
        self.assertEqual(len(completion_status["recent_results"]), 16)
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(row["status"] == "COMPLETED" for row in rows))
        np.testing.assert_array_equal([command["target_pose"] for command in commands], targets[:, :7])

    def test_batch_uncertain_ack_is_not_retried(self):
        targets = np.tile(np.append(TRAINING_REFERENCE, OPEN), (HORIZON, 1))
        client = Mock()
        client.request.side_effect = TimeoutError("lost reply")
        rows = []
        with self.assertRaises(TimeoutError):
            execute_action_chunk(
                client, targets, "batch", 0, native_chunk_submission=True, execution_rows=rows
            )
        self.assertEqual(client.request.call_count, 1)
        self.assertTrue(all(row["status"] == "SUBMISSION_UNCONFIRMED" for row in rows))

    def test_batch_incomplete_input_is_rejected_before_network(self):
        client = Mock()
        with self.assertRaisesRegex(ValueError, "complete H16"):
            execute_action_chunk(client, np.zeros((15, 8)), "bad", 0, native_chunk_submission=True)
        client.request.assert_not_called()

    def test_complete_horizon_is_sixteen(self):
        self.assertEqual(HORIZON, 16)

    def test_rot6d_xyzw_round_trip(self):
        original = Rotation.from_euler("xyz", [23.0, -41.0, 87.0], degrees=True).as_quat()
        recovered = rot6d_to_quaternion_xyzw(quaternion_xyzw_to_rot6d(original))
        self.assertAlmostEqual(abs(float(np.dot(original, recovered))), 1.0, places=6)

    def test_decode_action_chunk_returns_xyzw_and_clips_gripper(self):
        quaternion = Rotation.from_euler("z", 35.0, degrees=True).as_quat()
        rot6d = quaternion_xyzw_to_rot6d(quaternion)
        eef = np.tile(np.concatenate(([0.6, -0.25, 1.04], rot6d)), (HORIZON, 1))
        gripper = np.linspace(-1.0, 0.2, HORIZON, dtype=np.float32)[:, None]
        decoded = decode_action_chunk({"right_eef": eef[None], "right_gripper": gripper[None]})
        self.assertEqual(decoded.shape, (HORIZON, 8))
        self.assertTrue(np.all(decoded[:, 7] >= OPEN))
        self.assertTrue(np.all(decoded[:, 7] <= 0.0))
        self.assertAlmostEqual(abs(float(np.dot(decoded[0, 3:7], quaternion))), 1.0, places=6)

    def test_gripper_passes_every_finite_policy_value_unchanged(self):
        latch = GripperCommandLatch()
        values = [OPEN, -0.753, -0.68, -0.30, -0.60, 0.0, OPEN]
        self.assertEqual([latch.map(value) for value in values], values)
        self.assertTrue(latch.close_requested)

    def test_gripper_rejects_non_finite(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            GripperCommandLatch().map(float("nan"))

    def test_complete_chunk_validation(self):
        targets = np.tile(TRAINING_REFERENCE, (HORIZON, 1))
        targets = np.column_stack((targets, np.full(HORIZON, OPEN)))
        targets[:, 0] += np.arange(1, HORIZON + 1) * 0.0005
        metrics = validate_complete_chunk(TRAINING_REFERENCE, targets)
        self.assertEqual(len(metrics), HORIZON)

    def test_incomplete_chunk_is_rejected(self):
        targets = np.zeros((HORIZON - 1, 8), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "expected complete"):
            validate_complete_chunk(TRAINING_REFERENCE, targets)

    def test_chunk_discontinuity_is_measured_but_not_modified_or_rejected(self):
        targets = np.tile(np.append(TRAINING_REFERENCE, OPEN), (HORIZON, 1))
        targets[5, 0] += 0.04
        metrics = validate_complete_chunk(TRAINING_REFERENCE, targets)
        self.assertAlmostEqual(metrics[5]["translation_m"], 0.04)
        self.assertAlmostEqual(metrics[6]["translation_m"], 0.04)

    def _valid_info(self):
        return {
            "schema": BRIDGE_SCHEMA,
            "right_gripper_protected_closure_enabled": True,
            "right_gripper_physical_zero_closure_enabled": False,
            "right_arm_enabled": True,
            "left_arm_enabled": False,
            "head_waist_chassis_enabled": False,
            "model_waypoint_hz": 10.0,
        }

    def _valid_status(self):
        return {
            "ready": True,
            "fatal_error": None,
            "live_pose": TRAINING_REFERENCE.tolist(),
            "right_gripper": {
                "fault": None,
                "recovery_requested": False,
                "last_completed": OPEN,
            },
        }

    def test_preflight_accepts_exact_training_start(self):
        result = validate_preflight(self._valid_info(), self._valid_status())
        self.assertEqual(result["horizon"], 16)
        self.assertEqual(result["gripper_mapping"], "official_absolute_action_passthrough")

    def test_preflight_records_initial_offset_without_a_software_pose_gate(self):
        status = self._valid_status()
        status["live_pose"] = TRAINING_REFERENCE.copy()
        status["live_pose"][0] += 0.0003
        result = validate_preflight(self._valid_info(), status)
        self.assertAlmostEqual(result["initial_error_m"], 0.0003)

    def test_preflight_rejects_unprotected_gripper(self):
        info = self._valid_info()
        info["right_gripper_protected_closure_enabled"] = False
        with self.assertRaisesRegex(RuntimeError, "protected closure"):
            validate_preflight(info, self._valid_status())

    def test_preflight_rejects_left_arm(self):
        info = self._valid_info()
        info["left_arm_enabled"] = True
        with self.assertRaisesRegex(RuntimeError, "forbidden robot group"):
            validate_preflight(info, self._valid_status())

    def test_grasp_rejects_endpoint_only_gripper_bridge(self):
        info = self._valid_info()
        info["gripper_policy_mode"] = "endpoint_process_handoff"
        with self.assertRaisesRegex(RuntimeError, "per-row gripper control"):
            validate_preflight(info, self._valid_status())

    def test_model_modality_config_matches_live_contract(self):
        validate_model_modality_config(XICHONG_RIGHT_SINGLE_GRASP_CONFIG)

    def test_model_modality_config_rejects_wrong_horizon(self):
        configs = dict(XICHONG_RIGHT_SINGLE_GRASP_CONFIG)
        action = configs["action"]
        original = action.delta_indices
        try:
            action.delta_indices = list(range(8))
            with self.assertRaisesRegex(RuntimeError, "horizon"):
                validate_model_modality_config(configs)
        finally:
            action.delta_indices = original


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class RecordingBridge:
    def __init__(self, clock, delays=None):
        self.clock = clock
        self.delays = delays or {}
        self.submissions = []

    def request(self, payload):
        if payload["op"] == "status":
            return {
                "ok": True,
                "queue_depth": 0,
                "recent_results": [
                    {"command_id": submission["command_id"], "accepted": True}
                    for submission in self.submissions
                ],
            }
        index = len(self.submissions)
        self.submissions.append({**payload, "sent_at": self.clock.now})
        self.clock.sleep(self.delays.get(index, 0.0))
        return {"ok": True, "accepted": True, "command_id": payload["command_id"]}


class ChunkExecutionTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.targets = np.tile(np.append(TRAINING_REFERENCE, OPEN), (HORIZON, 1))
        self.targets[:, 0] += np.arange(HORIZON) * 0.001
        self.targets[:, 7] = np.linspace(OPEN, 0.0, HORIZON)
        self.monotonic = patch.object(runner.time, "monotonic", self.clock.monotonic)
        self.sleep = patch.object(runner.time, "sleep", self.clock.sleep)
        self.monotonic.start()
        self.sleep.start()
        self.addCleanup(self.monotonic.stop)
        self.addCleanup(self.sleep.stop)

    def test_regular_acknowledgements_keep_ten_hz_and_all_targets(self):
        client = RecordingBridge(self.clock, {index: 0.02 for index in range(HORIZON)})
        _, rows = execute_action_chunk(client, self.targets, "normal", 0)
        np.testing.assert_allclose(
            [submission["sent_at"] for submission in client.submissions],
            np.arange(HORIZON) * 0.1,
        )
        np.testing.assert_allclose(
            [submission["target_pose"] for submission in client.submissions],
            self.targets[:, :7],
        )
        np.testing.assert_allclose(
            [submission["target_gripper"] for submission in client.submissions],
            self.targets[:, 7],
        )
        self.assertTrue(all(row["status"] == "COMPLETED" for row in rows))

    def test_blocking_handoff_rebases_remaining_rows_without_burst(self):
        client = RecordingBridge(self.clock, {4: 10.0})
        _, rows = execute_action_chunk(client, self.targets, "handoff", 0)
        submitted = np.array([submission["sent_at"] for submission in client.submissions])
        self.assertAlmostEqual(submitted[5] - submitted[4], 10.1)
        self.assertTrue(np.all(np.diff(submitted) >= 0.1 - 1e-9))
        np.testing.assert_allclose(np.diff(submitted[5:]), 0.1)
        self.assertTrue(rows[4]["cadence_rebased"])

    def test_overdue_network_ack_does_not_shorten_following_period(self):
        client = RecordingBridge(self.clock, {2: 0.16, 8: 0.11})
        execute_action_chunk(client, self.targets, "network", 0)
        submitted = np.array([submission["sent_at"] for submission in client.submissions])
        self.assertAlmostEqual(submitted[3] - submitted[2], 0.26)
        self.assertAlmostEqual(submitted[9] - submitted[8], 0.21)
        self.assertTrue(np.all(np.diff(submitted) >= 0.1 - 1e-9))

    def test_native_slow_ack_uses_elapsed_period_without_burst_or_target_changes(self):
        delays = {index: 0.02 for index in range(HORIZON)}
        delays.update({2: 0.16, 8: 0.11, 11: 1.0})
        client = RecordingBridge(self.clock, delays)
        _, rows = execute_action_chunk(
            client, self.targets, "native-network", 0, native_ack_pacing=True
        )
        submitted = np.array([submission["sent_at"] for submission in client.submissions])
        expected_intervals = np.array(
            [max(0.1, delays[index]) for index in range(HORIZON - 1)]
        )
        np.testing.assert_allclose(np.diff(submitted), expected_intervals)
        self.assertTrue(np.all(np.diff(submitted) >= 0.1 - 1e-9))
        self.assertEqual(len(client.submissions), HORIZON)
        self.assertEqual(len({row["command_id"] for row in rows}), HORIZON)
        np.testing.assert_array_equal(
            [submission["target_pose"] for submission in client.submissions],
            self.targets[:, :7],
        )
        np.testing.assert_array_equal(
            [submission["target_gripper"] for submission in client.submissions],
            self.targets[:, 7],
        )
        self.assertTrue(all(row["status"] == "COMPLETED" for row in rows))
        self.assertTrue(all(not row["cadence_rebased"] for row in rows))
        self.assertEqual(
            [index for index, row in enumerate(rows) if row["acknowledgement_overdue"]],
            [2, 8, 11],
        )

    def test_native_pacing_still_requires_every_successful_completion_receipt(self):
        for failure in ("missing", "failed"):
            with self.subTest(failure=failure):
                client = RecordingBridge(self.clock, {2: 0.16})
                original_request = client.request

                def altered_receipts(payload):
                    response = original_request(payload)
                    if payload["op"] == "status":
                        if failure == "missing":
                            response["recent_results"] = response["recent_results"][1:]
                        else:
                            response["recent_results"][0]["accepted"] = False
                    return response

                client.request = altered_receipts
                rows = []
                expected_error = (
                    "missing completion receipt" if failure == "missing" else "command execution failed"
                )
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    execute_action_chunk(
                        client,
                        self.targets,
                        f"native-{failure}",
                        0,
                        execution_rows=rows,
                        native_ack_pacing=True,
                    )
                self.assertEqual(len(rows), HORIZON)
                self.assertEqual(rows[-1]["status"], "COMPLETED")
                self.assertEqual(
                    rows[0]["status"],
                    "COMPLETION_UNCONFIRMED" if failure == "missing" else "FAILED",
                )

    def test_status_failure_is_reported_immediately(self):
        client = Mock()
        client.request.return_value = {"ok": False, "message": "arm child disconnected"}
        with self.assertRaisesRegex(RuntimeError, "arm child disconnected"):
            wait_result(client, "last-command", timeout_s=25.0)
        self.assertEqual(client.request.call_count, 1)
        self.assertEqual(self.clock.now, 0.0)

    def test_final_receipt_alone_cannot_confirm_a_complete_chunk(self):
        client = RecordingBridge(self.clock)
        original_request = client.request

        def drop_earlier_receipts(payload):
            response = original_request(payload)
            if payload["op"] == "status":
                response["recent_results"] = response["recent_results"][-1:]
            return response

        client.request = drop_earlier_receipts
        rows = []
        with self.assertRaisesRegex(RuntimeError, "missing completion receipt"):
            execute_action_chunk(client, self.targets, "missing", 0, execution_rows=rows)
        self.assertEqual(len(rows), HORIZON)
        self.assertEqual(rows[0]["status"], "COMPLETION_UNCONFIRMED")
        self.assertIn("acknowledgement", rows[0])
        self.assertEqual(rows[-1]["status"], "COMPLETED")

    def test_failed_earlier_receipt_cannot_be_hidden_by_successful_final_row(self):
        client = RecordingBridge(self.clock)
        original_request = client.request

        def fail_first_receipt(payload):
            response = original_request(payload)
            if payload["op"] == "status":
                response["recent_results"][0]["accepted"] = False
            return response

        client.request = fail_first_receipt
        rows = []
        with self.assertRaisesRegex(RuntimeError, "command execution failed"):
            execute_action_chunk(client, self.targets, "failed-first", 0, execution_rows=rows)
        self.assertEqual(rows[0]["status"], "FAILED")
        self.assertFalse(rows[0]["completion"]["accepted"])
        self.assertEqual(rows[-1]["status"], "COMPLETED")

    def test_failed_completion_is_not_counted_as_success(self):
        client = Mock()
        client.request.return_value = {
            "ok": True,
            "queue_depth": 0,
            "recent_results": [{"command_id": "failed", "accepted": False}],
        }
        with self.assertRaisesRegex(RuntimeError, "command execution failed"):
            wait_result(client, "failed")

    def test_rejected_row_keeps_prior_rows_and_physical_evidence(self):
        client = RecordingBridge(self.clock)
        original_request = client.request
        release = {"actual_gripper": -0.74, "commands_sent": 3}

        def reject_third(payload):
            if len(client.submissions) == 2:
                return {"ok": False, "message": "restart failed", "release": release}
            return original_request(payload)

        client.request = reject_third
        rows = []
        with self.assertRaisesRegex(RuntimeError, "restart failed"):
            execute_action_chunk(client, self.targets, "partial", 0, execution_rows=rows)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["status"], "ACCEPTED")
        self.assertEqual(rows[-1]["status"], "FAILED")
        self.assertEqual(rows[-1]["release"], release)
        self.assertFalse(rows[-1]["acknowledgement"]["ok"])


if __name__ == "__main__":
    unittest.main()
