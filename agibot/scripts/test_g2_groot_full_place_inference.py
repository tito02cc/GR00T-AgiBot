#!/usr/bin/env python3

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agibot.configs.xichong_right_place_r0002_config import XICHONG_RIGHT_PLACE_R0002_CONFIG
from agibot.scripts import run_g2_groot_full_place_inference as runner
from agibot.scripts.run_g2_groot_full_place_inference import (
    BRIDGE_SCHEMA,
    FULLY_OPEN,
    HORIZON,
    MIN_RETRACTION_M,
    TRAINING_GRIPPER_REFERENCE,
    TRAINING_REFERENCE,
    PlacementBridgeSession,
    PlacementProgress,
    validate_model_modality_config,
    validate_preflight,
)
from agibot.tools.g2_gr00t_shadow_adapter import make_right_eef_state
import numpy as np


class FullPlaceInferenceTest(unittest.TestCase):
    def test_execution_options_reject_old_bridge_and_unarmed_or_wrong_compensation(self):
        args = SimpleNamespace(
            require_native_collision_latch=True, required_control_mode=3,
            freeze_compensation_after_calibration=True,
        )
        info = {
            "compensation_mode": "calibration_only",
            "native_collision_latch": {
                "protocol": "g2_native_collision_latch_v1", "required_control_mode": 3,
                "minimum_recovery_ms": 500, "fault": None, "armed": False,
            },
        }
        runner.validate_execution_options(info, args)
        with self.assertRaisesRegex(RuntimeError, "did not arm"):
            runner.validate_execution_options(info, args, activated=True)
        info["native_collision_latch"].update(
            armed=True, configuration_at_arm={"is_enabled": True, "checkout_timeout_ms": 500}
        )
        runner.validate_execution_options(info, args, activated=True)
        with self.assertRaisesRegex(RuntimeError, "contract"):
            runner.validate_execution_options({}, args)
        with self.assertRaisesRegex(RuntimeError, "calibration-only"):
            runner.validate_execution_options({**info, "compensation_mode": "online_adaptive"}, args)

    def test_reuse_validated_completion_status_without_another_request(self):
        client = Mock()
        final_status = {
            "ok": True, "queue_depth": 0, "ready": True, "fatal_error": None,
            "live_pose": TRAINING_REFERENCE.tolist(), "right_gripper": {},
        }
        self.assertIs(runner.chunk_end_status(client, final_status), final_status)
        client.request.assert_not_called()
        self.assertIs(runner.chunk_end_status(client), client.request.return_value)
        client.request.assert_called_once_with({"op": "status"})

    def test_unhealthy_completion_status_is_not_reused_or_repolled(self):
        client = Mock()
        for status in ({}, {"ok": True, "queue_depth": 0, "ready": False}):
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, "do not advance"):
                runner.chunk_end_status(client, status)
        client.request.assert_not_called()

    def test_contact_guard_contract_requires_same_confirmed_limits(self):
        from agibot.robot.g2_groot_contact_guard import ContactGuard
        from agibot.robot.test_g2_groot_contact_guard import synthetic_config

        guard = ContactGuard(synthetic_config())
        runner.validate_contact_guard_contract(guard, guard.snapshot())
        wrong = guard.snapshot()
        wrong["config"]["eef_min_z_m"] = 0.8
        for advertised in (None, {}, wrong, {**guard.snapshot(), "fault": "force exceeded"}):
            with self.subTest(advertised=advertised), self.assertRaises(RuntimeError):
                runner.validate_contact_guard_contract(guard, advertised)

    def test_unconfirmed_contact_guard_refuses_before_model_or_robot_connection(self):
        from agibot.robot.test_g2_groot_contact_guard import synthetic_config

        with TemporaryDirectory() as directory:
            args = self._args(directory)
            args.contact_guard_config = Path(directory) / "pending.json"
            args.contact_guard_config.write_text(json.dumps({**synthetic_config(), "confirmed": False}))
            with (
                patch.object(runner, "parse_args", return_value=args),
                patch.object(runner, "G2LiveObservationClient") as observations,
                patch.object(runner, "PolicyClient") as model,
                patch.object(runner, "PlacementBridgeSession") as bridge,
                self.assertRaisesRegex(ValueError, "site confirmation"),
            ):
                runner.main()
            observations.assert_not_called()
            model.assert_not_called()
            bridge.assert_not_called()
            self.assertEqual(json.loads(args.report.read_text())["status"], "FAILED")

    def _args(self, directory, max_cycles=1):
        return SimpleNamespace(
            initial_pose=TRAINING_REFERENCE,
            initial_gripper=TRAINING_GRIPPER_REFERENCE,
            minimum_retraction_m=MIN_RETRACTION_M,
            observation_port=19100,
            action_port=19200,
            model_port=5564,
            prompt="place the workpiece",
            max_cycles=max_cycles,
            report=Path(directory) / "run.json",
        )

    def _snapshot(self, offset=0.0):
        pose = TRAINING_REFERENCE.copy()
        pose[0] += offset
        return SimpleNamespace(
            metadata={
                "right_eef_xyz_quaternion_xyzw": pose.tolist(),
                "right_gripper": {"training_position": TRAINING_GRIPPER_REFERENCE},
                "camera_skew_ms": 0.0,
                "maximum_state_camera_skew_ms": 0.0,
            },
            head_color_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            hand_right_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        )

    def _model_action(self, pose=TRAINING_REFERENCE, horizon=HORIZON):
        return {
            "right_eef": np.tile(make_right_eef_state(pose), (1, horizon, 1)),
            "right_gripper": np.full((1, horizon, 1), TRAINING_GRIPPER_REFERENCE),
        }

    def _info(self):
        return {
            "activation_protocol": runner.ACTIVATION_PROTOCOL,
            "activation_state": "active",
            "arm_owner_active": True,
            "schema": BRIDGE_SCHEMA,
            "right_gripper_protected_closure_enabled": True,
            "shutdown_gripper_action": "hold",
            "gripper_command_mode": "process_handoff",
            "gripper_policy_mode": "endpoint_process_handoff",
            "gripper_partial_targets": "deferred_until_endpoint",
            "right_arm_enabled": True,
            "left_arm_enabled": False,
            "head_waist_chassis_enabled": False,
            "model_waypoint_hz": 10.0,
            "control_hz": 50.0,
        }

    def _status(self):
        return {
            "activation_protocol": runner.ACTIVATION_PROTOCOL,
            "activation_state": "active",
            "arm_owner_active": True,
            "ready": True,
            "fatal_error": None,
            "live_pose_stale": False,
            "live_pose": TRAINING_REFERENCE.tolist(),
            "right_gripper": {
                "fault": None,
                "recovery_requested": False,
                "last_completed": TRAINING_GRIPPER_REFERENCE,
                "feedback_encoding": "native_radians",
                "last_observation": {
                    "raw_position": TRAINING_GRIPPER_REFERENCE,
                },
            },
        }

    def _release_execution(self):
        return {
            "command_id": "physical-open",
            "gripper_handoff": {
                "direction": "opening",
                "status": "COMPLETED",
                "final_position": -0.76,
                "arm_pose_before_tool": TRAINING_REFERENCE.tolist(),
            },
        }

    def _native_release_execution(self):
        return {
            "command_id": "row-after-physical-release",
            "status": "COMPLETED",
            "completion": {
                "command_id": "row-after-physical-release",
                "accepted": True,
                "gripper_release_event": {
                    "command_id": "native-physical-open",
                    "feedback_position": -0.76,
                    "pose": TRAINING_REFERENCE.tolist(),
                    "monotonic_s": 123.0,
                },
            },
        }

    def test_preflight_accepts_closed_placement_start(self):
        result = validate_preflight(self._info(), self._status())
        self.assertEqual(result["horizon"], HORIZON)
        self.assertAlmostEqual(result["initial_gripper_error"], 0.0)

    def test_transport_optimization_default_preserves_legacy_status_requests(self):
        session = object.__new__(PlacementBridgeSession)
        self.assertFalse(session.compact_status)
        report = runner.configure_transport_optimization(session, self._info(), False)
        payload = {"op": "status"}
        with patch.object(runner.PersistentBridgeSession, "request", autospec=True) as request:
            session.request(payload)
        request.assert_called_once_with(session, payload)
        self.assertIs(request.call_args.args[1], payload)
        self.assertFalse(report["enabled"])
        self.assertFalse(report["native_ack_pacing"])
        self.assertFalse(report["compact_status_requested"])

    def test_transport_optimization_rejects_endpoint_or_non_immediate_ack_bridge(self):
        for info in (
            self._info(),
            {**self._info(), "acknowledgement": "immediate_queue_acceptance"},
            {**self._info(), "gripper_command_mode": "native_trajectory_tracking"},
            {
                **self._info(),
                "gripper_command_mode": "native_trajectory_tracking",
                "gripper_policy_mode": "per_row_absolute",
                "acknowledgement": "blocking_execution",
            },
            {
                **self._info(),
                "gripper_command_mode": "native_trajectory_tracking",
                "acknowledgement": "immediate_queue_acceptance",
            },
        ):
            with self.subTest(info=info):
                session = object.__new__(PlacementBridgeSession)
                with patch.object(runner.PersistentBridgeSession, "request") as request:
                    with self.assertRaisesRegex(RuntimeError, "immediate-ACK native bridge"):
                        runner.configure_transport_optimization(session, info, True)
                request.assert_not_called()
                self.assertFalse(session.compact_status)

    def test_compact_status_keeps_all_receipts_and_does_not_modify_commands(self):
        session = object.__new__(PlacementBridgeSession)
        info = {
            **self._info(),
            "gripper_command_mode": "native_trajectory_tracking",
            "gripper_policy_mode": "per_row_absolute",
            "acknowledgement": "immediate_queue_acceptance",
        }
        report = runner.configure_transport_optimization(session, info, True)
        self.assertTrue(report["native_ack_pacing"])
        self.assertEqual(report["policy"], "official_synchronous_h16_no_rtc")
        status_payload = {"op": "status"}
        release = self._native_release_execution()["completion"]["gripper_release_event"]
        status = {
            **self._status(),
            "recent_results": [
                {
                    "command_id": f"row-{index}",
                    "accepted": True,
                    "gripper_release_event": release,
                }
                for index in range(32)
            ],
        }
        with patch.object(
            runner.PersistentBridgeSession, "request", autospec=True, return_value=status
        ) as request:
            result = session.request(status_payload)
            request.assert_called_once_with(session, {"op": "status", "compact": True})
            self.assertNotIn("after_command_id", request.call_args.args[1])
            self.assertEqual(status_payload, {"op": "status"})
            self.assertIs(result, status)
            self.assertEqual(len(result["recent_results"]), 32)
            self.assertEqual(result["recent_results"][0]["command_id"], "row-0")
            self.assertEqual(result["recent_results"][-1]["gripper_release_event"], release)
            command = {
                "op": "execute_h1_gripper",
                "command_id": "exact-native-row",
                "timestamp_ns": 123,
                "target_pose": TRAINING_REFERENCE.tolist(),
                "target_gripper": -0.49795,
            }
            for payload in (command, {"op": "info"}, {"op": "shutdown"}):
                session.request(payload)
                self.assertIs(request.call_args.args[1], payload)
                self.assertNotIn("compact", payload)
                self.assertNotIn("after_command_id", payload)
        untouched_session = object.__new__(PlacementBridgeSession)
        self.assertFalse(untouched_session.compact_status)

    def test_preflight_rejects_open_gripper(self):
        status = self._status()
        status["right_gripper"]["last_completed"] = -0.785
        status["right_gripper"]["last_observation"]["raw_position"] = -0.785
        with self.assertRaisesRegex(RuntimeError, "initial gripper error"):
            validate_preflight(self._info(), status)

    def test_preflight_requires_hold_on_shutdown(self):
        info = self._info()
        info["shutdown_gripper_action"] = "open"
        with self.assertRaisesRegex(RuntimeError, "hold gripper"):
            validate_preflight(info, self._status())

    def test_preflight_requires_explicit_gripper_capability(self):
        info = self._info()
        del info["gripper_policy_mode"]
        with self.assertRaisesRegex(RuntimeError, "supported gripper command/policy pair"):
            validate_preflight(info, self._status())

    def test_preflight_accepts_native_unified_pair_and_reports_its_mapping(self):
        info = self._info()
        info.update(
            gripper_command_mode="native_trajectory_tracking",
            gripper_policy_mode="per_row_absolute",
            gripper_partial_targets="executed_per_row",
        )
        result = validate_preflight(info, self._status())
        self.assertEqual(result["gripper_command_mode"], "native_trajectory_tracking")
        self.assertEqual(result["gripper_policy_mode"], "per_row_absolute")
        self.assertEqual(
            result["gripper_mapping"],
            "official_absolute_targets_native_unified_trajectory_tracking",
        )
        self.assertEqual(result["gripper_partial_targets"], "executed_per_row")

    def test_preflight_legacy_mapping_still_discloses_endpoint_handoff(self):
        result = validate_preflight(self._info(), self._status())
        self.assertEqual(
            result["gripper_mapping"],
            "official_absolute_targets_with_endpoint_process_handoff",
        )
        self.assertEqual(result["gripper_partial_targets"], "deferred_until_endpoint")

    def test_preflight_rejects_mixed_or_unknown_backend_pairs(self):
        pairs = [
            ("native_trajectory_tracking", "endpoint_process_handoff"),
            ("process_handoff", "per_row_absolute"),
            ("in_process_servo", "per_row_absolute"),
            ("native_trajectory_tracking", None),
            (None, "per_row_absolute"),
        ]
        for command_mode, policy_mode in pairs:
            with self.subTest(command_mode=command_mode, policy_mode=policy_mode):
                info = self._info()
                info.update(gripper_command_mode=command_mode, gripper_policy_mode=policy_mode)
                with self.assertRaisesRegex(RuntimeError, "supported gripper command/policy pair"):
                    validate_preflight(info, self._status())

    def test_native_release_uses_physical_event_pose_not_row_target(self):
        execution = self._native_release_execution()
        execution["pose"] = (TRAINING_REFERENCE + np.array([0.08, 0, 0, 0, 0, 0, 0])).tolist()
        execution["gripper"] = -0.785
        progress = PlacementProgress()
        progress.observe_executions([execution], 3)
        status = self._status()
        status["live_pose"][0] += MIN_RETRACTION_M + 0.01
        status["right_gripper"]["last_observation"]["raw_position"] = -0.76
        result = progress.status(status)
        self.assertTrue(result["passed"])
        self.assertEqual(result["release_evidence"]["command_id"], "native-physical-open")
        self.assertEqual(result["release_evidence"]["monotonic_s"], 123.0)
        self.assertEqual(result["release_evidence"]["source"], "native_gripper_release_event")
        np.testing.assert_allclose(result["release_pose"], TRAINING_REFERENCE)

    def test_native_idle_release_event_carried_forward_is_consumed_once(self):
        progress = PlacementProgress()
        execution = self._native_release_execution()
        progress.observe_executions([execution], 3)
        later = self._native_release_execution()
        later["completion"]["gripper_release_event"]["pose"][0] += 0.05
        progress.observe_executions([later], 4)
        self.assertEqual(progress.release_cycle, 3)
        np.testing.assert_allclose(progress.release_pose, TRAINING_REFERENCE)

    def test_unconfirmed_or_failed_native_receipts_cannot_prove_release(self):
        overrides = [
            {"status": "ACCEPTED"},
            {"status": "COMPLETION_UNCONFIRMED"},
            {"status": "FAILED"},
            {"completion": {"accepted": False}},
            {"completion": {"accepted": None}},
            {"completion": {"ok": False}},
            {"completion": {"error": "trajectory rejected"}},
        ]
        for override in overrides:
            with self.subTest(override=override):
                execution = self._native_release_execution()
                if "completion" in override:
                    execution["completion"].update(override["completion"])
                else:
                    execution.update(override)
                progress = PlacementProgress()
                progress.observe_executions([execution], 0)
                self.assertIsNone(progress.release_evidence)

    def test_invalid_native_physical_events_are_ignored(self):
        overrides = [
            {"feedback_position": -0.5},
            {"feedback_position": -100.0},
            {"feedback_position": float("nan")},
            {"feedback_position": float("inf")},
            {"pose": None},
            {"pose": [0.0] * 6},
            {"pose": [float("nan")] * 7},
            {"monotonic_s": float("nan")},
            {"monotonic_s": -1.0},
            {"command_id": ""},
            {"command_id": None},
        ]
        for override in overrides:
            with self.subTest(override=override):
                execution = self._native_release_execution()
                execution["completion"]["gripper_release_event"].update(override)
                progress = PlacementProgress()
                progress.observe_executions([execution], 0)
                self.assertIsNone(progress.release_evidence)

    def test_native_predicted_or_ack_only_event_is_not_physical_evidence(self):
        execution = self._native_release_execution()
        event = execution["completion"].pop("gripper_release_event")
        execution["acknowledgement"] = {"gripper_release_event": event, "ok": True}
        execution["gripper"] = -0.785
        progress = PlacementProgress()
        progress.observe_executions([execution], 0)
        self.assertIsNone(progress.release_evidence)

    def test_progress_requires_release_open_and_retraction(self):
        progress = PlacementProgress()
        targets = np.tile(np.append(TRAINING_REFERENCE, -0.01), (HORIZON, 1))
        targets[7:, 7] = -0.785
        targets[7:, 0] += np.linspace(0.0, MIN_RETRACTION_M + 0.01, HORIZON - 7)
        progress.observe_targets(targets, 2)
        progress.observe_executions([self._release_execution()], 2)
        status = self._status()
        status["live_pose"] = targets[-1, :7].tolist()
        status["right_gripper"]["last_completed"] = FULLY_OPEN
        status["right_gripper"]["last_observation"]["raw_position"] = FULLY_OPEN
        result = progress.status(status)
        self.assertTrue(result["passed"])
        self.assertEqual(result["release_cycle"], 2)
        self.assertEqual(result["release_evidence"]["command_id"], "physical-open")

    def test_predicted_open_and_measured_distance_cannot_fake_release_evidence(self):
        progress = PlacementProgress()
        targets = np.tile(np.append(TRAINING_REFERENCE, -0.785), (HORIZON, 1))
        progress.observe_targets(targets, 0)
        status = self._status()
        status["live_pose"][0] += MIN_RETRACTION_M + 0.02
        status["right_gripper"]["last_observation"]["raw_position"] = -0.76
        result = progress.status(status)
        self.assertTrue(result["release_intent_seen"])
        self.assertFalse(result["release_seen"])
        self.assertFalse(result["passed"])

    def test_retraction_is_measured_from_actual_release_pose(self):
        progress = PlacementProgress()
        targets = np.tile(np.append(TRAINING_REFERENCE, -0.785), (HORIZON, 1))
        targets[:, 0] += 0.08
        progress.observe_targets(targets, 1)
        progress.observe_executions([self._release_execution()], 2)
        status = self._status()
        status["live_pose"][0] += MIN_RETRACTION_M + 0.01
        status["right_gripper"]["last_observation"]["raw_position"] = -0.76
        result = progress.status(status)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["retraction_m"], MIN_RETRACTION_M + 0.01)
        np.testing.assert_allclose(result["release_pose"], TRAINING_REFERENCE)
        self.assertEqual(result["release_cycle"], 2)

    def test_incomplete_or_unverified_handoffs_do_not_count_as_release(self):
        overrides = [
            {"status": "FAILED"},
            {"direction": "closing"},
            {"final_position": -0.5},
            {"arm_pose_before_tool": None},
            {"arm_pose_before_tool": [float("nan")] * 7},
        ]
        for override in overrides:
            with self.subTest(override=override):
                execution = self._release_execution()
                execution["gripper_handoff"].update(override)
                progress = PlacementProgress()
                progress.observe_executions([execution], 0)
                self.assertIsNone(progress.release_evidence)

    def test_fault_or_stale_status_prevents_success_even_after_physical_release(self):
        progress = PlacementProgress()
        progress.observe_executions([self._release_execution()], 0)
        failures = [
            {"ready": False},
            {"fatal_error": "arm child lost"},
            {"live_pose_stale": True},
            {"live_pose_stale": None},
            {"ok": False},
            {"right_gripper": {"fault": "tool fault"}},
            {"right_gripper": {"recovery_requested": True}},
        ]
        for failure in failures:
            with self.subTest(failure=failure):
                status = self._status()
                status["live_pose"][0] += MIN_RETRACTION_M + 0.02
                status["right_gripper"]["last_observation"]["raw_position"] = -0.76
                if "right_gripper" in failure:
                    status["right_gripper"].update(failure["right_gripper"])
                else:
                    status.update(failure)
                result = progress.status(status)
                self.assertFalse(result["passed"])
                self.assertFalse(result["completion_status_healthy"])

    def test_no_release_is_not_success(self):
        progress = PlacementProgress()
        status = self._status()
        status["right_gripper"]["last_completed"] = FULLY_OPEN
        self.assertFalse(progress.status(status)["passed"])

    def test_command_bookkeeping_cannot_fake_physical_release(self):
        progress = PlacementProgress()
        targets = np.tile(np.append(TRAINING_REFERENCE, -0.785), (HORIZON, 1))
        progress.observe_targets(targets, 1)
        progress.observe_executions([self._release_execution()], 1)
        status = self._status()
        status["live_pose"][0] += MIN_RETRACTION_M + 0.02
        status["right_gripper"]["last_completed"] = -0.785
        # The hardware remains closed even though the old bridge recorded the
        # requested target as completed.
        status["right_gripper"]["last_observation"]["raw_position"] = -0.009
        result = progress.status(status)
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["gripper_actual_position"], -0.009)

    def test_model_modality_config_matches_place_contract(self):
        validate_model_modality_config(XICHONG_RIGHT_PLACE_R0002_CONFIG)

    def test_prepare_model_warms_discards_actions_and_resets_official_policy(self):
        events = []
        model = Mock()
        model.ping.side_effect = lambda: events.append("ping") or True
        model.get_modality_config.side_effect = lambda: (
            events.append("modality") or XICHONG_RIGHT_PLACE_R0002_CONFIG
        )
        model.get_action.side_effect = lambda observation: (
            events.append("warmup") or (self._model_action(), {})
        )
        model.reset.side_effect = lambda: events.append("policy_reset") or {}
        observation = Mock()
        observation.get_snapshot.side_effect = lambda: events.append("snapshot") or self._snapshot()
        preparation = {}
        result = runner.prepare_model(model, observation, "place workpiece", preparation)
        self.assertIsNone(result)
        self.assertEqual(events, ["ping", "modality", "snapshot", "warmup", "policy_reset"])
        self.assertEqual(preparation["status"], "READY")
        self.assertFalse(preparation["warmup_actions_executed"])
        self.assertEqual(preparation["warmup_horizon"], HORIZON)
        for key in (
            "ping_s",
            "modality_validation_s",
            "observation_s",
            "warmup_inference_s",
            "total_s",
        ):
            self.assertGreaterEqual(preparation[key], 0.0)

    def test_prepare_model_stops_before_observation_when_model_preflight_fails(self):
        for failure in ("ping", "modality"):
            with self.subTest(failure=failure):
                model = Mock()
                model.ping.return_value = failure != "ping"
                model.get_modality_config.return_value = {}
                observation = Mock()
                preparation = {}
                with self.assertRaises(RuntimeError):
                    runner.prepare_model(model, observation, "place workpiece", preparation)
                observation.get_snapshot.assert_not_called()
                model.get_action.assert_not_called()
                model.reset.assert_not_called()
                self.assertEqual(preparation["status"], "FAILED")
                self.assertGreaterEqual(preparation["total_s"], 0.0)

    def test_prepare_model_rejects_incomplete_warmup_chunk(self):
        model = Mock()
        model.ping.return_value = True
        model.get_modality_config.return_value = XICHONG_RIGHT_PLACE_R0002_CONFIG
        model.get_action.return_value = (self._model_action(horizon=HORIZON - 1), {})
        observation = Mock()
        observation.get_snapshot.return_value = self._snapshot()
        preparation = {}
        with self.assertRaisesRegex(ValueError, "expected complete"):
            runner.prepare_model(model, observation, "place workpiece", preparation)
        self.assertEqual(preparation["status"], "FAILED")
        self.assertFalse(preparation["warmup_actions_executed"])
        model.reset.assert_not_called()

    def test_standby_probe_rejects_old_active_and_inconsistent_bridges(self):
        standby = {
            "activation_protocol": runner.ACTIVATION_PROTOCOL,
            "activation_state": "standby",
            "arm_owner_active": False,
        }
        for bad in (
            {},
            {**standby, "activation_state": "active", "arm_owner_active": True},
            {**standby, "arm_owner_active": True},
            {**standby, "activation_state": "fault"},
            {**standby, "ok": False},
        ):
            for bad_response in (0, 1):
                with self.subTest(bad=bad, bad_response=bad_response):
                    responses = [standby, standby]
                    responses[bad_response] = bad
                    with patch.object(runner, "PersistentBridgeSession") as session_class:
                        session = session_class.return_value.__enter__.return_value
                        session.request.side_effect = responses
                        with self.assertRaises(RuntimeError):
                            runner.inspect_standby_bridge("127.0.0.1", 19200)
                        self.assertEqual(
                            [call.args[0]["op"] for call in session.request.call_args_list],
                            ["info", "status"],
                        )

    def test_standby_probe_closes_without_shutdown_or_activation(self):
        standby = {
            "activation_protocol": runner.ACTIVATION_PROTOCOL,
            "activation_state": "standby",
            "arm_owner_active": False,
        }
        session = object.__new__(runner.PersistentBridgeSession)
        session.request = Mock(return_value=standby)
        session.close = Mock()
        with patch.object(runner, "PersistentBridgeSession", return_value=session):
            self.assertEqual(runner.inspect_standby_bridge("127.0.0.1", 19200), standby)
        session.close.assert_called_once()
        self.assertEqual(
            [call.args[0]["op"] for call in session.request.call_args_list], ["info", "status"]
        )

    def test_main_does_not_activate_when_standby_or_model_preparation_fails(self):
        for failure in ("standby", "ping", "modality", "warmup"):
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                args = self._args(directory)
                observation = Mock()
                observation.get_info.return_value = {"control_api_exposed": False}
                observation.get_snapshot.return_value = self._snapshot()
                model = Mock()
                model.ping.return_value = failure != "ping"
                model.get_modality_config.return_value = (
                    {} if failure == "modality" else XICHONG_RIGHT_PLACE_R0002_CONFIG
                )
                model.get_action.side_effect = RuntimeError("warmup failed")
                with (
                    patch.object(runner, "parse_args", return_value=args),
                    patch.object(runner, "G2LiveObservationClient") as observation_class,
                    patch.object(runner, "PolicyClient") as model_class,
                    patch.object(runner, "inspect_standby_bridge") as standby_probe,
                    patch.object(runner, "PlacementBridgeSession") as action_class,
                    patch.object(runner, "execute_action_chunk") as execute,
                ):
                    observation_class.return_value.__enter__.return_value = observation
                    model_class.return_value.__enter__.return_value = model
                    standby_probe.return_value = {"activation_state": "standby"}
                    if failure == "standby":
                        standby_probe.side_effect = RuntimeError("bridge already active")
                    with self.assertRaises(RuntimeError):
                        runner.main()
                    action_class.assert_not_called()
                    execute.assert_not_called()
                    if failure == "standby":
                        model.ping.assert_not_called()
                        model.get_action.assert_not_called()
                report = json.loads(args.report.read_text())
                self.assertEqual(report["status"], "FAILED")
                self.assertEqual(report["cycles"], [])

    def test_main_discards_warmup_and_activates_before_fresh_h16_observations(self):
        events = []
        snapshots = [self._snapshot(offset) for offset in (0.0, 0.001, 0.002)]
        observation = Mock()
        observation.get_info.return_value = {"control_api_exposed": False}

        def get_snapshot():
            index = observation.get_snapshot.call_count - 1
            events.append(f"snapshot{index}")
            return snapshots[index]

        observation.get_snapshot.side_effect = get_snapshot
        model = Mock()
        model.ping.side_effect = lambda: events.append("ping") or True
        model.get_modality_config.return_value = XICHONG_RIGHT_PLACE_R0002_CONFIG

        def get_action(policy_observation):
            index = model.get_action.call_count - 1
            events.append(f"inference{index}")
            return {
                "right_eef": np.repeat(policy_observation["state"]["right_eef"], HORIZON, axis=1),
                "right_gripper": np.repeat(
                    policy_observation["state"]["right_gripper"], HORIZON, axis=1
                ),
            }, {}

        model.get_action.side_effect = get_action
        model.reset.side_effect = lambda: events.append("policy_reset") or {}
        status = self._status()
        status["desired_pose"] = TRAINING_REFERENCE.tolist()

        def request(payload):
            operation = payload["op"]
            events.append(operation)
            if operation == "info":
                return self._info()
            if operation == "status":
                return status
            if operation == "activate":
                self.assertEqual(payload["confirm"], runner.ACTIVATION_CONFIRMATION)
                return {"ok": True}
            if operation == "shutdown":
                return {"ok": True}
            raise AssertionError(f"unexpected operation: {operation}")

        session = object.__new__(PlacementBridgeSession)
        session.request = Mock(side_effect=request)
        session.close = Mock()
        executed = []

        def execute(client, targets, prefix, clock_offset_ns, *, execution_rows):
            events.append(f"execute{len(executed)}")
            executed.append(targets.copy())

        with TemporaryDirectory() as directory:
            args = self._args(directory, max_cycles=2)
            with (
                patch.object(runner, "parse_args", return_value=args),
                patch.object(runner, "G2LiveObservationClient") as observation_class,
                patch.object(runner, "PolicyClient") as model_class,
                patch.object(
                    runner,
                    "inspect_standby_bridge",
                    side_effect=lambda host, port: events.append("standby_probe") or {},
                ),
                patch.object(
                    runner,
                    "PlacementBridgeSession",
                    side_effect=lambda host, port: events.append("open_action_session") or session,
                ),
                patch.object(runner, "calibrate_bridge_clock", return_value=(0, 0)),
                patch.object(runner, "execute_action_chunk", side_effect=execute),
                patch.object(PlacementProgress, "status", return_value={"passed": False}),
                patch("builtins.print"),
            ):
                observation_class.return_value.__enter__.return_value = observation
                model_class.return_value.__enter__.return_value = model
                self.assertEqual(runner.main(), 1)
            report = json.loads(args.report.read_text())
        for first, second in (
            ("standby_probe", "ping"),
            ("inference0", "policy_reset"),
            ("policy_reset", "open_action_session"),
            ("activate", "snapshot1"),
            ("execute0", "snapshot2"),
        ):
            self.assertLess(events.index(first), events.index(second))
        self.assertEqual(len(executed), 2)
        self.assertEqual(observation.get_snapshot.call_count, 3)
        self.assertEqual(model.get_action.call_count, 3)
        for index, targets in enumerate(executed, start=1):
            self.assertEqual(targets.shape, (HORIZON, 8))
            np.testing.assert_allclose(targets[:, 0], TRAINING_REFERENCE[0] + index * 0.001)
            np.testing.assert_allclose(targets[:, 7], TRAINING_GRIPPER_REFERENCE)
        self.assertTrue(report["action_bridge_startup"]["activated_after_model_ready"])
        self.assertFalse(report["model_preparation"]["warmup_actions_executed"])
        self.assertEqual(report["status"], "MAX_CYCLES_REACHED")
        session.close.assert_called_once()

    def test_rejected_activation_stops_without_executing_or_retrying(self):
        session = object.__new__(PlacementBridgeSession)
        session.request = Mock(
            side_effect=[
                {"ok": False, "message": "activation refused"},
                {"ok": True, "shutdown": True},
            ]
        )
        session.close = Mock()
        with TemporaryDirectory() as directory:
            args = self._args(directory)
            with (
                patch.object(runner, "parse_args", return_value=args),
                patch.object(runner, "G2LiveObservationClient") as observation_class,
                patch.object(runner, "PolicyClient"),
                patch.object(runner, "inspect_standby_bridge", return_value={}),
                patch.object(runner, "prepare_model"),
                patch.object(runner, "PlacementBridgeSession", return_value=session),
                patch.object(runner, "execute_action_chunk") as execute,
                patch("builtins.print"),
            ):
                observation = observation_class.return_value.__enter__.return_value
                observation.get_info.return_value = {"control_api_exposed": False}
                with self.assertRaisesRegex(RuntimeError, "activation refused"):
                    runner.main()
                execute.assert_not_called()
                observation.get_snapshot.assert_not_called()
            report = json.loads(args.report.read_text())
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["cycles"], [])
        self.assertEqual(
            [call.args[0]["op"] for call in session.request.call_args_list],
            ["activate", "shutdown"],
        )
        session.close.assert_called_once()

    def test_default_cycle_limit_remains_unbounded(self):
        with patch(
            "sys.argv",
            [
                "runner",
                "--execute",
                "--confirm",
                runner.CONFIRMATION,
                "--report",
                "/tmp/not-written-place-report.json",
            ],
        ):
            self.assertEqual(runner.parse_args().max_cycles, 0)

    def test_failed_cycle_report_keeps_targets_and_original_error_after_shutdown_failure(self):
        targets = np.tile(np.append(TRAINING_REFERENCE, -0.01), (HORIZON, 1))
        targets[:, 0] += np.arange(HORIZON) * 0.001
        status = self._status()
        status["desired_pose"] = TRAINING_REFERENCE.tolist()
        attempted = []

        def request(payload):
            if payload["op"] == "activate":
                return {"ok": True}
            if payload["op"] == "info":
                return self._info()
            if payload["op"] == "status":
                return status
            if payload["op"] == "shutdown":
                raise ConnectionError("shutdown connection lost")
            attempted.append(payload)
            if len(attempted) == 3:
                return {
                    "ok": False,
                    "message": "collision imminent",
                    "command_id": payload["command_id"],
                }
            return {"ok": True, "command_id": payload["command_id"]}

        session = object.__new__(PlacementBridgeSession)
        session.request = Mock(side_effect=request)
        session.close = Mock()
        snapshot = SimpleNamespace(
            metadata={
                "right_eef_xyz_quaternion_xyzw": TRAINING_REFERENCE.tolist(),
                "right_gripper": {"training_position": TRAINING_GRIPPER_REFERENCE},
                "camera_skew_ms": 0.0,
                "maximum_state_camera_skew_ms": 0.0,
            },
            head_color_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            hand_right_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        )
        observation = Mock()
        observation.get_info.return_value = {"control_api_exposed": False}
        observation.get_snapshot.return_value = snapshot
        model = Mock()
        model.ping.return_value = True
        model.get_modality_config.return_value = XICHONG_RIGHT_PLACE_R0002_CONFIG
        model.get_action.return_value = ({}, {})
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(
                initial_pose=TRAINING_REFERENCE,
                initial_gripper=TRAINING_GRIPPER_REFERENCE,
                minimum_retraction_m=MIN_RETRACTION_M,
                observation_port=19100,
                action_port=19200,
                model_port=5564,
                prompt="place the workpiece",
                max_cycles=1,
                report=Path(directory) / "failed.json",
            )
            with (
                patch.object(runner, "parse_args", return_value=args),
                patch.object(runner, "G2LiveObservationClient") as observation_class,
                patch.object(runner, "PolicyClient") as model_class,
                patch.object(
                    runner, "inspect_standby_bridge", return_value={"activation_state": "standby"}
                ),
                patch.object(runner, "PlacementBridgeSession", return_value=session),
                patch.object(runner, "calibrate_bridge_clock", return_value=(0, 0)),
                patch.object(runner, "build_policy_observation", return_value={}),
                patch.object(runner, "decode_action_chunk", return_value=targets),
                patch("agibot.scripts.run_g2_groot_full_protected_inference.time.sleep"),
            ):
                observation_class.return_value.__enter__.return_value = observation
                model_class.return_value.__enter__.return_value = model
                with self.assertRaisesRegex(RuntimeError, "collision imminent"):
                    runner.main()
            report = json.loads(args.report.read_text())
        self.assertEqual(report["status"], "FAILED")
        self.assertIn("collision imminent", report["error"])
        self.assertIn("shutdown connection lost", " ".join(report["error_notes"]))
        self.assertEqual(len(report["cycles"]), 1)
        cycle = report["cycles"][0]
        self.assertEqual(cycle["status"], "FAILED")
        np.testing.assert_allclose(cycle["targets"], targets)
        self.assertEqual(len(cycle["executions"]), 3)
        self.assertEqual(cycle["executions"][-1]["status"], "FAILED")
        self.assertEqual(cycle["executions"][-1]["pose"], targets[2, :7].tolist())
        session.close.assert_called_once()

    def test_shutdown_failure_without_an_original_error_is_reported(self):
        session = object.__new__(PlacementBridgeSession)
        session.request = Mock(return_value={"ok": False, "message": "shutdown refused"})
        session.close = Mock()
        with self.assertRaisesRegex(RuntimeError, "shutdown refused"):
            session.__exit__(None, None, None)
        session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
