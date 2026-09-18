#!/usr/bin/env python3
"""Run the complete Xichong right-arm placement policy on an Agibot G2."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agibot.robot.g2_groot_contact_guard import ContactGuard  # noqa: E402
from agibot.robot.g2_groot_gripper_state_machine import (  # noqa: E402
    FEEDBACK_NATIVE_RADIANS,
    feedback_to_command,
)
from agibot.scripts.run_g2_groot_full_protected_inference import (  # noqa: E402
    BRIDGE_SCHEMA,
    HORIZON,
    PersistentBridgeSession,
    calibrate_bridge_clock,
    execute_action_chunk,
    rotation_deg,
    validate_complete_chunk,
    validate_model_modality_config,
)
from agibot.tools.g2_gr00t_shadow_adapter import (  # noqa: E402
    build_policy_observation,
    decode_action_chunk,
)
from agibot.tools.g2_groot_live_observation_client import G2LiveObservationClient  # noqa: E402
from gr00t.policy.server_client import PolicyClient  # noqa: E402


CONFIRMATION = "EXECUTE_G2_GROOT_FULL_RIGHT_ARM_PLACE_INFERENCE"
RUNNER_RELEASE = "g2_groot_complete_place_inference_v1_20260903"
TASK_PROMPT = (
    "The right arm moves the gripped workpiece to the Xichong placement "
    "target, opens the right gripper to release the workpiece, then retracts "
    "the right arm to a safe pose."
)
TRAINING_REFERENCE = np.asarray(
    [
        0.5013904571533203,
        -0.1765020340681076,
        1.0539140701293945,
        0.5225321066771904,
        -0.001878945069661977,
        0.8526121988491865,
        0.0030175205845201216,
    ],
    dtype=np.float64,
)
TRAINING_GRIPPER_REFERENCE = -0.005764516536146402
MAX_INITIAL_REFERENCE_M = 0.005
MAX_INITIAL_REFERENCE_DEG = 2.0
MAX_INITIAL_GRIPPER_ERROR = 0.03
MAX_POSE_MISMATCH_M = 0.005
MAX_POSE_MISMATCH_DEG = 2.0
RELEASE_INTENT = -0.60
FULLY_OPEN = -0.72
MIN_RETRACTION_M = 0.13
ACTIVATION_PROTOCOL = "explicit_activate_v1"
ACTIVATION_CONFIRMATION = "ACTIVATE_G2_GROOT_PLACE_ARM"


class PlacementBridgeSession(PersistentBridgeSession):
    """Stop the bridge while preserving its final gripper command."""

    compact_status = False

    def request(self, payload: dict) -> dict:
        if self.compact_status and payload.get("op") == "status":
            # Retain ALL recent receipts: chunk completion requires evidence
            # for every one of the 16 rows, not just the latest incremental poll.
            payload = {**payload, "compact": True}
        return super().request(payload)

    def __exit__(self, *exc_info: object) -> None:
        cleanup_errors = []
        try:
            response = self.request({"op": "shutdown"})
            if not response.get("ok"):
                raise RuntimeError(f"bridge shutdown failed: {response}")
        except Exception as error:
            cleanup_errors.append(f"shutdown: {type(error).__name__}: {error}")
        try:
            super().__exit__(*exc_info)
        except Exception as error:
            cleanup_errors.append(f"close: {type(error).__name__}: {error}")
        if cleanup_errors:
            primary_error = exc_info[1]
            if isinstance(primary_error, BaseException):
                for message in cleanup_errors:
                    primary_error.add_note(message)
            else:
                raise RuntimeError("; ".join(cleanup_errors))


def configure_transport_optimization(client, info: dict, enabled: bool) -> dict:
    if enabled and (
        info.get("gripper_command_mode") != "native_trajectory_tracking"
        or info.get("gripper_policy_mode") != "per_row_absolute"
        or info.get("acknowledgement") != "immediate_queue_acceptance"
    ):
        raise RuntimeError("transport optimization requires the immediate-ACK native bridge")
    client.compact_status = bool(enabled)
    return {
        "enabled": bool(enabled),
        "native_ack_pacing": bool(enabled),
        "compact_status_requested": bool(enabled),
        "policy": "official_synchronous_h16_no_rtc",
        "model_waypoint_hz": 10.0,
        "control_hz": 50.0,
        "payload_hashes": False,
    }


@dataclass
class PlacementProgress:
    """Observe task completion without modifying any policy action."""

    release_pose: np.ndarray | None = None
    release_cycle: int | None = None
    minimum_retraction_m: float = MIN_RETRACTION_M
    release_intent_pose: np.ndarray | None = None
    release_intent_cycle: int | None = None
    release_evidence: dict | None = None

    def observe_targets(self, targets: np.ndarray, cycle: int) -> None:
        if self.release_intent_pose is not None:
            return
        released = np.flatnonzero(targets[:, 7] <= RELEASE_INTENT)
        if released.size:
            self.release_intent_pose = targets[int(released[0]), :7].copy()
            self.release_intent_cycle = int(cycle)

    def observe_executions(self, executions: list[dict], cycle: int) -> None:
        if self.release_evidence is not None:
            return
        for execution in executions:
            completion = execution.get("completion") or {}
            event = (
                completion.get("gripper_release_event") if isinstance(completion, dict) else None
            )
            if (
                execution.get("status") == "COMPLETED"
                and isinstance(completion, dict)
                and completion.get("accepted") is True
                and completion.get("ok") is not False
                and not completion.get("error")
                and isinstance(event, dict)
            ):
                try:
                    final_position = float(event["feedback_position"])
                    release_pose = np.asarray(event["pose"], dtype=np.float64)
                    measured_monotonic_s = float(event["monotonic_s"])
                    command_id = event["command_id"]
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    if (
                        np.isfinite(final_position)
                        and -0.785 - 1e-4 <= final_position <= FULLY_OPEN
                        and release_pose.shape == (7,)
                        and np.isfinite(release_pose).all()
                        and np.isfinite(measured_monotonic_s)
                        and measured_monotonic_s >= 0.0
                        and isinstance(command_id, str)
                        and command_id.strip()
                    ):
                        # The native worker records the first healthy physical
                        # open crossing alongside TF, including crossings in
                        # idle ticks. Later receipts carry that same event;
                        # neither a policy target nor an ACK proves release.
                        self.release_pose = release_pose.copy()
                        self.release_cycle = int(cycle)
                        self.release_evidence = {
                            "command_id": command_id,
                            "final_position": final_position,
                            "monotonic_s": measured_monotonic_s,
                            "source": "native_gripper_release_event",
                        }
                        return
            handoff = execution.get("gripper_handoff") or execution.get("release") or {}
            if handoff.get("status") != "COMPLETED" or handoff.get("direction") != "opening":
                continue
            try:
                final_position = float(handoff["final_position"])
                release_pose = np.asarray(handoff["arm_pose_before_tool"], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not np.isfinite(final_position)
                or final_position > FULLY_OPEN
                or release_pose.shape != (7,)
                or not np.isfinite(release_pose).all()
            ):
                continue
            self.release_pose = release_pose.copy()
            self.release_cycle = int(cycle)
            self.release_evidence = {
                "command_id": execution.get("command_id"),
                "final_position": final_position,
            }
            return

    def status(self, bridge_status: dict) -> dict:
        gripper = bridge_status.get("right_gripper") or {}
        completed = float(gripper.get("last_completed", 0.0))
        observation = gripper.get("last_observation") or {}
        raw_position = float(observation.get("raw_position", 0.0))
        feedback_encoding = str(gripper.get("feedback_encoding", FEEDBACK_NATIVE_RADIANS))
        actual_position = feedback_to_command(raw_position, feedback_encoding)
        live = np.asarray(bridge_status["live_pose"], dtype=np.float64)
        retraction_m = (
            None
            if self.release_pose is None
            else float(np.linalg.norm(live[:3] - self.release_pose[:3]))
        )
        healthy = bool(
            bridge_status.get("ok") is not False
            and bridge_status.get("ready")
            and not bridge_status.get("fatal_error")
            and bridge_status.get("live_pose_stale") is False
            and not gripper.get("fault")
            and not gripper.get("recovery_requested")
        )
        passed = bool(
            self.release_pose is not None
            and self.release_evidence is not None
            and healthy
            and actual_position <= FULLY_OPEN
            and retraction_m is not None
            and retraction_m >= self.minimum_retraction_m
        )
        return {
            "release_seen": self.release_pose is not None,
            "release_cycle": self.release_cycle,
            "release_pose": (None if self.release_pose is None else self.release_pose.tolist()),
            "release_evidence": self.release_evidence,
            "release_intent_seen": self.release_intent_pose is not None,
            "release_intent_cycle": self.release_intent_cycle,
            "gripper_last_completed": completed,
            "gripper_actual_position": actual_position,
            "retraction_m": retraction_m,
            "completion_status_healthy": healthy,
            "passed": passed,
        }


def validate_preflight(
    info: dict,
    status: dict,
    training_reference: np.ndarray = TRAINING_REFERENCE,
    training_gripper_reference: float = TRAINING_GRIPPER_REFERENCE,
) -> dict:
    if info.get("schema") != BRIDGE_SCHEMA:
        raise RuntimeError("unexpected action bridge schema")
    if info.get("right_gripper_protected_closure_enabled") is not True:
        raise RuntimeError("full-range gripper control is not enabled")
    if info.get("shutdown_gripper_action") != "hold":
        raise RuntimeError("placement bridge must hold gripper state on shutdown")
    gripper_modes = (info.get("gripper_command_mode"), info.get("gripper_policy_mode"))
    gripper_mappings = {
        (
            "process_handoff",
            "endpoint_process_handoff",
        ): "official_absolute_targets_with_endpoint_process_handoff",
        (
            "native_trajectory_tracking",
            "per_row_absolute",
        ): "official_absolute_targets_native_unified_trajectory_tracking",
    }
    if gripper_modes not in gripper_mappings:
        raise RuntimeError(
            "placement bridge must advertise an explicit supported gripper command/policy pair"
        )
    if info.get("right_arm_enabled") is not True:
        raise RuntimeError("right arm is not enabled")
    if (
        info.get("left_arm_enabled") is not False
        or info.get("head_waist_chassis_enabled") is not False
    ):
        raise RuntimeError("forbidden robot group is enabled")
    if float(info.get("model_waypoint_hz", 0.0)) != 10.0:
        raise RuntimeError("action bridge is not configured for 10 Hz waypoints")
    if float(info.get("control_hz", 0.0)) != 50.0:
        raise RuntimeError("action bridge is not configured for 50 Hz control")
    if not status.get("ready") or status.get("fatal_error"):
        raise RuntimeError("action bridge is not ready")

    live = np.asarray(status["live_pose"], dtype=np.float64)
    initial_error_m = float(np.linalg.norm(live[:3] - training_reference[:3]))
    initial_error_deg = rotation_deg(live, training_reference)
    if initial_error_m > MAX_INITIAL_REFERENCE_M:
        raise RuntimeError(
            f"initial EEF translation error {initial_error_m:.6f} m exceeds "
            f"{MAX_INITIAL_REFERENCE_M:.6f} m"
        )
    if initial_error_deg > MAX_INITIAL_REFERENCE_DEG:
        raise RuntimeError(
            f"initial EEF rotation error {initial_error_deg:.3f} deg exceeds "
            f"{MAX_INITIAL_REFERENCE_DEG:.3f} deg"
        )
    gripper = status.get("right_gripper") or {}
    if gripper.get("fault") or gripper.get("recovery_requested"):
        raise RuntimeError("gripper state machine is not clean")
    observation = gripper.get("last_observation") or {}
    if "raw_position" not in observation:
        raise RuntimeError("gripper physical position feedback is unavailable")
    actual_gripper = feedback_to_command(
        float(observation["raw_position"]),
        str(gripper.get("feedback_encoding", FEEDBACK_NATIVE_RADIANS)),
    )
    gripper_error = abs(actual_gripper - training_gripper_reference)
    if gripper_error > MAX_INITIAL_GRIPPER_ERROR:
        raise RuntimeError(
            f"initial gripper error {gripper_error:.6f} exceeds {MAX_INITIAL_GRIPPER_ERROR:.6f}"
        )
    return {
        "initial_error_m": initial_error_m,
        "initial_error_deg": initial_error_deg,
        "initial_gripper": actual_gripper,
        "initial_gripper_error": gripper_error,
        "horizon": HORIZON,
        "model_waypoint_hz": 10.0,
        "controller_hz": 50.0,
        "gripper_mapping": gripper_mappings[gripper_modes],
        "gripper_command_mode": gripper_modes[0],
        "gripper_policy_mode": gripper_modes[1],
        "gripper_partial_targets": info.get("gripper_partial_targets"),
    }


def validate_snapshot(snapshot: object, bridge_status: dict) -> dict:
    metadata = snapshot.metadata
    if float(metadata["camera_skew_ms"]) > 100.0:
        raise RuntimeError("camera pair skew exceeds 100 ms")
    if float(metadata["maximum_state_camera_skew_ms"]) > 50.0:
        raise RuntimeError("state/camera skew exceeds 50 ms")
    observed = np.asarray(metadata["right_eef_xyz_quaternion_xyzw"], dtype=np.float64)
    live = np.asarray(bridge_status["live_pose"], dtype=np.float64)
    mismatch_m = float(np.linalg.norm(live[:3] - observed[:3]))
    mismatch_deg = rotation_deg(live, observed)
    return {
        "camera_skew_ms": float(metadata["camera_skew_ms"]),
        "maximum_state_camera_skew_ms": float(metadata["maximum_state_camera_skew_ms"]),
        "pose_mismatch_m": mismatch_m,
        "pose_mismatch_deg": mismatch_deg,
        # These two bridges sample the same TF at different instants.  During
        # motion the delta is useful telemetry, but it is not a valid reason
        # to interrupt the official policy loop.
        "pose_mismatch_within_diagnostic_tolerance": bool(
            mismatch_m <= MAX_POSE_MISMATCH_M and mismatch_deg <= MAX_POSE_MISMATCH_DEG
        ),
    }


def validate_activation_state(info: dict, status: dict, expected: str) -> dict:
    """Require the deferred-start contract before warmup or policy execution."""
    for label, response in (("info", info), ("status", status)):
        if response.get("ok") is False:
            raise RuntimeError(f"action bridge {label} failed: {response}")
        if response.get("activation_protocol") != ACTIVATION_PROTOCOL:
            raise RuntimeError("action bridge does not support explicit deferred activation")
        if response.get("activation_state") != expected or response.get("arm_owner_active") is not (
            expected == "active"
        ):
            raise RuntimeError(f"action bridge {label} is not {expected}: {response}")
    return {
        "activation_protocol": ACTIVATION_PROTOCOL,
        "activation_state": expected,
        "arm_owner_active": expected == "active",
    }


def inspect_standby_bridge(host: str, port: int) -> dict:
    # The generic session only closes its socket. A placement session would
    # also send shutdown, which would incorrectly tear down the waiting mux.
    with PersistentBridgeSession(host, port) as client:
        return validate_activation_state(
            client.request({"op": "info"}),
            client.request({"op": "status"}),
            "standby",
        )


def prepare_model(
    model_client: PolicyClient,
    obs_client: G2LiveObservationClient,
    prompt: str,
    preparation: dict,
) -> None:
    """Warm the official policy without an active arm owner or queued targets.

    The external mux must advertise standby before this helper is called.
    Discard the warmup result: every executed H16 uses a new live observation.
    """
    started = time.monotonic()
    preparation.update(status="STARTED", warmup_actions_executed=False)
    try:
        stage_started = time.monotonic()
        if not model_client.ping():
            raise RuntimeError("model server ping failed")
        preparation["ping_s"] = time.monotonic() - stage_started
        stage_started = time.monotonic()
        validate_model_modality_config(model_client.get_modality_config())
        preparation["modality_validation_s"] = time.monotonic() - stage_started
        stage_started = time.monotonic()
        snapshot = obs_client.get_snapshot()
        pose = np.asarray(snapshot.metadata["right_eef_xyz_quaternion_xyzw"], dtype=np.float64)
        preparation["observation_s"] = time.monotonic() - stage_started
        stage_started = time.monotonic()
        action, _ = model_client.get_action(
            build_policy_observation(
                snapshot.head_color_rgb,
                snapshot.hand_right_rgb,
                pose,
                float(snapshot.metadata["right_gripper"]["training_position"]),
                prompt,
            )
        )
        preparation["warmup_inference_s"] = time.monotonic() - stage_started
        targets = decode_action_chunk(action)
        validate_complete_chunk(pose, targets)
        # This is the official policy reset endpoint, not a robot reset. The
        # current Gr00tPolicy is stateless; no warmup episode state is reused.
        model_client.reset()
        preparation.update(status="READY", warmup_horizon=len(targets))
    except Exception as error:
        preparation.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        preparation["total_s"] = time.monotonic() - started


def validate_execution_options(info: dict, args, *, activated: bool = False):
    if getattr(args, "require_native_collision_latch", False):
        latch = info.get("native_collision_latch") or {}
        if (
            latch.get("protocol") != "g2_native_collision_latch_v1"
            or latch.get("required_control_mode") != getattr(args, "required_control_mode", 3)
            or latch.get("minimum_recovery_ms", 0) < 500
            or latch.get("fault") is not None
        ):
            raise RuntimeError("native collision latch contract missing or mismatched")
        if activated:
            config = latch.get("configuration_at_arm") or {}
            if (
                latch.get("armed") is not True or config.get("is_enabled") is not True
                or config.get("checkout_timeout_ms", 0) < 500
            ):
                raise RuntimeError("native collision latch did not arm")
    if getattr(args, "freeze_compensation_after_calibration", False):
        if info.get("compensation_mode") != "calibration_only":
            raise RuntimeError("bridge must use calibration-only pose compensation")


def chunk_end_status(client, completion_status: dict | None = None):
    """Use the just-validated final receipt's snapshot, not a second network poll.

    This snapshot is used only for this completed chunk. The next inference
    still obtains a new camera observation and a new robot status.
    """
    if completion_status is None:
        return client.request({"op": "status"})
    if (
        completion_status.get("ok") is not True
        or completion_status.get("queue_depth") != 0
        or not completion_status.get("ready")
        or completion_status.get("fatal_error")
        or "live_pose" not in completion_status
        or "right_gripper" not in completion_status
    ):
        raise RuntimeError("missing or unhealthy final chunk status; do not advance")
    return completion_status


def validate_contact_guard_contract(expected: ContactGuard, advertised: dict | None):
    expected.ensure_ready()
    if (
        not isinstance(advertised, dict)
        or advertised.get("configured") is not True
        or advertised.get("fault") is not None
        or advertised.get("schema") != expected.snapshot()["schema"]
        or advertised.get("config") != expected.snapshot()["config"]
    ):
        raise RuntimeError("contact guard missing, faulted, or configuration mismatch; not activating")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--observation-port", type=int, default=19100)
    parser.add_argument("--action-port", type=int, default=19200)
    parser.add_argument("--model-port", type=int, default=5564)
    parser.add_argument("--contact-guard-config", type=Path)
    parser.add_argument("--require-native-collision-latch", action="store_true")
    parser.add_argument("--required-control-mode", type=int, choices=(2, 3), default=3)
    parser.add_argument("--freeze-compensation-after-calibration", action="store_true")
    parser.add_argument(
        "--native-chunk-submission", action="store_true",
        help="candidate: submit H16 once for robot-local pacing; requires atomic_h16_native_v1",
    )
    parser.add_argument(
        "--optimize-transport", action="store_true",
        help="native bridge only: compact status and no extra period after a slow ACK",
    )
    parser.add_argument("--prompt", default=TASK_PROMPT)
    parser.add_argument(
        "--initial-pose",
        type=float,
        nargs=7,
        default=TRAINING_REFERENCE.tolist(),
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
    )
    parser.add_argument(
        "--initial-gripper",
        type=float,
        default=TRAINING_GRIPPER_REFERENCE,
    )
    parser.add_argument(
        "--minimum-retraction-m",
        type=float,
        default=MIN_RETRACTION_M,
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="0 runs until placement completes or an operator/hardware stop",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute or args.confirm != CONFIRMATION:
        parser.error(f"physical execution requires --execute --confirm {CONFIRMATION}")
    if args.max_cycles < 0:
        parser.error("--max-cycles must be non-negative")
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.minimum_retraction_m <= 0:
        parser.error("--minimum-retraction-m must be positive")
    return args


def main() -> int:
    args = parse_args()
    optimize_transport = bool(getattr(args, "optimize_transport", False))
    native_chunk_submission = bool(getattr(args, "native_chunk_submission", False))
    report: dict = {
        "schema": "g2_groot_full_place_inference_v1",
        "runner_release": RUNNER_RELEASE,
        "generated_at": datetime.now().astimezone().isoformat(),
        "task_prompt": args.prompt,
        "status": "STARTED",
        "cycles": [],
    }
    training_reference = np.asarray(args.initial_pose, dtype=np.float64)
    progress = PlacementProgress(minimum_retraction_m=args.minimum_retraction_m)
    try:
        expected_guard = None
        if getattr(args, "contact_guard_config", None) is not None:
            expected_guard = ContactGuard.from_file(args.contact_guard_config)
            expected_guard.ensure_ready()
            report["contact_guard_config"] = expected_guard.snapshot()["config"]
        with ExitStack() as stack:
            obs_client = stack.enter_context(
                G2LiveObservationClient(
                    "127.0.0.1", args.observation_port, compute_payload_hashes=False
                )
            )
            model_client = stack.enter_context(
                PolicyClient(host="127.0.0.1", port=args.model_port, timeout_ms=60000)
            )
            observation_info = obs_client.get_info()
            if observation_info.get("control_api_exposed") is not False:
                raise RuntimeError("observation bridge is not read-only")
            report["action_bridge_startup"] = inspect_standby_bridge("127.0.0.1", args.action_port)
            report["model_preparation"] = {}
            prepare_model(model_client, obs_client, args.prompt, report["model_preparation"])
            print(json.dumps({"event": "model_ready", **report["model_preparation"]}), flush=True)
            action_client = stack.enter_context(
                PlacementBridgeSession("127.0.0.1", args.action_port)
            )
            if getattr(args, "require_native_collision_latch", False) or getattr(args, "freeze_compensation_after_calibration", False):
                validate_execution_options(action_client.request({"op": "info"}), args)
            if expected_guard is not None:
                guard_info = action_client.request({"op": "info"}).get("contact_guard")
                validate_contact_guard_contract(expected_guard, guard_info)
            if native_chunk_submission:
                batch_info = action_client.request({"op": "info"})
                if (
                    batch_info.get("chunk_submission") != "atomic_h16_native_v1"
                    or "execute_h16_gripper" not in batch_info.get("operations", [])
                    or batch_info.get("gripper_command_mode") != "native_trajectory_tracking"
                    or batch_info.get("model_waypoint_hz") != 10.0
                    or batch_info.get("control_hz") != 50.0
                    or batch_info.get("interpolation_ticks") != 5
                ):
                    raise RuntimeError("native H16 transport contract not supported; not activating")
            # Verify the optional pacing contract before enabling the owner.
            report["transport"] = configure_transport_optimization(
                action_client,
                action_client.request({"op": "info"}) if optimize_transport else {},
                optimize_transport,
            )
            report["transport"]["native_chunk_submission"] = native_chunk_submission
            activation_started = time.monotonic()
            activation = action_client.request(
                {"op": "activate", "confirm": ACTIVATION_CONFIRMATION}
            )
            if activation.get("ok") is not True:
                raise RuntimeError(f"action bridge activation failed: {activation}")
            info = action_client.request({"op": "info"})
            validate_execution_options(info, args, activated=True)
            report["execution_options"] = {
                "native_collision_latch": info.get("native_collision_latch"),
                "compensation_mode": info.get("compensation_mode"),
            }
            bridge_status = action_client.request({"op": "status"})
            report["action_bridge_startup"].update(
                validate_activation_state(info, bridge_status, "active"),
                activation_s=time.monotonic() - activation_started,
                activated_after_model_ready=True,
            )
            clock_offset_ns, clock_rtt_ns = calibrate_bridge_clock(action_client)
            if clock_rtt_ns > 500_000_000:
                raise RuntimeError("action bridge clock calibration RTT exceeds 0.5 s")
            bridge_status = action_client.request({"op": "status"})
            report["preflight"] = validate_preflight(
                info,
                bridge_status,
                training_reference,
                args.initial_gripper,
            )
            report["preflight"].update(
                {
                    "observation_bridge_read_only": True,
                    "clock_offset_ns": clock_offset_ns,
                    "clock_calibration_rtt_ns": clock_rtt_ns,
                    "training_reference": training_reference.tolist(),
                    "training_gripper_reference": args.initial_gripper,
                    "minimum_retraction_m": args.minimum_retraction_m,
                }
            )
            cycle = 0
            while args.max_cycles == 0 or cycle < args.max_cycles:
                cycle_started = time.monotonic()
                snapshot = obs_client.get_snapshot()
                observation_finished = time.monotonic()
                bridge_status = action_client.request({"op": "status"})
                status_finished = time.monotonic()
                timing = validate_snapshot(snapshot, bridge_status)
                timing.update(
                    observation_request_s=observation_finished - cycle_started,
                    pre_action_status_request_s=status_finished - observation_finished,
                )
                pose = np.asarray(
                    snapshot.metadata["right_eef_xyz_quaternion_xyzw"],
                    dtype=np.float64,
                )
                observed_gripper = float(snapshot.metadata["right_gripper"]["training_position"])
                started = time.monotonic()
                action, _ = model_client.get_action(
                    build_policy_observation(
                        snapshot.head_color_rgb,
                        snapshot.hand_right_rgb,
                        pose,
                        observed_gripper,
                        args.prompt,
                    )
                )
                inference_s = time.monotonic() - started
                targets = decode_action_chunk(action).astype(np.float64)
                chunk_metrics = validate_complete_chunk(
                    np.asarray(bridge_status["desired_pose"], dtype=np.float64),
                    targets,
                )
                progress.observe_targets(targets, cycle)
                row = {
                    "cycle": cycle,
                    "inference_s": inference_s,
                    "timing": timing,
                    "targets": targets.tolist(),
                    "chunk_metrics": chunk_metrics,
                    "executions": [],
                    "status": "EXECUTING",
                }
                report["cycles"].append(row)
                if expected_guard is not None:
                    for target in targets:
                        expected_guard.check_pose(target[:7], source="model_target")
                execution_started = time.monotonic()
                completion_status = {} if native_chunk_submission else None
                execute_action_chunk(
                    action_client,
                    targets,
                    f"place-c{cycle}",
                    clock_offset_ns,
                    execution_rows=row["executions"],
                    **({"native_ack_pacing": True} if optimize_transport else {}),
                    **({"native_chunk_submission": True} if native_chunk_submission else {}),
                    **({"completion_status_out": completion_status} if native_chunk_submission else {}),
                )
                timing["chunk_submit_and_completion_wait_s"] = time.monotonic() - execution_started
                progress.observe_executions(row["executions"], cycle)
                end_status_started = time.monotonic()
                end_status = chunk_end_status(action_client, completion_status)
                timing["post_action_status_request_s"] = time.monotonic() - end_status_started
                timing["reused_final_completion_status"] = native_chunk_submission
                timing["cycle_elapsed_s"] = time.monotonic() - cycle_started
                task_progress = progress.status(end_status)
                row["task_progress"] = task_progress
                row["status"] = "COMPLETED"
                print(
                    json.dumps(
                        {
                            "cycle": cycle,
                            "inference_s": inference_s,
                            **task_progress,
                        }
                    ),
                    flush=True,
                )
                if task_progress["passed"]:
                    report["status"] = "PASS_PLACED_RELEASED_AND_RETRACTED"
                    report["final_bridge_status"] = end_status
                    break
                cycle += 1
            else:
                report["status"] = "MAX_CYCLES_REACHED"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
        report["error_notes"] = list(getattr(error, "__notes__", []))
        if report["cycles"] and report["cycles"][-1].get("status") == "EXECUTING":
            report["cycles"][-1]["status"] = "FAILED"
            report["cycles"][-1]["error"] = report["error"]
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": report["status"],
                "cycles": len(report["cycles"]),
                "report": str(args.report),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS_PLACED_RELEASED_AND_RETRACTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
