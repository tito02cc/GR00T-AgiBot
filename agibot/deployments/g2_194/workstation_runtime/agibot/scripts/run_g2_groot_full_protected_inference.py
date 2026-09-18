#!/usr/bin/env python3
"""Run the complete right-arm GR00T policy, including protected gripper closure."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agibot.robot.g2_groot_h1_bridge_client import BridgeSession  # noqa: E402
from agibot.scripts.run_g2_groot_live_model_h2 import (  # noqa: E402
    PROMPT,
    TRAINING_REFERENCE,
    results_by_id,
    rotation_deg,
)
from agibot.tools.g2_gr00t_shadow_adapter import (  # noqa: E402
    build_policy_observation,
    decode_action_chunk,
)
from agibot.tools.g2_groot_live_observation_client import G2LiveObservationClient  # noqa: E402
from gr00t.policy.server_client import PolicyClient  # noqa: E402


CONFIRMATION = "EXECUTE_G2_GROOT_FULL_RIGHT_ARM_PROTECTED_INFERENCE"
RUNNER_RELEASE = "g2_groot_complete_inference_v1_20260827"
BRIDGE_SCHEMA = "g2_groot_persistent_h1_action_bridge_v3"
HORIZON = 16
MODEL_PERIOD_S = 0.1
CHUNK_COMPLETION_TIMEOUT_S = 25.0
OPEN = -0.785
PROTECTED_MAXIMUM = 0.0
MAX_MODEL_STEP_M = 0.035
MAX_MODEL_ROTATION_DEG = 3.2
MAX_POSE_MISMATCH_M = 0.0005
MAX_POSE_MISMATCH_DEG = 0.5
CLOSE_INTENT = -0.70
FULLY_CLOSED = -0.10
MIN_LIFT_M = 0.03
MAX_INITIAL_REFERENCE_M = 0.0002
MAX_INITIAL_REFERENCE_DEG = 0.02
WORKSPACE_MIN = np.asarray([0.503, -0.311, 1.005], dtype=np.float64)
WORKSPACE_MAX = np.asarray([0.746, -0.227, 1.081], dtype=np.float64)


@dataclass
class GripperCommandLatch:
    """Pass official absolute gripper actions through unchanged.

    The state only records whether closure has begun for grasp/lift reporting;
    it does not clip, normalize, latch, smooth, or force the policy command.
    """

    close_requested: bool = False
    desired: float = OPEN

    def map(self, policy_value: float) -> float:
        value = float(policy_value)
        if not np.isfinite(value):
            raise ValueError("non-finite policy gripper target")
        self.close_requested = self.close_requested or value > CLOSE_INTENT
        self.desired = value
        return self.desired


class PersistentBridgeSession(BridgeSession):
    """Close the client socket without forcing the gripper open on errors."""


def validate_complete_chunk(current: np.ndarray, targets: np.ndarray) -> list[dict]:
    if targets.shape != (HORIZON, 8):
        raise ValueError(f"expected complete ({HORIZON},8) action chunk, got {targets.shape}")
    if not np.isfinite(targets).all():
        raise ValueError("action chunk contains NaN/Inf")
    previous = np.asarray(current, dtype=np.float64)
    metrics = []
    for index, target in enumerate(targets):
        metrics.append(
            {
                "index": index,
                "translation_m": float(np.linalg.norm(target[:3] - previous[:3])),
                "rotation_deg": rotation_deg(previous, target[:7]),
                "gripper": float(target[7]),
            }
        )
        previous = target[:7]
    return metrics


def calibrate_bridge_clock(client: BridgeSession, samples: int = 5) -> tuple[int, int]:
    measurements = []
    for _ in range(samples):
        before_ns = time.time_ns()
        info = client.request({"op": "info"})
        after_ns = time.time_ns()
        server_time_ns = int(info.get("server_time_ns", 0))
        if server_time_ns <= 0:
            raise RuntimeError("action bridge does not expose server_time_ns")
        rtt_ns = after_ns - before_ns
        midpoint_ns = (before_ns + after_ns) // 2
        measurements.append((rtt_ns, server_time_ns - midpoint_ns))
    best_rtt_ns, offset_ns = min(measurements, key=lambda item: item[0])
    return offset_ns, best_rtt_ns


def validate_preflight(
    info: dict,
    status: dict,
    training_reference: np.ndarray = TRAINING_REFERENCE,
) -> dict:
    if info.get("schema") != BRIDGE_SCHEMA:
        raise RuntimeError("unexpected action bridge schema")
    if info.get("gripper_policy_mode") == "endpoint_process_handoff":
        raise RuntimeError(
            "full grasp requires per-row gripper control; endpoint process handoff "
            "defers partial gripper targets and cannot provide that contract"
        )
    if info.get("right_gripper_protected_closure_enabled") is not True:
        raise RuntimeError("protected closure is not enabled")
    if info.get("right_arm_enabled") is not True:
        raise RuntimeError("right arm is not enabled")
    if (
        info.get("left_arm_enabled") is not False
        or info.get("head_waist_chassis_enabled") is not False
    ):
        raise RuntimeError("forbidden robot group is enabled")
    if float(info.get("model_waypoint_hz", 0.0)) != 10.0:
        raise RuntimeError("action bridge is not configured for 10 Hz waypoints")
    if not status.get("ready") or status.get("fatal_error"):
        raise RuntimeError("action bridge is not ready")
    initial_live = np.asarray(status["live_pose"], dtype=np.float64)
    initial_error_m = float(np.linalg.norm(initial_live[:3] - training_reference[:3]))
    initial_error_deg = rotation_deg(initial_live, training_reference)
    initial_gripper = status.get("right_gripper") or {}
    if initial_gripper.get("fault") or initial_gripper.get("recovery_requested"):
        raise RuntimeError("gripper state machine is not clean")
    if abs(float(initial_gripper.get("last_completed", 0.0)) - OPEN) > 0.01:
        raise RuntimeError("gripper is not fully open at policy start")
    return {
        "initial_error_m": initial_error_m,
        "initial_error_deg": initial_error_deg,
        "horizon": HORIZON,
        "model_waypoint_hz": 10.0,
        "gripper_mapping": "official_absolute_action_passthrough",
    }


def _enum_name(value: object) -> str:
    return str(getattr(value, "name", value)).split(".")[-1].upper()


def validate_model_modality_config(configs: dict) -> None:
    expected_keys = {
        "video": ["head_color", "hand_right"],
        "state": ["right_eef", "right_gripper"],
        "action": ["right_eef", "right_gripper"],
        "language": ["annotation.human.task_description"],
    }
    for group, keys in expected_keys.items():
        config = configs.get(group)
        if config is None or list(config.modality_keys) != keys:
            raise RuntimeError(f"unexpected model {group} modality keys")
    action = configs["action"]
    if list(action.delta_indices) != list(range(HORIZON)):
        raise RuntimeError("model action horizon is not exactly 16")
    action_configs = list(action.action_configs or [])
    if len(action_configs) != 2:
        raise RuntimeError("model action config count is not 2")
    expected = [
        ("RELATIVE", "EEF", "XYZ_ROT6D", "right_eef"),
        ("ABSOLUTE", "NON_EEF", "DEFAULT", "right_gripper"),
    ]
    actual = [
        (
            _enum_name(item.rep),
            _enum_name(item.type),
            _enum_name(item.format),
            str(item.state_key),
        )
        for item in action_configs
    ]
    if actual != expected:
        raise RuntimeError(f"unexpected model action semantics: {actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--observation-port", type=int, default=19100)
    parser.add_argument("--action-port", type=int, default=19200)
    parser.add_argument("--model-port", type=int, default=5564)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument(
        "--initial-pose",
        type=float,
        nargs=7,
        default=TRAINING_REFERENCE.tolist(),
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
    )
    parser.add_argument("--minimum-task-progress-m", type=float, default=MIN_LIFT_M)
    parser.add_argument(
        "--max-cycles", type=int, default=0, help="0 means run until task success or hardware stop"
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute or args.confirm != CONFIRMATION:
        parser.error(f"physical execution requires --execute --confirm {CONFIRMATION}")
    return args


def wait_result(
    client: BridgeSession,
    command_id: str,
    timeout_s: float = 4.0,
    *,
    require_settled: bool = True,
) -> dict:
    deadline = time.monotonic() + timeout_s
    last = {}
    while time.monotonic() < deadline:
        last = client.request({"op": "status"})
        if last.get("ok") is False:
            raise RuntimeError(f"bridge status failed for {command_id}: {last}")
        if last.get("fatal_error"):
            raise RuntimeError(str(last["fatal_error"]))
        result = results_by_id(last).get(command_id)
        if result and (
            result.get("ok") is False or result.get("accepted") is False or result.get("error")
        ):
            raise RuntimeError(f"command execution failed: {result}")
        if result and last.get("queue_depth") == 0:
            return last
        time.sleep(0.05)
    raise TimeoutError(f"command did not settle: {command_id}")


def execute_target(
    client: BridgeSession,
    start_pose: np.ndarray,
    target: np.ndarray,
    prefix: str,
    clock_offset_ns: int,
    *,
    settle_final: bool,
) -> tuple[np.ndarray, list[dict]]:
    poses = [target[:7]]
    target_gripper = float(target[7])
    rows = []
    for index, pose in enumerate(poses):
        command_id = f"{prefix}-{index}-{uuid.uuid4()}"
        acknowledgement = {}
        for retry in range(3):
            if retry:
                command_id = f"{prefix}-{index}-retry{retry}-{uuid.uuid4()}"
            acknowledgement = client.request(
                {
                    "op": "execute_h1_gripper",
                    "command_id": command_id,
                    "timestamp_ns": time.time_ns() + clock_offset_ns,
                    "target_pose": pose.tolist(),
                    "target_gripper": target_gripper,
                }
            )
            if acknowledgement.get("ok"):
                break
            if "command age" not in str(acknowledgement.get("message", "")):
                break
        if not acknowledgement.get("ok"):
            raise RuntimeError(f"command rejected: {acknowledgement}")
        status = wait_result(
            client,
            command_id,
            require_settled=settle_final and index == len(poses) - 1,
        )
        rows.append(
            {
                "command_id": command_id,
                "pose": pose.tolist(),
                "gripper": target_gripper,
                "position_error_m": status["live_target_position_error_m"],
                "rotation_error_rad": status["live_target_rotation_error_rad"],
            }
        )
    return target[:7].copy(), rows


def submit_native_action_chunk(client, targets, prefix, clock_offset_ns, rows):
    """One H16 network request; execution receipts are still checked per row.

    An uncertain reply must never be retried: the robot may already be moving.
    """
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (HORIZON, 8) or not np.isfinite(targets).all():
        raise ValueError("native batch requires a finite complete H16 (16, 8)")
    submitted = time.monotonic()
    timestamp_ns = time.time_ns() + clock_offset_ns
    commands = [
        {
            "command_id": f"{prefix}-h{index}-{uuid.uuid4()}",
            "timestamp_ns": timestamp_ns,
            "target_pose": target[:7].tolist(),
            "target_gripper": float(target[7]),
        }
        for index, target in enumerate(targets)
    ]
    batch_rows = [
        {
            "command_id": command["command_id"],
            "pose": command["target_pose"],
            "gripper": command["target_gripper"],
            "submitted_monotonic_s": submitted,
            "transport": "atomic_h16_native_v1",
            "status": "SUBMITTED",
        }
        for command in commands
    ]
    rows.extend(batch_rows)
    try:
        ack = client.request({"op": "execute_h16_gripper", "commands": commands})
        acknowledged = time.monotonic()
        if (
            ack.get("ok") is not True
            or ack.get("accepted") is not True
            or ack.get("chunk_submission") != "atomic_h16_native_v1"
            or ack.get("command_ids") != [command["command_id"] for command in commands]
        ):
            raise RuntimeError(f"invalid H16 acceptance receipt (do not retry): {ack}")
        for row in batch_rows:
            row.update(
                status="ACCEPTED",
                acknowledged_monotonic_s=acknowledged,
                acknowledgement_s=acknowledged - submitted,
            )
    except Exception as error:
        for row in batch_rows:
            row.update(
                status="SUBMISSION_UNCONFIRMED",
                error=f"{type(error).__name__}: {error}",
            )
        raise
    return commands[-1]["command_id"]


def execute_action_chunk(
    client: BridgeSession,
    targets: np.ndarray,
    prefix: str,
    clock_offset_ns: int,
    *,
    execution_rows: list[dict] | None = None,
    native_ack_pacing: bool = False,
    native_chunk_submission: bool = False,
    completion_status_out: dict | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Execute the policy chunk exactly like the official real-robot loop.

    One decoded policy row is submitted every 100 ms.  The G2 bridge performs
    the five 50 Hz interpolation ticks for that row; no extra policy-space
    subdivision, clipping, or continuity gate is applied here. With an
    immediate-acceptance native bridge, a slow network ACK already consumes
    the period: do not add another 100 ms. Each next row remains anchored to
    its own actual submission, never to an overdue catch-up schedule.
    Optional native_chunk_submission sends H16 once and delegates the exact
    same 10 Hz row cadence to the robot, removing per-row network waits.
    """
    rows = [] if execution_rows is None else execution_rows
    if completion_status_out is not None:
        completion_status_out.clear()
    chunk_rows_start = len(rows)
    last_command_id = ""
    if native_chunk_submission:
        last_command_id = submit_native_action_chunk(client, targets, prefix, clock_offset_ns, rows)
    for index, target in enumerate([] if native_chunk_submission else targets):
        command_id = f"{prefix}-h{index}-{uuid.uuid4()}"
        submitted = time.monotonic()
        row = {
            "command_id": command_id,
            "pose": target[:7].tolist(),
            "gripper": float(target[7]),
            "submitted_monotonic_s": submitted,
            "status": "SUBMITTED",
        }
        rows.append(row)
        try:
            acknowledgement = client.request(
                {
                    "op": "execute_h1_gripper",
                    "command_id": command_id,
                    "timestamp_ns": time.time_ns() + clock_offset_ns,
                    "target_pose": target[:7].tolist(),
                    "target_gripper": float(target[7]),
                }
            )
            acknowledged = time.monotonic()
            row["acknowledged_monotonic_s"] = acknowledged
            row["acknowledgement_s"] = acknowledged - submitted
            row["acknowledgement"] = acknowledgement
            for key in ("release", "gripper_handoff", "gripper_execution"):
                if key in acknowledgement:
                    row[key] = acknowledgement[key]
            if not acknowledgement.get("ok"):
                raise RuntimeError(f"command rejected: {acknowledgement}")
            row["status"] = "ACCEPTED"
        except Exception as error:
            row["status"] = "FAILED"
            row["error"] = f"{type(error).__name__}: {error}"
            raise
        last_command_id = command_id
        deadline = submitted + MODEL_PERIOD_S
        row["acknowledgement_overdue"] = acknowledged > deadline
        row["cadence_rebased"] = row["acknowledgement_overdue"] and not native_ack_pacing
        if row["cadence_rebased"]:
            deadline = acknowledged + MODEL_PERIOD_S
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    # A placement waypoint may perform the serialized GDK handoff: allow up
    # to five seconds for physical tool feedback and fifteen seconds for the
    # Cartesian-arm child to restart before declaring the H16 chunk lost.
    # This changes only acknowledgement waiting, never policy targets or the
    # 10 Hz waypoint cadence outside the handoff.
    try:
        status = wait_result(
            client,
            last_command_id,
            timeout_s=CHUNK_COMPLETION_TIMEOUT_S,
            require_settled=False,
        )
    except Exception as error:
        rows[-1]["completion_wait_error"] = f"{type(error).__name__}: {error}"
        raise
    completed = results_by_id(status)
    completion_errors = []
    for row in rows[chunk_rows_start:]:
        result = completed.get(row["command_id"])
        if result is None:
            row["status"] = "COMPLETION_UNCONFIRMED"
            row["error"] = f"missing completion receipt: {row['command_id']}"
            completion_errors.append(row["error"])
        elif result.get("ok") is False or result.get("accepted") is False or result.get("error"):
            row["completion"] = result
            row["status"] = "FAILED"
            row["error"] = f"command execution failed: {result}"
            completion_errors.append(row["error"])
        else:
            row["completion"] = result
            row["status"] = "COMPLETED"
    if completion_errors:
        raise RuntimeError("; ".join(completion_errors))
    if completion_status_out is not None:
        completion_status_out.update(status)
    return targets[-1, :7].copy(), rows


def run_inference(args: argparse.Namespace, report: dict) -> None:
    training_reference = np.asarray(args.initial_pose, dtype=np.float64)
    if not args.prompt.strip():
        raise ValueError("task prompt must not be empty")
    if args.minimum_task_progress_m <= 0:
        raise ValueError("--minimum-task-progress-m must be positive")
    close_pose = None
    gripper_latch = GripperCommandLatch()
    with (
        G2LiveObservationClient("127.0.0.1", args.observation_port) as obs_client,
        PolicyClient(host="127.0.0.1", port=args.model_port, timeout_ms=60000) as model_client,
        PersistentBridgeSession("127.0.0.1", args.action_port) as action_client,
    ):
        observation_info = obs_client.get_info()
        if observation_info.get("control_api_exposed") is not False:
            raise RuntimeError("observation bridge is not read-only")
        info = action_client.request({"op": "info"})
        clock_offset_ns, clock_rtt_ns = calibrate_bridge_clock(action_client)
        if clock_rtt_ns > 500_000_000:
            raise RuntimeError("action bridge clock calibration RTT exceeds 0.5 s")
        initial_status = action_client.request({"op": "status"})
        report["preflight"] = validate_preflight(info, initial_status, training_reference)
        report["preflight"].update(
            {
                "observation_bridge_read_only": True,
                "clock_offset_ns": clock_offset_ns,
                "clock_calibration_rtt_ns": clock_rtt_ns,
                "task_prompt": args.prompt,
                "training_reference": training_reference.tolist(),
                "minimum_task_progress_m": args.minimum_task_progress_m,
            }
        )
        if not model_client.ping():
            raise RuntimeError("model server ping failed")
        validate_model_modality_config(model_client.get_modality_config())

        cycle = 0
        while args.max_cycles == 0 or cycle < args.max_cycles:
            snapshot = obs_client.get_snapshot()
            pose = np.asarray(snapshot.metadata["right_eef_xyz_quaternion_xyzw"], dtype=np.float64)
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
            status = action_client.request({"op": "status"})
            live = np.asarray(status["live_pose"], dtype=np.float64)
            mismatch_m = float(np.linalg.norm(live[:3] - pose[:3]))
            mismatch_deg = rotation_deg(live, pose)
            current = np.asarray(status["desired_pose"], dtype=np.float64)
            chunk_metrics = validate_complete_chunk(current, targets)
            row = {
                "cycle": cycle,
                "inference_s": inference_s,
                "pose_mismatch_m": mismatch_m,
                "pose_mismatch_deg": mismatch_deg,
                "targets": targets.tolist(),
                "chunk_metrics": chunk_metrics,
                "executions": [],
                "status": "EXECUTING",
            }
            report["cycles"].append(row)
            mapped_targets = targets.copy()
            for index, target in enumerate(targets):
                was_close_requested = gripper_latch.close_requested
                mapped_targets[index, 7] = gripper_latch.map(float(target[7]))
                if not was_close_requested and gripper_latch.close_requested:
                    close_pose = target[:7].copy()
                    report["first_close_pose"] = close_pose.tolist()
            current, _ = execute_action_chunk(
                action_client,
                mapped_targets,
                f"full-c{cycle}",
                clock_offset_ns,
                execution_rows=row["executions"],
            )
            row["status"] = "COMPLETED"
            end_status = action_client.request({"op": "status"})
            gripper_status = end_status.get("right_gripper") or {}
            contact = bool(gripper_status.get("contact_latched"))
            fully_closed = float(gripper_status.get("last_completed", OPEN)) >= FULLY_CLOSED
            if close_pose is not None and (contact or fully_closed):
                lift_m = float(end_status["live_pose"][2] - close_pose[2])
                report["lift_m"] = lift_m
                report["contact_latched"] = contact
                if lift_m >= args.minimum_task_progress_m:
                    report["status"] = "PASS_GRASPED_AND_LIFTED"
                    report["final_bridge_status"] = end_status
                    break
            print(
                json.dumps(
                    {
                        "cycle": cycle,
                        "inference_s": inference_s,
                        "gripper": gripper_status.get("last_completed"),
                        "contact": contact,
                        "lift_m": report.get("lift_m"),
                    }
                ),
                flush=True,
            )
            cycle += 1
        else:
            report["status"] = "MAX_CYCLES_REACHED"
        if report["status"] != "PASS_GRASPED_AND_LIFTED":
            action_client.request({"op": "shutdown"})


def main() -> int:
    args = parse_args()
    report = {
        "schema": "g2_groot_full_protected_inference_v1",
        "runner_release": RUNNER_RELEASE,
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "STARTED",
        "cycles": [],
    }
    try:
        run_inference(args, report)
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
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
                "first_close_pose": report.get("first_close_pose"),
                "lift_m": report.get("lift_m"),
                "contact_latched": report.get("contact_latched"),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS_GRASPED_AND_LIFTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
