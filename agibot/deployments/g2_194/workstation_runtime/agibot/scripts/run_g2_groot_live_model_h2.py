#!/usr/bin/env python3
"""Execute one strictly guarded GR00T chunk, then reverse it."""

from __future__ import annotations

import argparse
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
from agibot.tools.g2_gr00t_shadow_adapter import (  # noqa: E402
    build_policy_observation,
    check_shadow_target,
    decode_action_chunk,
)
from agibot.tools.g2_groot_live_observation_client import G2LiveObservationClient  # noqa: E402
from gr00t.policy.server_client import PolicyClient  # noqa: E402


CONFIRMATIONS = {
    2: "EXECUTE_G2_GROOT_MODEL_H2_RIGHT_ARM_ONLY",
    4: "EXECUTE_G2_GROOT_MODEL_H4_RIGHT_ARM_ONLY",
}
TINY_GRIPPER_CONFIRMATIONS = {
    2: "EXECUTE_G2_GROOT_MODEL_H2_RIGHT_ARM_ZERO_GRIPPER_MOTION",
    4: "EXECUTE_G2_GROOT_MODEL_H4_RIGHT_ARM_ZERO_GRIPPER_MOTION",
}
ARM_ONLY_ACTION_SCHEMA = "g2_groot_persistent_h1_action_bridge_v2"
TINY_GRIPPER_ACTION_SCHEMA = "g2_groot_persistent_h1_action_bridge_v3"
PROMPT = (
    "starting with the right gripper open and the left arm and left gripper stationary, "
    "grasp the metal_workpiece with the right gripper, lift it at least 3 cm clear of "
    "the pickup position, and hold the right arm stable"
)
MAX_STEP_M = 0.002
MAX_ROTATION_DEG = 0.5
MAX_CUMULATIVE_M = 0.004
MAX_OBSERVATION_TO_COMMAND_S = 1.0
MAX_POSE_MISMATCH_M = 0.00025
MAX_POSE_MISMATCH_DEG = 0.1
OPEN_GRIPPER_MAX = -0.70
MAX_GRIPPER_CHANGE = 0.10
GRIPPER_OPEN_COMMAND = -0.785
MAX_ZERO_MOTION_GRIPPER_DEVIATION = 0.012
SETTLE_POSITION_M = 0.0002
SETTLE_ROTATION_RAD = 0.002
SETTLE_TIMEOUT_S = 2.5
_ACTIVE_REPORT: dict | None = None
_ACTIVE_REPORT_PATH: Path | None = None
TRAINING_REFERENCE = np.asarray(
    [
        0.5081030780841185,
        -0.25893071977192195,
        1.04842646506165,
        0.6620299522145109,
        -0.020616324660443348,
        0.7490960119476248,
        0.01210266138135472,
    ],
    dtype=np.float64,
)
MAX_INITIAL_REFERENCE_M = 0.00005
MAX_INITIAL_REFERENCE_DEG = 0.02


def rotation_deg(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(first[3:7], second[3:7])))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))))


def results_by_id(status: dict) -> dict[str, dict]:
    return {
        str(item["command_id"]): item for item in status.get("recent_results", [])
    }


def send_timed(
    client: BridgeSession,
    targets: list[np.ndarray],
    prefix: str,
    gripper_targets: list[float] | None = None,
) -> tuple[list[str], list[dict], float]:
    if gripper_targets is not None and len(gripper_targets) != len(targets):
        raise ValueError("gripper target count must match pose target count")
    ids = []
    acknowledgements = []
    started = time.monotonic()
    deadline = started
    for index, target in enumerate(targets):
        command_id = f"{prefix}-{index}-{uuid.uuid4()}"
        payload = {
            "op": (
                "execute_h1_gripper"
                if gripper_targets is not None
                else "execute_h1"
            ),
            "command_id": command_id,
            "timestamp_ns": time.time_ns(),
            "target_pose": target.tolist(),
        }
        if gripper_targets is not None:
            payload["target_gripper"] = gripper_targets[index]
        acknowledgement = client.request(payload)
        if not acknowledgement.get("ok"):
            raise RuntimeError(f"command {index} rejected: {acknowledgement}")
        ids.append(command_id)
        acknowledgements.append(acknowledgement)
        deadline += 0.1
        wait = deadline - time.monotonic()
        if wait > 0:
            time.sleep(wait)
    return ids, acknowledgements, time.monotonic() - started


def wait_for_completion(
    client: BridgeSession, command_ids: list[str]
) -> tuple[dict, list[dict | None], float, bool, bool]:
    """Wait for command completion and bounded settling without blocking recovery."""
    started = time.monotonic()
    while True:
        status = client.request({"op": "status"})
        available = results_by_id(status)
        results = [available.get(item) for item in command_ids]
        commands_complete = bool(
            all(results)
            and status.get("ready")
            and status.get("fatal_error") is None
            and status.get("queue_depth") == 0
        )
        settled = bool(
            commands_complete
            and status["live_target_position_error_m"] <= SETTLE_POSITION_M
            and status["live_target_rotation_error_rad"] <= SETTLE_ROTATION_RAD
        )
        elapsed = time.monotonic() - started
        if settled or status.get("fatal_error") is not None or elapsed >= SETTLE_TIMEOUT_S:
            return status, results, elapsed, commands_complete, settled
        time.sleep(0.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--horizon", type=int, choices=(2, 4), default=2)
    parser.add_argument("--tiny-gripper-zero-motion", action="store_true")
    parser.add_argument("--observation-port", type=int, default=19100)
    parser.add_argument("--action-port", type=int, default=19200)
    parser.add_argument("--model-port", type=int, default=5564)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    confirmation = (
        TINY_GRIPPER_CONFIRMATIONS[args.horizon]
        if args.tiny_gripper_zero_motion
        else CONFIRMATIONS[args.horizon]
    )
    if not args.execute or args.confirm != confirmation:
        parser.error(f"physical execution requires --execute --confirm {confirmation}")
    if args.report is None:
        suffix = "_tiny_gripper_zero_motion" if args.tiny_gripper_zero_motion else ""
        date_token = datetime.now().astimezone().strftime("%Y%m%d")
        args.report = (
            REPO_ROOT
            / f"agibot/reports/g2_groot_live_model_h{args.horizon}{suffix}_{date_token}.json"
        )
    return args


def main() -> int:
    global _ACTIVE_REPORT, _ACTIVE_REPORT_PATH
    args = parse_args()
    report = {
        "schema": (
            f"g2_groot_live_model_h{args.horizon}_tiny_gripper_v1"
            if args.tiny_gripper_zero_motion
            else f"g2_groot_live_model_h{args.horizon}_v1"
        ),
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "STARTED",
        "scope": (
            f"{args.horizon} model EEF waypoints at 10 Hz, "
            + (
                "v3 gripper fields constrained to zero hardware motion, "
                if args.tiny_gripper_zero_motion
                else "no gripper, "
            )
            + "then reverse return"
        ),
        "model_waypoints_sent": 0,
        "return_waypoints_sent": 0,
        "gripper_commands_sent": 0,
    }
    _ACTIVE_REPORT = report
    _ACTIVE_REPORT_PATH = args.report
    exit_code = 1
    with (
        G2LiveObservationClient("127.0.0.1", args.observation_port) as observation_client,
        PolicyClient(host="127.0.0.1", port=args.model_port, timeout_ms=60000) as model_client,
        BridgeSession("127.0.0.1", args.action_port) as action_client,
    ):
        observation_info = observation_client.get_info()
        action_info = action_client.request({"op": "info"})
        before = action_client.request({"op": "status"})
        if observation_info.get("control_api_exposed") is not False:
            raise RuntimeError("observation bridge is not read-only")
        expected_action_schema = (
            TINY_GRIPPER_ACTION_SCHEMA
            if args.tiny_gripper_zero_motion
            else ARM_ONLY_ACTION_SCHEMA
        )
        if action_info.get("schema") != expected_action_schema:
            raise RuntimeError("unexpected action bridge schema")
        if args.tiny_gripper_zero_motion:
            if action_info.get("right_gripper_enabled") is not True:
                raise RuntimeError("provisional right gripper is not enabled")
            if action_info.get("right_gripper_full_closure_enabled") is not False:
                raise RuntimeError("full gripper closure is unexpectedly enabled")
            if action_info.get("right_gripper_command_range") != [
                GRIPPER_OPEN_COMMAND,
                -0.76,
            ]:
                raise RuntimeError("unexpected provisional gripper command range")
        elif action_info.get("right_gripper_enabled") is not False:
            raise RuntimeError("gripper unexpectedly enabled")
        if not before.get("ready") or before.get("fatal_error"):
            raise RuntimeError("action bridge is not ready")
        if not model_client.ping():
            raise RuntimeError("model server ping failed")

        # Warm the model before acquiring the freshness-critical observation.
        # Cold-start latency is diagnostic only and can never lead to motion.
        warmup_started = time.monotonic()
        warmup_snapshot_started = time.monotonic()
        warmup_snapshot = observation_client.get_snapshot()
        warmup_observation_s = time.monotonic() - warmup_snapshot_started
        warmup_pose = np.asarray(
            warmup_snapshot.metadata["right_eef_xyz_quaternion_xyzw"],
            dtype=np.float64,
        )
        warmup_gripper = float(
            warmup_snapshot.metadata["right_gripper"]["training_position"]
        )
        warmup_observation = build_policy_observation(
            warmup_snapshot.head_color_rgb,
            warmup_snapshot.hand_right_rgb,
            warmup_pose,
            warmup_gripper,
            PROMPT,
        )
        warmup_inference_started = time.monotonic()
        model_client.get_action(warmup_observation)
        report["warmup"] = {
            "snapshot_index": int(warmup_snapshot.metadata["snapshot_index"]),
            "observation_s": warmup_observation_s,
            "inference_s": time.monotonic() - warmup_inference_started,
            "total_s": time.monotonic() - warmup_started,
            "motion_authorized": False,
        }

        origin = np.asarray(before["desired_pose"], dtype=np.float64)
        started = time.monotonic()
        snapshot_started = time.monotonic()
        snapshot = observation_client.get_snapshot()
        observation_fetch_s = time.monotonic() - snapshot_started
        pose = np.asarray(
            snapshot.metadata["right_eef_xyz_quaternion_xyzw"], dtype=np.float64
        )
        gripper = float(snapshot.metadata["right_gripper"]["training_position"])
        observation = build_policy_observation(
            snapshot.head_color_rgb,
            snapshot.hand_right_rgb,
            pose,
            gripper,
            PROMPT,
        )
        inference_started = time.monotonic()
        action, _ = model_client.get_action(observation)
        model_latency_s = time.monotonic() - inference_started
        targets = decode_action_chunk(action)[: args.horizon].astype(np.float64)
        if targets.shape != (args.horizon, 8):
            raise RuntimeError(
                f"expected {(args.horizon, 8)} decoded actions, got {targets.shape}"
            )
        elapsed_s = time.monotonic() - started

        live = np.asarray(before["live_pose"], dtype=np.float64)
        mismatch_m = float(np.linalg.norm(live[:3] - pose[:3]))
        mismatch_deg = rotation_deg(live, pose)
        initial_reference_m = float(
            np.linalg.norm(pose[:3] - TRAINING_REFERENCE[:3])
        )
        initial_reference_deg = rotation_deg(pose, TRAINING_REFERENCE)
        previous = pose
        step_metrics = []
        violations = []
        previous_gripper = gripper
        for index, target in enumerate(targets):
            check = check_shadow_target(previous, target)
            cumulative = float(np.linalg.norm(target[:3] - pose[:3]))
            gripper_change = abs(float(target[7]) - previous_gripper)
            step_metrics.append(
                {
                    "step": index,
                    "translation_m": check.translation_step_m,
                    "rotation_deg": check.rotation_step_deg,
                    "cumulative_translation_m": cumulative,
                    "predicted_gripper": float(target[7]),
                    "gripper_change": gripper_change,
                }
            )
            if check.translation_step_m > MAX_STEP_M:
                violations.append(f"step_{index}_translation")
            if check.rotation_step_deg > MAX_ROTATION_DEG:
                violations.append(f"step_{index}_rotation")
            if cumulative > MAX_CUMULATIVE_M:
                violations.append(f"step_{index}_cumulative")
            if target[7] > OPEN_GRIPPER_MAX or gripper_change > MAX_GRIPPER_CHANGE:
                violations.append(f"step_{index}_gripper_not_open")
            if (
                args.tiny_gripper_zero_motion
                and abs(float(target[7]) - GRIPPER_OPEN_COMMAND)
                > MAX_ZERO_MOTION_GRIPPER_DEVIATION
            ):
                violations.append(f"step_{index}_gripper_zero_motion_deadband")
            previous = target[:7]
            previous_gripper = float(target[7])
        if elapsed_s > MAX_OBSERVATION_TO_COMMAND_S:
            violations.append("observation_to_command_timeout")
        if mismatch_m > MAX_POSE_MISMATCH_M:
            violations.append("observation_bridge_translation_mismatch")
        if mismatch_deg > MAX_POSE_MISMATCH_DEG:
            violations.append("observation_bridge_rotation_mismatch")
        if initial_reference_m > MAX_INITIAL_REFERENCE_M:
            violations.append("initial_training_reference_translation")
        if initial_reference_deg > MAX_INITIAL_REFERENCE_DEG:
            violations.append("initial_training_reference_rotation")

        report["preflight"] = {
            "snapshot_index": int(snapshot.metadata["snapshot_index"]),
            "current_pose": pose.tolist(),
            "bridge_origin": origin.tolist(),
            "targets": targets.tolist(),
            "model_latency_s": model_latency_s,
            "observation_fetch_s": observation_fetch_s,
            "observation_to_command_s": elapsed_s,
            "observation_bridge_mismatch_m": mismatch_m,
            "observation_bridge_mismatch_deg": mismatch_deg,
            "initial_training_reference_m": initial_reference_m,
            "initial_training_reference_deg": initial_reference_deg,
            "step_metrics": step_metrics,
            "violations": violations,
            "tiny_gripper_zero_motion": args.tiny_gripper_zero_motion,
            "maximum_zero_motion_gripper_deviation": (
                MAX_ZERO_MOTION_GRIPPER_DEVIATION
                if args.tiny_gripper_zero_motion
                else None
            ),
        }
        if violations:
            report["status"] = "REJECTED_BEFORE_MOTION"
            exit_code = 2
        else:
            outbound_targets = [row[:7].copy() for row in targets]
            out_ids, out_acks, out_duration = send_timed(
                action_client,
                outbound_targets,
                f"model-h{args.horizon}",
                (
                    [float(row[7]) for row in targets]
                    if args.tiny_gripper_zero_motion
                    else None
                ),
            )
            report["model_waypoints_sent"] = args.horizon
            (
                outbound_status,
                out_results,
                outbound_settle_s,
                outbound_commands_complete,
                outbound_pass,
            ) = wait_for_completion(action_client, out_ids)
            report["outbound"] = {
                "ids": out_ids,
                "acknowledgements": out_acks,
                "stream_duration_s": out_duration,
                "settle_wait_s": outbound_settle_s,
                "results": out_results,
                "commands_completed": outbound_commands_complete,
                "settled_position_error_m": outbound_status.get(
                    "live_target_position_error_m"
                ),
                "settled_rotation_error_rad": outbound_status.get(
                    "live_target_rotation_error_rad"
                ),
                "passed": outbound_pass,
            }
            safe_to_reverse = bool(
                outbound_commands_complete
                and outbound_status.get("ready")
                and outbound_status.get("fatal_error") is None
                and outbound_status.get("queue_depth") == 0
            )
            if not safe_to_reverse:
                report["status"] = (
                    f"MODEL_H{args.horizon}_FAILED_UNSAFE_TO_AUTOMATICALLY_RETURN"
                )
            else:
                reverse_targets = [
                    row[:7].copy() for row in targets[-2::-1]
                ] + [origin.copy()]
                return_ids, return_acks, return_duration = send_timed(
                    action_client,
                    reverse_targets,
                    f"model-h{args.horizon}-return",
                    (
                        [GRIPPER_OPEN_COMMAND] * args.horizon
                        if args.tiny_gripper_zero_motion
                        else None
                    ),
                )
                report["return_waypoints_sent"] = args.horizon
                (
                    final,
                    return_results,
                    return_settle_s,
                    return_commands_complete,
                    return_pass,
                ) = wait_for_completion(action_client, return_ids)
                report["return"] = {
                    "ids": return_ids,
                    "acknowledgements": return_acks,
                    "stream_duration_s": return_duration,
                    "settle_wait_s": return_settle_s,
                    "results": return_results,
                    "commands_completed": return_commands_complete,
                    "settled_position_error_m": final.get(
                        "live_target_position_error_m"
                    ),
                    "settled_rotation_error_rad": final.get(
                        "live_target_rotation_error_rad"
                    ),
                    "passed": return_pass,
                }
                if args.tiny_gripper_zero_motion:
                    gripper_status = final.get("right_gripper") or {}
                    gripper_command_count = int(gripper_status.get("command_count", -1))
                    report["gripper_commands_sent"] = gripper_command_count
                    report["zero_gripper_motion_gate"] = {
                        "command_count": gripper_command_count,
                        "passed": gripper_command_count == 0,
                        "status": gripper_status,
                    }
                    return_pass = return_pass and gripper_command_count == 0
                    report["return"]["passed"] = return_pass
                if return_pass and outbound_pass:
                    report["status"] = f"PASS_MODEL_H{args.horizon}_RETURNED"
                    exit_code = 0
                elif return_pass:
                    report["status"] = "RETURNED_WITH_OUTBOUND_SETTLE_WARNING"
                    exit_code = 1
                else:
                    report["status"] = "RETURN_FAILED"
                    exit_code = 1

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    _ACTIVE_REPORT = None
    _ACTIVE_REPORT_PATH = None
    return exit_code


if __name__ == "__main__":
    try:
        result = main()
    except Exception as error:
        if _ACTIVE_REPORT is not None and _ACTIVE_REPORT_PATH is not None:
            _ACTIVE_REPORT["status"] = (
                "ERROR_BEFORE_MOTION"
                if int(_ACTIVE_REPORT.get("model_waypoints_sent", 0)) == 0
                else "ERROR_AFTER_MOTION_REQUIRES_OPERATOR_INSPECTION"
            )
            _ACTIVE_REPORT["exception"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            _ACTIVE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            _ACTIVE_REPORT_PATH.write_text(
                json.dumps(_ACTIVE_REPORT, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(_ACTIVE_REPORT, indent=2))
        else:
            print(json.dumps({"status": "ERROR", "message": str(error)}))
        result = 1
    raise SystemExit(result)
