#!/usr/bin/env python3
"""Execute exactly one guarded GR00T H1 on the G2 right arm, then return.

The robot-side action bridge must already be explicitly armed.  This client
does not expose gripper control and rejects any prediction that is stale,
discontinuous, misaligned with the live bridge pose, or requests closure.
"""

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
from agibot.tools.g2_groot_live_observation_client import (  # noqa: E402
    G2LiveObservationClient,
)
from gr00t.policy.server_client import PolicyClient  # noqa: E402


CONFIRMATION = "EXECUTE_G2_GROOT_MODEL_SINGLE_H1_RIGHT_ARM_ONLY"
ACTION_SCHEMA = "g2_groot_persistent_h1_action_bridge_v2"
DEFAULT_PROMPT = (
    "starting with the right gripper open and the left arm and left gripper stationary, "
    "grasp the metal_workpiece with the right gripper, lift it at least 3 cm clear of "
    "the pickup position, and hold the right arm stable"
)
MAX_TRANSLATION_M = 0.002
MAX_ROTATION_DEG = 0.5
MAX_OBSERVATION_TO_COMMAND_S = 1.0
MAX_OBSERVATION_BRIDGE_TRANSLATION_M = 0.00025
MAX_OBSERVATION_BRIDGE_ROTATION_DEG = 0.1
OPEN_GRIPPER_MAX = -0.70
MAX_GRIPPER_CHANGE = 0.10
SETTLE_POSITION_M = 0.0002
SETTLE_ROTATION_RAD = 0.002


def rotation_deg(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(first[3:7], second[3:7])))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))))


def result_for(status: dict, command_id: str) -> dict | None:
    for item in status.get("recent_results", []):
        if item.get("command_id") == command_id:
            return item
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--observation-host", default="127.0.0.1")
    parser.add_argument("--observation-port", type=int, default=19100)
    parser.add_argument("--action-host", default="127.0.0.1")
    parser.add_argument("--action-port", type=int, default=19200)
    parser.add_argument("--model-host", default="127.0.0.1")
    parser.add_argument("--model-port", type=int, default=5564)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "agibot/reports/g2_groot_live_model_single_h1_20260826.json",
    )
    args = parser.parse_args()
    if not args.execute or args.confirm != CONFIRMATION:
        parser.error(f"physical execution requires --execute --confirm {CONFIRMATION}")
    return args


def main() -> int:
    args = parse_args()
    report: dict = {
        "schema": "g2_groot_live_model_single_h1_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "STARTED",
        "scope": "one model-predicted right-arm EEF H1, no gripper, then deterministic return",
        "motor_commands_sent": 0,
        "gripper_commands_sent": 0,
    }
    exit_code = 1
    with (
        G2LiveObservationClient(
            args.observation_host, args.observation_port
        ) as observation_client,
        PolicyClient(
            host=args.model_host, port=args.model_port, timeout_ms=60000
        ) as model_client,
        BridgeSession(args.action_host, args.action_port) as action_client,
    ):
        observation_info = observation_client.get_info()
        action_info = action_client.request({"op": "info"})
        action_before = action_client.request({"op": "status"})
        if observation_info.get("control_api_exposed") is not False:
            raise RuntimeError("observation bridge is not read-only")
        if action_info.get("schema") != ACTION_SCHEMA:
            raise RuntimeError("unexpected action bridge schema")
        if action_info.get("right_gripper_enabled") is not False:
            raise RuntimeError("action bridge unexpectedly enables gripper")
        if not action_before.get("ready") or action_before.get("fatal_error"):
            raise RuntimeError("action bridge is not ready")
        if not model_client.ping():
            raise RuntimeError("model server ping failed")

        origin = np.asarray(action_before["desired_pose"], dtype=np.float64)
        snapshot_started = time.monotonic()
        snapshot = observation_client.get_snapshot()
        pose = np.asarray(
            snapshot.metadata["right_eef_xyz_quaternion_xyzw"], dtype=np.float64
        )
        gripper = float(snapshot.metadata["right_gripper"]["training_position"])
        observation = build_policy_observation(
            snapshot.head_color_rgb,
            snapshot.hand_right_rgb,
            pose,
            gripper,
            args.prompt,
        )
        inference_started = time.monotonic()
        action, _ = model_client.get_action(observation)
        model_latency_s = time.monotonic() - inference_started
        decoded = decode_action_chunk(action)
        target = decoded[0].astype(np.float64)
        elapsed_s = time.monotonic() - snapshot_started
        predicted = check_shadow_target(pose, target)

        action_live = np.asarray(action_before["live_pose"], dtype=np.float64)
        observation_bridge_translation_m = float(np.linalg.norm(pose[:3] - action_live[:3]))
        observation_bridge_rotation_deg = rotation_deg(pose, action_live)
        origin_target_translation_m = float(np.linalg.norm(target[:3] - origin[:3]))
        origin_target_rotation_deg = rotation_deg(origin, target[:7])
        violations = []
        if elapsed_s > MAX_OBSERVATION_TO_COMMAND_S:
            violations.append("observation_to_command_timeout")
        if predicted.translation_step_m > MAX_TRANSLATION_M:
            violations.append("translation_step")
        if predicted.rotation_step_deg > MAX_ROTATION_DEG:
            violations.append("rotation_step")
        if origin_target_translation_m > MAX_TRANSLATION_M:
            violations.append("bridge_translation_step")
        if origin_target_rotation_deg > MAX_ROTATION_DEG:
            violations.append("bridge_rotation_step")
        if observation_bridge_translation_m > MAX_OBSERVATION_BRIDGE_TRANSLATION_M:
            violations.append("observation_bridge_translation_mismatch")
        if observation_bridge_rotation_deg > MAX_OBSERVATION_BRIDGE_ROTATION_DEG:
            violations.append("observation_bridge_rotation_mismatch")
        if target[7] > OPEN_GRIPPER_MAX or abs(float(target[7]) - gripper) > MAX_GRIPPER_CHANGE:
            violations.append("gripper_not_open")

        report["preflight"] = {
            "snapshot_index": int(snapshot.metadata["snapshot_index"]),
            "camera_skew_ms": float(snapshot.metadata["camera_skew_ms"]),
            "maximum_state_camera_skew_ms": float(
                snapshot.metadata["maximum_state_camera_skew_ms"]
            ),
            "source_payload_sha256": snapshot.source_payload_sha256,
            "current_pose": pose.tolist(),
            "action_bridge_origin": origin.tolist(),
            "target_pose": target[:7].tolist(),
            "predicted_gripper_ignored": float(target[7]),
            "current_gripper": gripper,
            "model_latency_s": model_latency_s,
            "observation_to_command_s": elapsed_s,
            "translation_step_m": predicted.translation_step_m,
            "rotation_step_deg": predicted.rotation_step_deg,
            "observation_bridge_translation_mismatch_m": observation_bridge_translation_m,
            "observation_bridge_rotation_mismatch_deg": observation_bridge_rotation_deg,
            "origin_target_translation_m": origin_target_translation_m,
            "origin_target_rotation_deg": origin_target_rotation_deg,
            "violations": violations,
        }
        if violations:
            report["status"] = "REJECTED_BEFORE_MOTION"
            exit_code = 2
        else:
            command_id = f"model-single-h1-{uuid.uuid4()}"
            acknowledgement = action_client.request(
                {
                    "op": "execute_h1",
                    "command_id": command_id,
                    "timestamp_ns": time.time_ns(),
                    "target_pose": target[:7].tolist(),
                }
            )
            if not acknowledgement.get("ok"):
                raise RuntimeError(f"model H1 rejected: {acknowledgement}")
            report["motor_commands_sent"] = 1
            time.sleep(0.6)
            target_status = action_client.request({"op": "status"})
            target_result = result_for(target_status, command_id)
            target_pass = bool(
                target_result
                and target_status.get("fatal_error") is None
                and target_status.get("queue_depth") == 0
                and target_status["live_target_position_error_m"] <= SETTLE_POSITION_M
                and target_status["live_target_rotation_error_rad"] <= SETTLE_ROTATION_RAD
            )
            report["model_h1"] = {
                "command_id": command_id,
                "acknowledgement": acknowledgement,
                "execution_result": target_result,
                "settled_position_error_m": target_status.get(
                    "live_target_position_error_m"
                ),
                "settled_rotation_error_rad": target_status.get(
                    "live_target_rotation_error_rad"
                ),
                "passed": target_pass,
            }
            if not target_pass:
                report["status"] = "MODEL_H1_FAILED_NO_RETURN"
            else:
                return_id = f"model-single-h1-return-{uuid.uuid4()}"
                return_ack = action_client.request(
                    {
                        "op": "execute_h1",
                        "command_id": return_id,
                        "timestamp_ns": time.time_ns(),
                        "target_pose": origin.tolist(),
                    }
                )
                if not return_ack.get("ok"):
                    raise RuntimeError(f"return H1 rejected: {return_ack}")
                report["motor_commands_sent"] = 2
                time.sleep(0.6)
                final_status = action_client.request({"op": "status"})
                return_result = result_for(final_status, return_id)
                return_pass = bool(
                    return_result
                    and final_status.get("fatal_error") is None
                    and final_status.get("queue_depth") == 0
                    and final_status["live_target_position_error_m"] <= SETTLE_POSITION_M
                    and final_status["live_target_rotation_error_rad"] <= SETTLE_ROTATION_RAD
                )
                report["return_h1"] = {
                    "command_id": return_id,
                    "acknowledgement": return_ack,
                    "execution_result": return_result,
                    "settled_position_error_m": final_status.get(
                        "live_target_position_error_m"
                    ),
                    "settled_rotation_error_rad": final_status.get(
                        "live_target_rotation_error_rad"
                    ),
                    "passed": return_pass,
                }
                report["status"] = (
                    "PASS_MODEL_SINGLE_H1_RETURNED" if return_pass else "RETURN_FAILED"
                )
                exit_code = 0 if return_pass else 1

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
