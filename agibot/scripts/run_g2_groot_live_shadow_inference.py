#!/usr/bin/env python3
"""Run the real GR00T model on live G2 observations without robot commands."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agibot.tools.g2_gr00t_shadow_adapter import (  # noqa: E402
    build_policy_observation,
    decode_action_chunk,
)
from agibot.tools.g2_groot_live_observation_client import (  # noqa: E402
    G2LiveObservationClient,
)
from agibot.tools.g2_shadow_safety import (  # noqa: E402
    evaluate_shadow_chunk,
    load_shadow_safety_limits,
)
from gr00t.policy.server_client import PolicyClient  # noqa: E402


DEFAULT_PROMPT = (
    "starting with the right gripper open and the left arm and left gripper stationary, "
    "grasp the metal_workpiece with the right gripper, lift it at least 3 cm clear of "
    "the pickup position, and hold the right arm stable"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=19100)
    parser.add_argument("--model-host", default="127.0.0.1")
    parser.add_argument("--model-port", type=int, required=True)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--safety-config",
        type=Path,
        default=REPO_ROOT / "agibot/configs/xichong_shadow_safety.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "agibot/reports/g2_groot_live_shadow_inference_h8.json",
    )
    parser.add_argument("--kill-model-server", action="store_true")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def latency_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    warm = array[1:] if len(array) > 1 else array
    return {
        "cold_first": float(array[0]),
        "warm_mean": float(warm.mean()),
        "warm_p90": float(np.percentile(warm, 90)),
        "warm_max": float(warm.max()),
    }


def main() -> None:
    args = parse_args()
    if args.calls <= 0:
        raise ValueError("calls must be positive")
    if not 1 <= args.execution_horizon <= 16:
        raise ValueError("execution horizon must be in 1..16")
    limits = load_shadow_safety_limits(args.safety_config)
    errors: list[str] = []
    calls: list[dict[str, Any]] = []

    with (
        G2LiveObservationClient(args.bridge_host, args.bridge_port) as observation_client,
        PolicyClient(host=args.model_host, port=args.model_port, timeout_ms=60000) as client,
    ):
        bridge = observation_client.get_info()
        ping = client.ping()
        modality = client.get_modality_config()
        modality_keys = {name: config.modality_keys for name, config in modality.items()}
        expected_keys = {
            "video": ["head_color", "hand_right"],
            "state": ["right_eef", "right_gripper"],
            "action": ["right_eef", "right_gripper"],
            "language": ["annotation.human.task_description"],
        }
        if modality_keys != expected_keys:
            errors.append(f"modality mismatch: {modality_keys}")
        action_horizon = len(modality["action"].delta_indices)
        if action_horizon != 16:
            errors.append(f"action horizon={action_horizon}, expected=16")

        for index in range(args.calls):
            cycle_started = time.perf_counter()
            snapshot_started = time.perf_counter()
            snapshot = observation_client.get_snapshot()
            snapshot_latency = time.perf_counter() - snapshot_started
            metadata = snapshot.metadata
            pose = np.asarray(
                metadata["right_eef_xyz_quaternion_xyzw"], dtype=np.float64
            )
            gripper = float(metadata["right_gripper"]["training_position"])
            preprocess_started = time.perf_counter()
            observation = build_policy_observation(
                snapshot.head_color_rgb,
                snapshot.hand_right_rgb,
                pose,
                gripper,
                args.prompt,
            )
            preprocess_latency = time.perf_counter() - preprocess_started
            started = time.perf_counter()
            action, server_info = client.get_action(observation)
            model_latency = time.perf_counter() - started
            end_to_end_latency = time.perf_counter() - cycle_started
            eef = np.asarray(action.get("right_eef"), dtype=np.float32)
            gripper_action = np.asarray(action.get("right_gripper"), dtype=np.float32)
            if eef.shape != (1, 16, 9):
                errors.append(f"call {index}: right_eef shape={eef.shape}")
            if gripper_action.shape != (1, 16, 1):
                errors.append(f"call {index}: right_gripper shape={gripper_action.shape}")
            if not np.isfinite(eef).all() or not np.isfinite(gripper_action).all():
                errors.append(f"call {index}: non-finite action")

            decoded = decode_action_chunk(action)
            decision = evaluate_shadow_chunk(
                pose,
                gripper,
                decoded,
                limits,
                execution_horizon=args.execution_horizon,
            )
            calls.append(
                {
                    "call": index,
                    "snapshot_index": int(metadata["snapshot_index"]),
                    "camera_skew_ms": float(metadata["camera_skew_ms"]),
                    "maximum_state_camera_skew_ms": float(
                        metadata["maximum_state_camera_skew_ms"]
                    ),
                    "capture_duration_ms": float(metadata["capture_duration_ms"]),
                    "observation_transport_latency_seconds": snapshot_latency,
                    "observation_preprocess_latency_seconds": preprocess_latency,
                    "source_payload_sha256": snapshot.source_payload_sha256,
                    "current_pose_xyz_quaternion_xyzw": pose.tolist(),
                    "current_gripper_training": gripper,
                    "model_latency_seconds": model_latency,
                    "end_to_end_latency_seconds": end_to_end_latency,
                    "server_info": json_safe(server_info),
                    "action_shapes": {
                        "right_eef": list(eef.shape),
                        "right_gripper": list(gripper_action.shape),
                    },
                    "decoded_first_h8_targets": decoded[: args.execution_horizon].tolist(),
                    "distribution_monitor": {
                        "chunk_allowed": decision.chunk_allowed,
                        "max_translation_step_m": decision.max_translation_step_m,
                        "max_rotation_step_deg": decision.max_rotation_step_deg,
                        "violations": list(decision.violations),
                        "fixed_xyz_gripper_events": list(decision.gripper_gate_events),
                        "note": (
                            "fixed XYZ closure bounds are a training-distribution monitor only, "
                            "not live task logic"
                        ),
                    },
                }
            )
        if args.kill_model_server:
            client.kill_server()

    model_latencies = [item["model_latency_seconds"] for item in calls]
    observation_latencies = [
        item["observation_transport_latency_seconds"] for item in calls
    ]
    preprocess_latencies = [
        item["observation_preprocess_latency_seconds"] for item in calls
    ]
    end_to_end_latencies = [item["end_to_end_latency_seconds"] for item in calls]
    cycle_budget_s = args.execution_horizon / 10.0
    warm_end_to_end = (
        end_to_end_latencies[1:]
        if len(end_to_end_latencies) > 1
        else end_to_end_latencies
    )
    h8_timing_candidate = max(warm_end_to_end) <= cycle_budget_s
    rejected = sum(not item["distribution_monitor"]["chunk_allowed"] for item in calls)
    gripper_events = sum(
        bool(item["distribution_monitor"]["fixed_xyz_gripper_events"])
        for item in calls
    )
    status = "PASS_LIVE_SHADOW_MODEL" if ping and not errors and rejected == 0 else "FAIL"
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "scope": "live G2 sensors through real GR00T model, observation only",
        "checkpoint": (
            "agibot/models/xichong_rgrasp_n1d7_checkpoint-30000/model"
        ),
        "execution_horizon": args.execution_horizon,
        "model_action_horizon": action_horizon,
        "prompt": args.prompt,
        "bridge": bridge,
        "model_server": {
            "host": args.model_host,
            "port": args.model_port,
            "ping": ping,
            "modality_keys": modality_keys,
        },
        "latency_seconds": {
            "observation_transport": latency_summary(observation_latencies),
            "observation_preprocess": latency_summary(preprocess_latencies),
            "model": latency_summary(model_latencies),
            "end_to_end": latency_summary(end_to_end_latencies),
        },
        "replanning_timing_assessment": {
            "cycle_budget_seconds": cycle_budget_s,
            "warm_end_to_end_max_within_budget": h8_timing_candidate,
            "status": (
                "PASS_H8_TIMING_CANDIDATE"
                if h8_timing_candidate
                else "FAIL_H8_TIMING_CANDIDATE"
            ),
            "note": (
                "This is a non-actuating shadow timing measurement. The future 10 Hz "
                "low-level action loop is separate from the model replanning cycle."
            ),
        },
        "prediction_summary": {
            "maximum_translation_step_m": float(
                max(item["distribution_monitor"]["max_translation_step_m"] for item in calls)
            ),
            "maximum_rotation_step_deg": float(
                max(item["distribution_monitor"]["max_rotation_step_deg"] for item in calls)
            ),
            "predicted_gripper_min": float(
                min(target[7] for item in calls for target in item["decoded_first_h8_targets"])
            ),
            "predicted_gripper_max": float(
                max(target[7] for item in calls for target in item["decoded_first_h8_targets"])
            ),
        },
        "calls": calls,
        "rejected_chunks": rejected,
        "fixed_xyz_gripper_monitor_events": gripper_events,
        "errors": errors,
        "live_execution_enabled": False,
        "motor_commands_sent": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS_LIVE_SHADOW_MODEL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
