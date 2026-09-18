#!/usr/bin/env python3
"""Exercise a local ReplayPolicy or GR00T server without actuating a robot."""

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
    decode_action_chunk,
    rot6d_to_quaternion_xyzw,
)
from agibot.tools.g2_shadow_safety import (  # noqa: E402
    evaluate_shadow_chunk,
    load_shadow_safety_limits,
)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.policy.server_client import PolicyClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=("replay", "model"), required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "agibot/gr00t_data/xichong_right_single_grasp_300",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--calls", type=int, default=4)
    parser.add_argument(
        "--step-indices",
        type=int,
        nargs="+",
        help="Recorded observation steps to query. Model mode may use non-consecutive steps.",
    )
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument(
        "--safety-config",
        type=Path,
        default=REPO_ROOT / "agibot/configs/xichong_shadow_safety.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--kill-server", action="store_true")
    return parser.parse_args()


def extract_column(trajectory: Any, key: str) -> np.ndarray:
    return np.vstack([np.asarray(value) for value in trajectory[key]])


def make_observation(trajectory: Any, step: int) -> dict[str, Any]:
    return {
        "video": {
            "head_color": np.asarray(trajectory["video.head_color"].iloc[step])[None, None],
            "hand_right": np.asarray(trajectory["video.hand_right"].iloc[step])[None, None],
        },
        "state": {
            "right_eef": np.asarray(
                trajectory["state.right_eef"].iloc[step], dtype=np.float32
            )[None, None],
            "right_gripper": np.asarray(
                trajectory["state.right_gripper"].iloc[step], dtype=np.float32
            )[None, None],
        },
        "language": {
            "annotation.human.task_description": [[
                str(trajectory["language.annotation.human.task_description"].iloc[step])
            ]],
        },
    }


def expected_replay(values: np.ndarray, start: int, horizon: int) -> np.ndarray:
    chunk = values[start : start + horizon]
    if len(chunk) < horizon:
        chunk = np.concatenate((chunk, np.tile(values[-1:], (horizon - len(chunk), 1))))
    return chunk[None].astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.calls <= 0 or args.execution_horizon <= 0:
        raise ValueError("calls and execution horizon must be positive")
    if args.step_indices is not None and args.mode == "replay":
        expected_steps = [index * args.execution_horizon for index in range(len(args.step_indices))]
        if args.step_indices != expected_steps:
            raise ValueError(
                "Replay mode step indices must follow the server execution horizon: "
                f"expected {expected_steps}"
            )
    errors: list[str] = []
    call_reports: list[dict[str, Any]] = []
    limits = load_shadow_safety_limits(args.safety_config)

    with PolicyClient(host=args.host, port=args.port, timeout_ms=60000) as client:
        ping = client.ping()
        modality = client.get_modality_config()
        modality_keys = {
            key: value.modality_keys for key, value in modality.items()
        }
        expected_keys = {
            "video": ["head_color", "hand_right"],
            "state": ["right_eef", "right_gripper"],
            "action": ["right_eef", "right_gripper"],
            "language": ["annotation.human.task_description"],
        }
        if modality_keys != expected_keys:
            errors.append(f"modality keys mismatch: {modality_keys}")
        action_horizon = len(modality["action"].delta_indices)
        if action_horizon != 16:
            errors.append(f"model action horizon={action_horizon}, expected=16")

        loader = LeRobotEpisodeLoader(args.dataset.resolve(), modality)
        trajectory = loader[args.episode]
        eef_actions = extract_column(trajectory, "action.right_eef").astype(np.float32)
        gripper_actions = extract_column(trajectory, "action.right_gripper").astype(np.float32)
        reset_info = client.reset({"episode_index": args.episode, "step_index": 0})

        requested_steps = args.step_indices or [
            index * args.execution_horizon for index in range(args.calls)
        ]
        for call_index, requested_step in enumerate(requested_steps):
            step = min(requested_step, len(trajectory) - 1)
            observation = make_observation(trajectory, step)
            started = time.perf_counter()
            action, info = client.get_action(observation)
            elapsed = time.perf_counter() - started
            eef = np.asarray(action.get("right_eef"), dtype=np.float32)
            gripper = np.asarray(action.get("right_gripper"), dtype=np.float32)
            if eef.shape != (1, 16, 9):
                errors.append(f"call {call_index}: right_eef shape={eef.shape}")
            if gripper.shape != (1, 16, 1):
                errors.append(f"call {call_index}: right_gripper shape={gripper.shape}")
            if not np.isfinite(eef).all() or not np.isfinite(gripper).all():
                errors.append(f"call {call_index}: non-finite action")

            replay_max_abs = None
            if args.mode == "replay":
                server_step = int(info["current_step"])
                expected_eef = expected_replay(eef_actions, server_step, 16)
                expected_gripper = expected_replay(gripper_actions, server_step, 16)
                replay_max_abs = float(
                    max(
                        np.max(np.abs(eef - expected_eef)),
                        np.max(np.abs(gripper - expected_gripper)),
                    )
                )
                if replay_max_abs > 1e-7:
                    errors.append(f"call {call_index}: replay max abs={replay_max_abs}")

            decoded = decode_action_chunk(action)
            state_eef = np.asarray(trajectory["state.right_eef"].iloc[step], dtype=np.float32)
            current_pose = np.concatenate(
                (state_eef[:3], rot6d_to_quaternion_xyzw(state_eef[3:9]))
            )
            current_gripper = float(trajectory["state.right_gripper"].iloc[step][0])
            decision = evaluate_shadow_chunk(
                current_pose,
                current_gripper,
                decoded,
                limits,
                execution_horizon=args.execution_horizon,
            )
            call_reports.append(
                {
                    "call": call_index,
                    "recorded_observation_step": step,
                    "latency_seconds": elapsed,
                    "server_info": info,
                    "action_shapes": {
                        "right_eef": list(eef.shape),
                        "right_gripper": list(gripper.shape),
                    },
                    "replay_max_abs": replay_max_abs,
                    "shadow_chunk_allowed": decision.chunk_allowed,
                    "shadow_max_translation_step_m": decision.max_translation_step_m,
                    "shadow_max_rotation_step_deg": decision.max_rotation_step_deg,
                    "shadow_violations": list(decision.violations),
                    "gripper_gate_events": list(decision.gripper_gate_events),
                }
            )

        if args.kill_server:
            client.kill_server()

    latencies = np.asarray([row["latency_seconds"] for row in call_reports])
    warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "PASS" if ping and not errors else "FAIL",
        "scope": "non-actuating PolicyClient transport and shadow safety smoke",
        "mode": args.mode,
        "host": args.host,
        "port": args.port,
        "dataset": str(args.dataset.resolve()),
        "episode": args.episode,
        "recorded_observation_steps": requested_steps,
        "execution_horizon": args.execution_horizon,
        "model_action_horizon": action_horizon,
        "ping": ping,
        "modality_keys": modality_keys,
        "reset_info": reset_info,
        "latency_seconds": {
            "mean": float(latencies.mean()),
            "p90": float(np.percentile(latencies, 90)),
            "max": float(latencies.max()),
            "warm_mean": float(warm_latencies.mean()),
            "warm_p90": float(np.percentile(warm_latencies, 90)),
            "warm_max": float(warm_latencies.max()),
        },
        "safety_config": str(args.safety_config.resolve()),
        "safety_approval_status": limits.approval_status,
        "live_execution_enabled": limits.live_execution_enabled,
        "calls": call_reports,
        "errors": errors,
        "motor_commands_sent": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
