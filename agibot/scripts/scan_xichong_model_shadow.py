#!/usr/bin/env python3
"""Scan recorded held-out observations through a remote model and shadow gate."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

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
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "agibot/gr00t_data/xichong_right_single_grasp_eval_heldout12",
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


def observation(trajectory, step: int) -> dict:
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


def main() -> None:
    args = parse_args()
    limits = load_shadow_safety_limits(args.safety_config)
    errors: list[str] = []
    violations: list[dict] = []
    gripper_gates: list[dict] = []
    episode_reports: list[dict] = []
    latencies: list[float] = []
    max_translation = 0.0
    max_rotation = 0.0
    calls = 0

    with PolicyClient(host=args.host, port=args.port, timeout_ms=60000) as client:
        ping = client.ping()
        modality = client.get_modality_config()
        loader = LeRobotEpisodeLoader(args.dataset.resolve(), modality)
        for episode_id in range(len(loader)):
            trajectory = loader[episode_id]
            episode_calls = 0
            episode_rejected = 0
            episode_gates = 0
            for step in range(0, len(trajectory), args.execution_horizon):
                started = time.perf_counter()
                action, _ = client.get_action(observation(trajectory, step))
                latencies.append(time.perf_counter() - started)
                eef = np.asarray(action.get("right_eef"), dtype=np.float32)
                gripper = np.asarray(action.get("right_gripper"), dtype=np.float32)
                if eef.shape != (1, 16, 9) or gripper.shape != (1, 16, 1):
                    errors.append(
                        f"episode={episode_id} step={step} shapes={eef.shape}/{gripper.shape}"
                    )
                    continue
                if not np.isfinite(eef).all() or not np.isfinite(gripper).all():
                    errors.append(f"episode={episode_id} step={step} non-finite action")
                    continue
                decoded = decode_action_chunk(action)
                state_eef = np.asarray(
                    trajectory["state.right_eef"].iloc[step], dtype=np.float32
                )
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
                max_translation = max(max_translation, decision.max_translation_step_m)
                max_rotation = max(max_rotation, decision.max_rotation_step_deg)
                if not decision.chunk_allowed:
                    episode_rejected += 1
                    violations.append(
                        {
                            "episode": episode_id,
                            "step": step,
                            "violations": list(decision.violations),
                        }
                    )
                if decision.gripper_gate_events:
                    episode_gates += 1
                    gripper_gates.append(
                        {
                            "episode": episode_id,
                            "step": step,
                            "events": list(decision.gripper_gate_events),
                        }
                    )
                calls += 1
                episode_calls += 1
            episode_reports.append(
                {
                    "episode": episode_id,
                    "frames": len(trajectory),
                    "calls": episode_calls,
                    "rejected_chunks": episode_rejected,
                    "gripper_gate_chunks": episode_gates,
                }
            )
        if args.kill_server:
            client.kill_server()

    latency = np.asarray(latencies, dtype=np.float64)
    warm = latency[1:] if len(latency) > 1 else latency
    status = "PASS" if ping and not errors else "FAIL"
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "scope": "all held-out H8 observations through remote model and non-actuating safety",
        "dataset": str(args.dataset.resolve()),
        "host": args.host,
        "port": args.port,
        "ping": ping,
        "episodes": len(episode_reports),
        "calls": calls,
        "execution_horizon": args.execution_horizon,
        "model_action_horizon": 16,
        "latency_seconds": {
            "cold_first": float(latency[0]),
            "warm_mean": float(warm.mean()),
            "warm_p90": float(np.percentile(warm, 90)),
            "warm_max": float(warm.max()),
        },
        "safety_config": str(args.safety_config.resolve()),
        "approval_status": limits.approval_status,
        "live_execution_enabled": limits.live_execution_enabled,
        "max_translation_step_m": max_translation,
        "max_rotation_step_deg": max_rotation,
        "rejected_chunks": len(violations),
        "gripper_gate_chunks": len(gripper_gates),
        "violation_details": violations,
        "gripper_gate_details": gripper_gates,
        "episode_reports": episode_reports,
        "errors": errors,
        "motor_commands_sent": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
