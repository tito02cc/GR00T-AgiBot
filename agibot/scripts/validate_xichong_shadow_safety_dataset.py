#!/usr/bin/env python3
"""Replay all selected demonstrations through the shadow safety decision layer."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "agibot/gr00t_data/xichong_right_single_grasp_300",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "agibot/configs/xichong_shadow_safety.json",
    )
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "agibot/reports/xichong_shadow_safety_dataset_gate_h8.json",
    )
    return parser.parse_args()


def padded(values: np.ndarray, start: int, horizon: int) -> np.ndarray:
    chunk = values[start : start + horizon]
    if len(chunk) < horizon:
        chunk = np.concatenate((chunk, np.tile(values[-1:], (horizon - len(chunk), 1))))
    return chunk


def main() -> None:
    args = parse_args()
    limits = load_shadow_safety_limits(args.config)
    paths = sorted(args.dataset.resolve().glob("data/chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet episodes under {args.dataset}")
    action_horizon = 16
    rejected: list[dict] = []
    gates: list[dict] = []
    chunks = 0
    executed_steps = 0
    max_translation = 0.0
    max_rotation = 0.0

    for episode_id, path in enumerate(paths):
        frame = pd.read_parquet(path, columns=["action", "observation.state"])
        state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        action_all = np.stack(frame["action"].to_numpy()).astype(np.float32)
        state_eef = state[:, :9]
        state_gripper = state[:, 9:10]
        action_eef = action_all[:, :9]
        action_gripper = action_all[:, 9:10]
        for step in range(0, len(frame), args.execution_horizon):
            action = {
                "right_eef": padded(action_eef, step, action_horizon)[None],
                "right_gripper": padded(action_gripper, step, action_horizon)[None],
            }
            decoded = decode_action_chunk(action)
            current_pose = np.concatenate(
                (state_eef[step, :3], rot6d_to_quaternion_xyzw(state_eef[step, 3:9]))
            )
            decision = evaluate_shadow_chunk(
                current_pose,
                float(state_gripper[step, 0]),
                decoded,
                limits,
                execution_horizon=args.execution_horizon,
            )
            max_translation = max(max_translation, decision.max_translation_step_m)
            max_rotation = max(max_rotation, decision.max_rotation_step_deg)
            if not decision.chunk_allowed:
                rejected.append(
                    {
                        "episode": episode_id,
                        "step": step,
                        "violations": list(decision.violations),
                    }
                )
            if decision.gripper_gate_events:
                gates.append(
                    {
                        "episode": episode_id,
                        "step": step,
                        "events": list(decision.gripper_gate_events),
                    }
                )
            chunks += 1
            executed_steps += min(args.execution_horizon, len(frame) - step)

    status = "PASS" if not rejected and not gates else "FAIL"
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "scope": "all selected demonstrations through non-actuating H8 shadow safety",
        "dataset": str(args.dataset.resolve()),
        "config": str(args.config.resolve()),
        "approval_status": limits.approval_status,
        "live_execution_enabled": limits.live_execution_enabled,
        "episodes": len(paths),
        "chunks": chunks,
        "recorded_action_steps": executed_steps,
        "execution_horizon": args.execution_horizon,
        "max_translation_step_m": max_translation,
        "max_rotation_step_deg": max_rotation,
        "rejected_chunks": rejected,
        "gripper_gate_chunks": gates,
        "motor_commands_sent": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
