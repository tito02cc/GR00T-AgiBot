#!/usr/bin/env python3
"""Derive a shadow-only anomaly envelope from the selected training set."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agibot.tools.g2_gr00t_shadow_adapter import rot6d_to_quaternion_xyzw  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "agibot/gr00t_data/xichong_right_single_grasp_300",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "agibot/configs/xichong_shadow_safety.json",
    )
    parser.add_argument("--workspace-margin-m", type=float, default=0.01)
    parser.add_argument("--closure-margin-m", type=float, default=0.01)
    parser.add_argument("--motion-max-margin", type=float, default=1.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.dataset.resolve().glob("data/chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet episodes under {args.dataset}")

    all_xyz: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    closure_xyz: list[np.ndarray] = []
    frames = 0
    for path in paths:
        frame = pd.read_parquet(path, columns=["action", "observation.state"])
        action = np.stack(frame["action"].to_numpy()).astype(np.float64)
        state = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
        frames += len(action)
        all_xyz.extend((state[:, :3], action[:, :3]))
        translations.append(np.linalg.norm(np.diff(action[:, :3], axis=0), axis=1))
        quaternions = np.stack(
            [rot6d_to_quaternion_xyzw(row) for row in action[:, 3:9]]
        )
        relative = Rotation.from_quat(quaternions[:-1]).inv() * Rotation.from_quat(
            quaternions[1:]
        )
        rotations.append(np.degrees(relative.magnitude()))
        gripper = action[:, 9]
        onsets = np.flatnonzero(
            (gripper >= -0.55) & np.concatenate(([True], gripper[:-1] < -0.55))
        )
        if len(onsets):
            closure_xyz.append(action[int(onsets[0]), :3])

    xyz = np.concatenate(all_xyz)
    step_m = np.concatenate(translations)
    step_deg = np.concatenate(rotations)
    closures = np.stack(closure_xyz)
    workspace_min = xyz.min(axis=0) - args.workspace_margin_m
    workspace_max = xyz.max(axis=0) + args.workspace_margin_m
    closure_min = np.quantile(closures, 0.01, axis=0) - args.closure_margin_m
    closure_max = np.quantile(closures, 0.99, axis=0) + args.closure_margin_m

    report = {
        "schema": "xichong_g2_shadow_safety_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "approval_status": "SHADOW_ONLY_DATA_DERIVED_NOT_LIVE_APPROVED",
        "live_execution_enabled": False,
        "dataset": str(args.dataset.resolve()),
        "episodes": len(paths),
        "frames": frames,
        "sample_period_s": 0.1,
        "workspace": {
            "min_xyz_m": workspace_min.tolist(),
            "max_xyz_m": workspace_max.tolist(),
            "observed_min_xyz_m": xyz.min(axis=0).tolist(),
            "observed_max_xyz_m": xyz.max(axis=0).tolist(),
            "margin_m": args.workspace_margin_m,
        },
        "motion": {
            "max_translation_step_m": float(step_m.max() * args.motion_max_margin),
            "max_rotation_step_deg": float(step_deg.max() * args.motion_max_margin),
            "observed_translation_step_m": {
                "q99": float(np.quantile(step_m, 0.99)),
                "q999": float(np.quantile(step_m, 0.999)),
                "max": float(step_m.max()),
            },
            "observed_rotation_step_deg": {
                "q99": float(np.quantile(step_deg, 0.99)),
                "q999": float(np.quantile(step_deg, 0.999)),
                "max": float(step_deg.max()),
            },
            "margin_multiplier": args.motion_max_margin,
        },
        "gripper_gate": {
            "close_threshold_training_units": -0.55,
            "closure_min_xyz_m": closure_min.tolist(),
            "closure_max_xyz_m": closure_max.tolist(),
            "closure_q01_xyz_m": np.quantile(closures, 0.01, axis=0).tolist(),
            "closure_q99_xyz_m": np.quantile(closures, 0.99, axis=0).tolist(),
            "closure_samples": len(closures),
            "margin_m": args.closure_margin_m,
        },
        "deployment_semantics": {
            "fixed_waypoint_execution_prohibited": True,
            "workspace": "physical safety boundary, not a task waypoint",
            "motion": "controller safety boundary, not a task trajectory",
            "gripper_gate": "shadow distribution monitor only; absolute XYZ is not approved live task logic",
            "required_live_grasp_reference": "object-relative pose from current perception or contact",
        },
        "required_before_live_use": [
            "replace data-derived workspace with robot-cell approved limits",
            "replace motion anomaly limits with G2 controller approved velocity/acceleration limits",
            "replace absolute closure envelope with a perception-derived object-relative grasp gate",
            "validate collision geometry, E-stop, watchdog, and command timeout",
            "operator approval for motor enable",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
