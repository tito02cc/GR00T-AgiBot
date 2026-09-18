#!/usr/bin/env python3
"""Measure the position coverage that current Xichong evaluations can support."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-dataset",
        type=Path,
        default=REPO_ROOT / "agibot/gr00t_data/xichong_right_single_grasp",
    )
    parser.add_argument(
        "--training-dataset",
        type=Path,
        default=REPO_ROOT / "agibot/gr00t_data/xichong_right_single_grasp_300",
    )
    parser.add_argument(
        "--heldout-dataset",
        type=Path,
        default=REPO_ROOT
        / "agibot/gr00t_data/xichong_right_single_grasp_eval_heldout12",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "agibot/reports/xichong_generalization_coverage.json",
    )
    parser.add_argument("--close-threshold", type=float, default=-0.55)
    return parser.parse_args()


def xyz_stats(values: np.ndarray) -> dict[str, object]:
    q01 = np.quantile(values, 0.01, axis=0)
    q99 = np.quantile(values, 0.99, axis=0)
    return {
        "min_xyz_m": values.min(axis=0).tolist(),
        "q01_xyz_m": q01.tolist(),
        "median_xyz_m": np.quantile(values, 0.5, axis=0).tolist(),
        "q99_xyz_m": q99.tolist(),
        "max_xyz_m": values.max(axis=0).tolist(),
        "q01_q99_span_xyz_m": (q99 - q01).tolist(),
        "standard_deviation_xyz_m": values.std(axis=0).tolist(),
    }


def scan_dataset(root: Path, close_threshold: float) -> dict[str, object]:
    paths = sorted(root.resolve().glob("data/chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Parquet episodes under {root}")
    starts: list[np.ndarray] = []
    closures: list[np.ndarray] = []
    missing_closure: list[str] = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["action", "observation.state"])
        action = np.stack(frame["action"].to_numpy()).astype(np.float64)
        state = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
        gripper = action[:, 9]
        onset = np.flatnonzero(
            (gripper >= close_threshold)
            & np.concatenate(([True], gripper[:-1] < close_threshold))
        )
        starts.append(state[0, :3])
        if len(onset):
            closures.append(action[int(onset[0]), :3])
        else:
            missing_closure.append(path.name)
    return {
        "dataset": str(root.resolve()),
        "episodes": len(paths),
        "episodes_with_closure": len(closures),
        "episodes_missing_closure": missing_closure,
        "start_eef_xyz": xyz_stats(np.stack(starts)),
        "first_closure_xyz": xyz_stats(np.stack(closures)),
    }


def main() -> None:
    args = parse_args()
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "IN_DISTRIBUTION_GENERALIZATION_ONLY",
        "coordinate_frame": "base_link_tf",
        "close_threshold_training_units": args.close_threshold,
        "datasets": {
            "full_636": scan_dataset(args.full_dataset, args.close_threshold),
            "training_300": scan_dataset(args.training_dataset, args.close_threshold),
            "heldout_12": scan_dataset(args.heldout_dataset, args.close_threshold),
        },
        "conclusions": [
            "the policy is image-and-state conditioned and is not a fixed waypoint program",
            "the recorded initial EEF pose is effectively fixed, so base/start-pose variation has not been validated",
            "the held-out set samples the same collection distribution and does not prove out-of-distribution robot placement generalization",
            "the absolute XYZ closure envelope is only a shadow distribution monitor and must not be deployed as task logic",
            "live grasp gating must be relative to an object pose estimated from current perception or contact, while physical workspace limits remain an independent safety layer",
        ],
        "required_before_claiming_position_generalization": [
            "define the intended robot-to-workcell translation and yaw envelope",
            "collect or reserve successful demonstrations across that envelope, including boundary cases",
            "create position-stratified held-out sets that are outside the training bins",
            "evaluate success by position bin on the physical robot in non-actuating shadow mode first",
            "do not use synthetic image shifts unless camera geometry, robot state, and action labels are transformed consistently",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
