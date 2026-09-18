#!/usr/bin/env python3
"""Capture and audit one real G2 observation through the read-only bridge."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agibot.tools.g2_gr00t_shadow_adapter import (  # noqa: E402
    make_right_eef_state,
    rot6d_to_quaternion_xyzw,
)
from agibot.tools.g2_groot_live_observation_client import (  # noqa: E402
    get_bridge_info,
    get_live_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19100)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--training-dataset",
        type=Path,
        default=REPO_ROOT / "agibot/gr00t_data/xichong_right_single_grasp_300",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "agibot/live_shadow/g2_initial_observation_20260826",
    )
    parser.add_argument("--initial-translation-tolerance-m", type=float, default=0.02)
    parser.add_argument("--initial-rotation-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--initial-gripper-tolerance", type=float, default=0.10)
    return parser.parse_args()


def load_training_starts(dataset: Path) -> np.ndarray:
    starts = []
    for path in sorted(dataset.resolve().glob("data/chunk-*/*.parquet")):
        frame = pd.read_parquet(path, columns=["observation.state"])
        starts.append(np.asarray(frame["observation.state"].iloc[0], dtype=np.float64))
    if not starts:
        raise FileNotFoundError(f"no training episodes under {dataset}")
    array = np.stack(starts)
    if array.shape[1] != 10 or not np.isfinite(array).all():
        raise ValueError(f"unexpected training state shape {array.shape}")
    return array


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to save {path}")


def main() -> None:
    args = parse_args()
    if args.timeout_s <= 0:
        raise ValueError("timeout must be positive")
    bridge_info = get_bridge_info(args.host, args.port, args.timeout_s)
    snapshot = get_live_snapshot(args.host, args.port, args.timeout_s)
    metadata = snapshot.metadata
    pose = np.asarray(metadata["right_eef_xyz_quaternion_xyzw"], dtype=np.float64)
    gripper = float(metadata["right_gripper"]["training_position"])
    live_state = np.concatenate((make_right_eef_state(pose), [gripper]))
    starts = load_training_starts(args.training_dataset)

    xyz_q01 = np.quantile(starts[:, :3], 0.01, axis=0)
    xyz_q99 = np.quantile(starts[:, :3], 0.99, axis=0)
    xyz_median = np.median(starts[:, :3], axis=0)
    translation_from_median = float(np.linalg.norm(live_state[:3] - xyz_median))
    live_rotation = Rotation.from_quat(pose[3:7])
    training_quaternions = np.stack(
        [rot6d_to_quaternion_xyzw(row[3:9]) for row in starts]
    )
    rotation_errors = np.degrees(
        (Rotation.from_quat(training_quaternions).inv() * live_rotation).magnitude()
    )
    rotation_error_median = float(np.median(rotation_errors))
    training_gripper_median = float(np.median(starts[:, 9]))
    gripper_from_median = abs(gripper - training_gripper_median)

    body = metadata["right_body_health"]
    joint_errors = [
        item
        for item in metadata["right_joint_health"]["joints"]
        if int(item["error_code"]) != 0
    ]
    protocol_pass = bool(
        bridge_info["control_api_exposed"] is False
        and bridge_info["live_execution_enabled"] is False
        and int(bridge_info["motor_commands_sent"]) == 0
        and metadata["control_api_exposed"] is False
        and metadata["live_execution_enabled"] is False
        and int(metadata["motor_commands_sent"]) == 0
    )
    sensor_pass = bool(
        snapshot.head_color_rgb.shape == (480, 640, 3)
        and snapshot.hand_right_rgb.shape == (480, 640, 3)
        and snapshot.head_color_rgb.dtype == np.uint8
        and snapshot.hand_right_rgb.dtype == np.uint8
        and not joint_errors
        and int(body["right_arm_error"]) == 0
        and int(body["right_end_error"]) == 0
    )
    initial_pose_pass = bool(
        translation_from_median <= args.initial_translation_tolerance_m
        and rotation_error_median <= args.initial_rotation_tolerance_deg
        and gripper_from_median <= args.initial_gripper_tolerance
    )
    status = (
        "PASS_LIVE_OBSERVATION_INITIAL_POSE"
        if protocol_pass and sensor_pass and initial_pose_pass
        else "CAPTURED_WITH_GATE_FAILURE"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    head_path = args.output_dir / "head_color_rgb.png"
    wrist_path = args.output_dir / "hand_right_rgb.png"
    save_rgb(head_path, snapshot.head_color_rgb)
    save_rgb(wrist_path, snapshot.hand_right_rgb)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "scope": "one synchronized live G2 observation, no model and no actuation",
        "bridge": bridge_info,
        "snapshot_metadata": metadata,
        "processed_images": {
            "head_color": str(head_path.resolve()),
            "hand_right": str(wrist_path.resolve()),
            "shape": [480, 640, 3],
            "dtype": "uint8",
            "color": "RGB",
            "source_payload_sha256": snapshot.source_payload_sha256,
            "preprocessing": snapshot.preprocessing,
        },
        "live_gr00t_state": {
            "right_eef_xyz_rot6d": live_state[:9].tolist(),
            "right_gripper": gripper,
        },
        "training_initial_comparison": {
            "episodes": int(len(starts)),
            "xyz_q01_m": xyz_q01.tolist(),
            "xyz_median_m": xyz_median.tolist(),
            "xyz_q99_m": xyz_q99.tolist(),
            "live_translation_from_median_m": translation_from_median,
            "live_inside_strict_q01_q99_xyz": bool(
                np.all(live_state[:3] >= xyz_q01) and np.all(live_state[:3] <= xyz_q99)
            ),
            "live_rotation_error_to_training_median_deg": rotation_error_median,
            "training_gripper_median": training_gripper_median,
            "live_gripper_abs_error_from_median": gripper_from_median,
            "practical_tolerances": {
                "translation_m": args.initial_translation_tolerance_m,
                "rotation_deg": args.initial_rotation_tolerance_deg,
                "gripper_training_units": args.initial_gripper_tolerance,
            },
        },
        "gates": {
            "protocol_read_only": protocol_pass,
            "sensors_and_right_arm_health": sensor_pass,
            "initial_pose": initial_pose_pass,
            "camera_pair_skew_within_bridge_limit": bool(
                metadata["camera_skew_ms"] <= bridge_info["max_camera_skew_ms"]
            ),
            "state_camera_skew_within_bridge_limit": bool(
                metadata["maximum_state_camera_skew_ms"]
                <= bridge_info["max_state_camera_skew_ms"]
            ),
        },
        "live_execution_enabled": False,
        "motor_commands_sent": 0,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS_LIVE_OBSERVATION_INITIAL_POSE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
