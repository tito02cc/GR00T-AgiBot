#!/usr/bin/env python3
"""Replay all selected frames through the non-actuating G2/GR00T adapter."""

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
sys.path.insert(0, str(REPO_ROOT / "agibot/tools"))

from g2_gr00t_shadow_adapter import (  # noqa: E402
    build_policy_observation,
    check_shadow_target,
    decode_action_chunk,
    g2_gripper_raw_mm_to_training,
    make_right_eef_state,
)

from agibot.configs.xichong_right_single_grasp_config import (  # noqa: E402
    XICHONG_RIGHT_SINGLE_GRASP_CONFIG,
)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.state_action.state_action_processor import StateActionProcessor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "agibot/gr00t_data/xichong_right_single_grasp_300",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=REPO_ROOT / "agibot/data/xichong_right_single_grasp",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "agibot/reports/xichong_right_single_grasp_300_g2_shadow_gate.json",
    )
    return parser.parse_args()


def load_frames(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def rotation_error_deg(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    delta = Rotation.from_quat(quat_a).inv() * Rotation.from_quat(quat_b)
    return float(np.degrees(delta.magnitude()))


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    raw_root = args.raw_root.resolve()
    source_map = json.loads((dataset / "meta/source_episode_map.json").read_text(encoding="utf-8"))

    # Exercise NVIDIA's actual N1.7 state/action processor without loading model
    # weights. Disabling clipping makes apply->unapply an exact mathematical
    # round trip, while retaining the real q01/q99 parameters and relative EEF
    # conversion used by fine-tuning/inference.
    embodiment = "new_embodiment"
    episode_loader = LeRobotEpisodeLoader(dataset, XICHONG_RIGHT_SINGLE_GRASP_CONFIG)
    processor = StateActionProcessor(
        modality_configs={embodiment: XICHONG_RIGHT_SINGLE_GRASP_CONFIG},
        statistics={embodiment: episode_loader.get_dataset_statistics()},
        use_percentiles=True,
        clip_outliers=False,
        use_relative_action=True,
    )

    maxima = {
        "state_eef_abs": 0.0,
        "state_gripper_abs": 0.0,
        "action_xyz_m": 0.0,
        "action_rotation_deg": 0.0,
        "action_gripper_abs": 0.0,
        "raw_gripper_roundtrip_abs": 0.0,
        "shadow_translation_step_m": 0.0,
        "shadow_rotation_step_deg": 0.0,
        "official_processor_eef_roundtrip_abs": 0.0,
        "official_processor_gripper_vs_q01_q99_clipped_abs": 0.0,
        "official_processor_gripper_intentional_clip_abs": 0.0,
    }
    errors: list[str] = []
    frame_count = 0
    observation_contract_checked = False
    processor_windows_checked = 0

    for mapping in source_map["episodes"]:
        subset_index = int(mapping["subset_episode_index"])
        source_episode = mapping["source_episode"]
        raw_frames = load_frames(raw_root / source_episode / "frames.jsonl")
        parquet_path = dataset / f"data/chunk-{subset_index // 1000:03d}/episode_{subset_index:06d}.parquet"
        dataframe = pd.read_parquet(parquet_path)
        if len(dataframe) != len(raw_frames):
            errors.append(f"episode {subset_index}: parquet/raw length mismatch")
            continue

        episode_states = np.stack(dataframe["observation.state"].to_numpy()).astype(np.float32)
        episode_actions = np.stack(dataframe["action"].to_numpy()).astype(np.float32)
        horizon = len(XICHONG_RIGHT_SINGLE_GRASP_CONFIG["action"].delta_indices)
        for start in range(len(dataframe) - horizon + 1):
            state_groups = {
                "right_eef": episode_states[start : start + 1, :9],
                "right_gripper": episode_states[start : start + 1, 9:10],
            }
            action_groups = {
                "right_eef": episode_actions[start : start + horizon, :9],
                "right_gripper": episode_actions[start : start + horizon, 9:10],
            }
            normalized = processor.apply_action(action_groups, embodiment, state_groups)
            restored = processor.unapply_action(normalized, embodiment, state_groups)
            maxima["official_processor_eef_roundtrip_abs"] = max(
                maxima["official_processor_eef_roundtrip_abs"],
                float(np.max(np.abs(restored["right_eef"] - action_groups["right_eef"]))),
            )
            gripper_q01 = processor.norm_params[embodiment]["action"]["right_gripper"]["min"]
            gripper_q99 = processor.norm_params[embodiment]["action"]["right_gripper"]["max"]
            expected_clipped_gripper = np.clip(
                action_groups["right_gripper"], gripper_q01, gripper_q99
            )
            maxima["official_processor_gripper_vs_q01_q99_clipped_abs"] = max(
                maxima["official_processor_gripper_vs_q01_q99_clipped_abs"],
                float(
                    np.max(
                        np.abs(restored["right_gripper"] - expected_clipped_gripper)
                    )
                ),
            )
            maxima["official_processor_gripper_intentional_clip_abs"] = max(
                maxima["official_processor_gripper_intentional_clip_abs"],
                float(
                    np.max(
                        np.abs(expected_clipped_gripper - action_groups["right_gripper"])
                    )
                ),
            )
            processor_windows_checked += 1

        for row_index, (row, raw) in enumerate(zip(dataframe.itertuples(index=False), raw_frames)):
            state = np.asarray(getattr(row, "_1"), dtype=np.float32)
            action = np.asarray(row.action, dtype=np.float32)
            raw_pose = np.asarray(raw["right_ee_pose"], dtype=np.float64)
            raw_next = np.asarray(raw["next_right_ee_pose"], dtype=np.float64)
            raw_gripper = float(raw["right_gripper"]["position"])
            raw_gripper_mm = float(raw["right_gripper"]["position_raw"])

            rebuilt_state_eef = make_right_eef_state(raw_pose)
            maxima["state_eef_abs"] = max(
                maxima["state_eef_abs"], float(np.max(np.abs(rebuilt_state_eef - state[:9])))
            )
            maxima["state_gripper_abs"] = max(
                maxima["state_gripper_abs"], abs(raw_gripper - float(state[9]))
            )
            maxima["raw_gripper_roundtrip_abs"] = max(
                maxima["raw_gripper_roundtrip_abs"],
                abs(float(g2_gripper_raw_mm_to_training(raw_gripper_mm)) - raw_gripper),
            )

            decoded = decode_action_chunk(
                {
                    "right_eef": action[None, None, :9],
                    "right_gripper": action[None, None, 9:10],
                }
            )[0]
            maxima["action_xyz_m"] = max(
                maxima["action_xyz_m"], float(np.max(np.abs(decoded[:3] - raw_next[:3])))
            )
            maxima["action_rotation_deg"] = max(
                maxima["action_rotation_deg"], rotation_error_deg(decoded[3:7], raw_next[3:7])
            )
            maxima["action_gripper_abs"] = max(
                maxima["action_gripper_abs"], abs(float(decoded[7]) - float(raw["next_right_gripper"]))
            )
            shadow = check_shadow_target(raw_pose, decoded)
            maxima["shadow_translation_step_m"] = max(
                maxima["shadow_translation_step_m"], shadow.translation_step_m
            )
            maxima["shadow_rotation_step_deg"] = max(
                maxima["shadow_rotation_step_deg"], shadow.rotation_step_deg
            )

            if not observation_contract_checked:
                black = np.zeros((480, 640, 3), dtype=np.uint8)
                obs = build_policy_observation(
                    black,
                    black,
                    raw_pose,
                    raw_gripper,
                    raw["prompt"],
                )
                expected_shapes = {
                    "head": (1, 1, 480, 640, 3),
                    "wrist": (1, 1, 480, 640, 3),
                    "eef": (1, 1, 9),
                    "gripper": (1, 1, 1),
                }
                actual_shapes = {
                    "head": obs["video"]["head_color"].shape,
                    "wrist": obs["video"]["hand_right"].shape,
                    "eef": obs["state"]["right_eef"].shape,
                    "gripper": obs["state"]["right_gripper"].shape,
                }
                if actual_shapes != expected_shapes:
                    errors.append(f"observation contract shapes: {actual_shapes}")
                observation_contract_checked = True

            frame_count += 1

    tolerances = {
        "state_eef_abs": 1e-6,
        "state_gripper_abs": 1e-6,
        "action_xyz_m": 1e-6,
        "action_rotation_deg": 1e-4,
        "action_gripper_abs": 1e-6,
        # Raw feedback is quantized; the source stores higher precision than float32.
        "raw_gripper_roundtrip_abs": 1e-6,
        "official_processor_eef_roundtrip_abs": 2e-6,
        "official_processor_gripper_vs_q01_q99_clipped_abs": 1e-6,
    }
    for name, tolerance in tolerances.items():
        if maxima[name] > tolerance:
            errors.append(f"{name}={maxima[name]} exceeds {tolerance}")

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "scope": "non-actuating G2/GR00T observation and physical action protocol replay",
        "dataset": str(dataset),
        "raw_root": str(raw_root),
        "episodes_checked": len(source_map["episodes"]),
        "frames_checked": frame_count,
        "official_processor_windows_checked": processor_windows_checked,
        "official_q01_q99_gripper_decode_range": [
            float(processor.norm_params[embodiment]["action"]["right_gripper"]["min"][0]),
            float(processor.norm_params[embodiment]["action"]["right_gripper"]["max"][0]),
        ],
        "observation_contract_checked": observation_contract_checked,
        "max_errors": maxima,
        "tolerances": tolerances,
        "errors": errors,
        "not_proven_by_this_gate": [
            "base-model download and A100 fine-tuning smoke test",
            "trained-checkpoint open-loop accuracy",
            "live G2 camera/TF/GDK transport",
            "robot-cell workspace, speed, acceleration, collision, and E-stop limits",
            "closed-loop task success on the physical G2",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
