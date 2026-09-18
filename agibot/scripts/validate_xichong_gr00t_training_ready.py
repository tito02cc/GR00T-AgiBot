#!/usr/bin/env python3
"""Hard-gate an Agibot xichong GR00T dataset before paid training.

This validator independently reconstructs every converted state/action from the
selected raw episodes, recomputes normalization statistics, fully decodes every
video, checks video/source-image correspondence, exercises the official episode
loader, and writes a SHA-256 upload manifest. A non-zero exit status means the
dataset must not be used for training.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm


AGIBOT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = AGIBOT_ROOT / "data/xichong_right_single_grasp"
DEFAULT_DATASET = AGIBOT_ROOT / "gr00t_data/xichong_right_single_grasp_300"
DEFAULT_REPORT = AGIBOT_ROOT / "reports/xichong_right_single_grasp_300_hard_gate.json"
DEFAULT_MANIFEST = AGIBOT_ROOT / "reports/xichong_right_single_grasp_300.sha256"
CAMERAS = ("head_color", "hand_right")
HORIZON = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-camera-span-ms", type=float, default=100.0)
    parser.add_argument("--max-head-state-offset-ms", type=float, default=100.0)
    parser.add_argument("--max-hand-state-offset-ms", type=float, default=50.0)
    parser.add_argument("--min-video-source-psnr-db", type=float, default=25.0)
    parser.add_argument("--initial-open-gripper-max", type=float, default=-0.7)
    parser.add_argument("--held-gripper-min", type=float, default=-0.55)
    parser.add_argument("--min-grasp-lift-m", type=float, default=0.03)
    parser.add_argument("--terminal-hold-frames", type=int, default=5)
    parser.add_argument("--terminal-stability-max-m", type=float, default=0.01)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def matrix_to_rot6d(matrices: np.ndarray) -> np.ndarray:
    return np.asarray(matrices)[..., :2, :].reshape(*np.asarray(matrices).shape[:-2], 6)


def rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64).reshape(-1, 2, 3)
    row0 = rows[:, 0]
    row0 = row0 / np.linalg.norm(row0, axis=1, keepdims=True)
    row1 = rows[:, 1] - np.sum(row0 * rows[:, 1], axis=1, keepdims=True) * row0
    row1 = row1 / np.linalg.norm(row1, axis=1, keepdims=True)
    row2 = np.cross(row0, row1)
    return np.stack((row0, row1, row2), axis=1)


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - right), initial=0.0))


def array_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "q01": np.quantile(values, 0.01, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }


def compare_stats(
    actual: dict[str, Any], expected: dict[str, np.ndarray], prefix: str, errors: list[str]
) -> float:
    largest = 0.0
    for key, expected_value in expected.items():
        if key not in actual:
            errors.append(f"{prefix}: missing statistic {key}")
            continue
        actual_value = np.asarray(actual[key], dtype=np.float64)
        if actual_value.shape != expected_value.shape:
            errors.append(
                f"{prefix}.{key}: shape {actual_value.shape} != {expected_value.shape}"
            )
            continue
        difference = max_abs(actual_value, expected_value)
        largest = max(largest, difference)
        if not np.isfinite(actual_value).all() or difference > 2e-5:
            errors.append(f"{prefix}.{key}: nonfinite or max_abs_error={difference:.3g}")
    return largest


def source_timing(frames: list[dict[str, Any]]) -> dict[str, float]:
    offsets: dict[str, float] = {}
    for camera in CAMERAS:
        offsets[camera] = max(
            abs(
                float(frame["image_meta"][camera]["software_midpoint_monotonic"])
                - float(frame["timestamp_monotonic"])
            )
            * 1000.0
            for frame in frames
        )
    spans = []
    for frame in frames:
        midpoints = [
            float(meta["software_midpoint_monotonic"])
            for meta in frame["image_meta"].values()
        ]
        spans.append((max(midpoints) - min(midpoints)) * 1000.0)
    return {
        "max_head_state_offset_ms": offsets["head_color"],
        "max_hand_state_offset_ms": offsets["hand_right"],
        "max_camera_span_ms": max(spans),
    }


def validate_video(
    video: Path,
    source_dir: Path,
    camera: str,
    expected_frames: int,
    min_psnr: float,
) -> dict[str, Any]:
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(video),
    ]
    stream = json.loads(subprocess.check_output(probe_command, text=True))["streams"][0]
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    checks = {
        "codec_h264": stream.get("codec_name") == "h264",
        "size_640x480": (int(stream["width"]), int(stream["height"])) == (640, 480),
        "pixel_format_yuv420p": stream.get("pix_fmt") == "yuv420p",
        "fps_10": stream.get("r_frame_rate") == "10/1",
        "frame_count": int(stream.get("nb_read_frames", -1)) == expected_frames,
    }

    capture = cv2.VideoCapture(str(video))
    psnrs: list[float] = []
    for frame_index in sorted({0, expected_frames // 2, expected_frames - 1}):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, decoded = capture.read()
        source = cv2.imread(str(source_dir / "images" / f"{camera}_{frame_index:06d}.jpg"))
        if not ok or decoded is None or source is None or decoded.shape != source.shape:
            psnrs.append(0.0)
            continue
        mse = float(np.mean((decoded.astype(np.float32) - source.astype(np.float32)) ** 2))
        psnrs.append(float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse))
    capture.release()
    minimum_psnr = min(psnrs, default=0.0)
    checks["source_frame_correspondence"] = minimum_psnr >= min_psnr
    return {
        "path": str(video),
        "checks": checks,
        "sample_psnr_db": psnrs,
        "minimum_sample_psnr_db": minimum_psnr,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    dataset = args.dataset.resolve()
    report_path = args.report.resolve()
    manifest_path = args.manifest.resolve()
    errors: list[str] = []
    info = read_json(dataset / "meta/info.json")
    episodes = read_jsonl(dataset / "meta/episodes.jsonl")
    tasks = {row["task_index"]: row["task"] for row in read_jsonl(dataset / "meta/tasks.jsonl")}
    source_map_rows = read_json(dataset / "meta/source_episode_map.json")["episodes"]
    if len(episodes) != 300 or len(source_map_rows) != 300:
        errors.append(
            f"metadata episode count is not 300: episodes={len(episodes)}, map={len(source_map_rows)}"
        )
    if int(info["total_episodes"]) != len(episodes):
        errors.append("info.total_episodes disagrees with episodes.jsonl")

    metric_maxima = {
        "state_xyz_m": 0.0,
        "state_rot6d": 0.0,
        "state_gripper": 0.0,
        "action_xyz_m": 0.0,
        "action_rot6d": 0.0,
        "action_gripper": 0.0,
        "raw_next_xyz_m": 0.0,
        "raw_next_rotation_deg": 0.0,
        "raw_next_gripper": 0.0,
        "timestamp_s": 0.0,
    }
    timing_maxima = {
        "max_camera_span_ms": 0.0,
        "max_head_state_offset_ms": 0.0,
        "max_hand_state_offset_ms": 0.0,
    }
    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    relative_trajectories: list[np.ndarray] = []
    terminal_semantic_details: list[dict[str, Any]] = []
    video_jobs: list[tuple[Path, Path, str, int, float]] = []
    expected_global_index = 0

    for target_index, (episode_meta, mapping) in enumerate(
        tqdm(zip(episodes, source_map_rows), total=len(episodes), desc="Auditing fields")
    ):
        source_dir = raw_root / mapping["source_episode"]
        frames = read_jsonl(source_dir / "frames.jsonl")
        quality = read_json(source_dir / "quality_report.json")
        with np.load(source_dir / "arrays.npz") as arrays:
            raw_states = arrays["states"].astype(np.float64)
            raw_actions = arrays["actions"].astype(np.float64)

        terminal_quality = quality.get("xichong_terminal", {})
        terminal_metrics = terminal_quality.get("metrics", {})
        held_indices = np.flatnonzero(raw_states[:, 15] >= args.held_gripper_min)
        semantic_detail: dict[str, Any] = {
            "episode_index": target_index,
            "source_episode": mapping["source_episode"],
            "source_terminal_ok": terminal_quality.get("ok") is True,
            "initial_gripper": float(raw_states[0, 15]),
            "terminal_gripper": float(raw_states[-1, 15]),
        }
        if terminal_quality.get("ok") is not True:
            errors.append(f"episode {target_index}: source xichong terminal gate is not ok")
        if raw_states[0, 15] > args.initial_open_gripper_max:
            errors.append(f"episode {target_index}: gripper is not initially open")
        if held_indices.size == 0:
            semantic_detail["semantic_ok"] = False
            errors.append(f"episode {target_index}: gripper never reaches held threshold")
        else:
            held_anchor = int(held_indices[0])
            terminal_run = 0
            for gripper_position in raw_states[::-1, 15]:
                if gripper_position < args.held_gripper_min:
                    break
                terminal_run += 1
            lift_m = float(raw_states[-1, 10] - raw_states[held_anchor, 10])
            tail_count = min(args.terminal_hold_frames, len(raw_states))
            terminal_xyz = raw_states[-tail_count:, 8:11]
            stability_radius_m = float(
                np.max(np.linalg.norm(terminal_xyz - terminal_xyz[-1], axis=1), initial=0.0)
            )
            source_lift_m = terminal_metrics.get("lift_z_m", {}).get("right")
            lift_report_error_m = (
                abs(lift_m - float(source_lift_m)) if source_lift_m is not None else math.inf
            )
            semantic_ok = (
                raw_states[-1, 15] >= args.held_gripper_min
                and lift_m >= args.min_grasp_lift_m
                and terminal_run >= args.terminal_hold_frames
                and stability_radius_m <= args.terminal_stability_max_m
                and lift_report_error_m <= 2e-6
            )
            semantic_detail.update(
                {
                    "held_anchor_index": held_anchor,
                    "lift_from_grasp_anchor_m": lift_m,
                    "source_reported_lift_m": source_lift_m,
                    "lift_report_error_m": lift_report_error_m,
                    "terminal_held_run_frames": terminal_run,
                    "terminal_stability_radius_m": stability_radius_m,
                    "semantic_ok": semantic_ok,
                }
            )
            if not semantic_ok:
                errors.append(
                    f"episode {target_index}: grasp/lift/terminal-hold semantic gate failed"
                )
        terminal_semantic_details.append(semantic_detail)
        length = len(frames)
        parquet_path = dataset / "data/chunk-000" / f"episode_{target_index:06d}.parquet"
        dataframe = pd.read_parquet(parquet_path)
        if length != len(dataframe) or length != int(episode_meta["length"]):
            errors.append(f"episode {target_index}: raw/parquet/metadata length mismatch")
            continue
        state = np.stack(dataframe["observation.state"].to_numpy()).astype(np.float64)
        action = np.stack(dataframe["action"].to_numpy()).astype(np.float64)
        if state.shape != (length, 10) or action.shape != (length, 10):
            errors.append(f"episode {target_index}: state/action shape mismatch")
            continue
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            errors.append(f"episode {target_index}: state/action contains NaN or Inf")

        state_rotation = Rotation.from_quat(raw_states[:, 11:15])
        expected_state_rot6d = matrix_to_rot6d(state_rotation.as_matrix())
        target_xyz = raw_states[:, 8:11] + raw_actions[:, 7:10]
        target_rotation = Rotation.from_rotvec(raw_actions[:, 10:13]) * state_rotation
        expected_action_rot6d = matrix_to_rot6d(target_rotation.as_matrix())
        comparisons = {
            "state_xyz_m": max_abs(state[:, :3], raw_states[:, 8:11]),
            "state_rot6d": max_abs(state[:, 3:9], expected_state_rot6d),
            "state_gripper": max_abs(state[:, 9], raw_states[:, 15]),
            "action_xyz_m": max_abs(action[:, :3], target_xyz),
            "action_rot6d": max_abs(action[:, 3:9], expected_action_rot6d),
            "action_gripper": max_abs(action[:, 9], raw_actions[:, 13]),
        }
        for key, value in comparisons.items():
            metric_maxima[key] = max(metric_maxima[key], value)
            if value > 2e-6:
                errors.append(f"episode {target_index}: {key} max_abs_error={value:.3g}")

        frame_next_xyz = np.asarray([frame["next_right_ee_pose"][:3] for frame in frames])
        frame_next_quat = np.asarray([frame["next_right_ee_pose"][3:] for frame in frames])
        frame_next_gripper = np.asarray([frame["next_right_gripper"] for frame in frames])
        raw_xyz_error = max_abs(target_xyz, frame_next_xyz)
        raw_angle_error = float(
            np.rad2deg(
                np.max((target_rotation.inv() * Rotation.from_quat(frame_next_quat)).magnitude())
            )
        )
        raw_gripper_error = max_abs(raw_actions[:, 13], frame_next_gripper)
        metric_maxima["raw_next_xyz_m"] = max(metric_maxima["raw_next_xyz_m"], raw_xyz_error)
        metric_maxima["raw_next_rotation_deg"] = max(
            metric_maxima["raw_next_rotation_deg"], raw_angle_error
        )
        metric_maxima["raw_next_gripper"] = max(
            metric_maxima["raw_next_gripper"], raw_gripper_error
        )
        if raw_xyz_error > 2e-5 or raw_angle_error > 0.002 or raw_gripper_error > 2e-5:
            errors.append(f"episode {target_index}: raw next-state action semantics failed")

        timestamp = dataframe["timestamp"].to_numpy(dtype=np.float64)
        expected_timestamp = np.arange(length, dtype=np.float64) / float(info["fps"])
        timestamp_error = max_abs(timestamp, expected_timestamp)
        metric_maxima["timestamp_s"] = max(metric_maxima["timestamp_s"], timestamp_error)
        if timestamp_error > 1e-6:
            errors.append(f"episode {target_index}: timestamp is off fixed 10 Hz grid")
        expected_index = np.arange(expected_global_index, expected_global_index + length)
        if not np.array_equal(dataframe["index"].to_numpy(), expected_index):
            errors.append(f"episode {target_index}: global index is not contiguous")
        if not np.array_equal(dataframe["frame_index"].to_numpy(), np.arange(length)):
            errors.append(f"episode {target_index}: frame index is not contiguous")
        if not np.all(dataframe["episode_index"].to_numpy() == target_index):
            errors.append(f"episode {target_index}: wrong episode_index column")
        task_indices = np.unique(dataframe["task_index"].to_numpy())
        if len(task_indices) != 1 or tasks.get(int(task_indices[0])) not in episode_meta["tasks"]:
            errors.append(f"episode {target_index}: task mapping mismatch")
        expected_global_index += length

        timing = source_timing(frames)
        for key, value in timing.items():
            timing_maxima[key] = max(timing_maxima[key], value)
        if quality["contract"]["warnings"] or quality["timing"]["late_frame_count"]:
            errors.append(f"episode {target_index}: raw collector warning/late frame present")
        if sum(quality["images"]["duplicate_previous_counts"].values()):
            errors.append(f"episode {target_index}: duplicate source image present")
        if timing["max_camera_span_ms"] > args.max_camera_span_ms:
            errors.append(f"episode {target_index}: camera span exceeds hard gate")
        if timing["max_head_state_offset_ms"] > args.max_head_state_offset_ms:
            errors.append(f"episode {target_index}: head/state offset exceeds hard gate")
        if timing["max_hand_state_offset_ms"] > args.max_hand_state_offset_ms:
            errors.append(f"episode {target_index}: hand/state offset exceeds hard gate")

        state_matrices = rot6d_to_matrix(state[:, 3:9])
        action_matrices = rot6d_to_matrix(action[:, 3:9])
        for start in range(length - HORIZON + 1):
            reference_rotation = state_matrices[start]
            relative_xyz = (action[start : start + HORIZON, :3] - state[start, :3]) @ reference_rotation
            relative_rotation = (
                reference_rotation.T[None, :, :] @ action_matrices[start : start + HORIZON]
            )
            relative_trajectories.append(
                np.concatenate((relative_xyz, matrix_to_rot6d(relative_rotation)), axis=1).astype(
                    np.float32
                )
            )

        all_states.append(state.astype(np.float32))
        all_actions.append(action.astype(np.float32))
        all_timestamps.append(timestamp.astype(np.float32)[:, None])
        for camera in CAMERAS:
            video_jobs.append(
                (
                    dataset
                    / "videos/chunk-000"
                    / f"observation.images.{camera}"
                    / f"episode_{target_index:06d}.mp4",
                    source_dir,
                    camera,
                    length,
                    args.min_video_source_psnr_db,
                )
            )

    if expected_global_index != int(info["total_frames"]):
        errors.append("info.total_frames disagrees with audited Parquet rows")

    absolute_stats = read_json(dataset / "meta/stats.json")
    absolute_stat_error = 0.0
    for key, values in (
        ("action", np.concatenate(all_actions)),
        ("observation.state", np.concatenate(all_states)),
        ("timestamp", np.concatenate(all_timestamps)),
    ):
        absolute_stat_error = max(
            absolute_stat_error,
            compare_stats(absolute_stats.get(key, {}), array_stats(values), key, errors),
        )
    relative_stats = read_json(dataset / "meta/relative_stats.json")
    independent_relative = np.stack(relative_trajectories)
    relative_stat_error = compare_stats(
        relative_stats.get("right_eef", {}),
        array_stats(independent_relative),
        "relative_stats.right_eef",
        errors,
    )

    video_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(validate_video, *job): job[0] for job in video_jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Decoding videos"):
            path = futures[future]
            try:
                result = future.result()
                video_results.append(result)
                failed = [name for name, passed in result["checks"].items() if not passed]
                if failed:
                    errors.append(f"{path}: video checks failed: {failed}")
            except Exception as exc:  # noqa: BLE001 - aggregate every corrupt video
                errors.append(f"{path}: video validation exception: {type(exc).__name__}: {exc}")
    video_results.sort(key=lambda row: row["path"])

    try:
        repository_root = AGIBOT_ROOT.parent
        if str(repository_root) not in sys.path:
            sys.path.insert(0, str(repository_root))
        import agibot.configs.xichong_right_single_grasp_config  # noqa: F401
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.embodiment_tags import EmbodimentTag

        loader = LeRobotEpisodeLoader(dataset, MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value])
        loader_shapes = {}
        for index in (0, len(episodes) // 2, len(episodes) - 1):
            loaded = loader[index]
            loader_shapes[str(index)] = {column: list(loaded[column].shape) for column in loaded}
    except Exception as exc:  # noqa: BLE001 - loader compatibility is a hard gate
        loader_shapes = {}
        errors.append(f"official loader failed: {type(exc).__name__}: {exc}")

    dataset_files = sorted(path for path in dataset.rglob("*") if path.is_file())
    hashes: dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(sha256_file, path): path for path in dataset_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Hashing dataset"):
            hashes[futures[future]] = future.result()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        "".join(f"{hashes[path]}  {path.relative_to(dataset)}\n" for path in dataset_files)
    )
    temporary_manifest.replace(manifest_path)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset": str(dataset),
        "raw_root": str(raw_root),
        "status": "PASS" if not errors else "FAIL",
        "hard_gate_errors": errors,
        "counts": {
            "episodes": len(episodes),
            "frames": expected_global_index,
            "videos": len(video_results),
            "dataset_files_hashed": len(dataset_files),
            "relative_action_windows": len(relative_trajectories),
        },
        "conversion_max_errors": metric_maxima,
        "source_timing_maxima": timing_maxima,
        "normalization_stats_max_errors": {
            "absolute": absolute_stat_error,
            "relative": relative_stat_error,
        },
        "task_semantic_gate": {
            "description": (
                "right gripper starts open, closes on the metal workpiece, lifts it at "
                "least 3 cm from the grasp anchor, and holds it stably at the end"
            ),
            "thresholds": {
                "initial_open_gripper_max": args.initial_open_gripper_max,
                "held_gripper_min": args.held_gripper_min,
                "min_grasp_lift_m": args.min_grasp_lift_m,
                "terminal_hold_frames": args.terminal_hold_frames,
                "terminal_stability_max_m": args.terminal_stability_max_m,
            },
            "checked": len(terminal_semantic_details),
            "passed": sum(
                detail.get("semantic_ok") is True for detail in terminal_semantic_details
            ),
            "minimum_lift_from_grasp_anchor_m": min(
                (
                    detail["lift_from_grasp_anchor_m"]
                    for detail in terminal_semantic_details
                    if "lift_from_grasp_anchor_m" in detail
                ),
                default=None,
            ),
            "minimum_terminal_held_run_frames": min(
                (
                    detail["terminal_held_run_frames"]
                    for detail in terminal_semantic_details
                    if "terminal_held_run_frames" in detail
                ),
                default=None,
            ),
            "maximum_terminal_stability_radius_m": max(
                (
                    detail["terminal_stability_radius_m"]
                    for detail in terminal_semantic_details
                    if "terminal_stability_radius_m" in detail
                ),
                default=None,
            ),
            "details": terminal_semantic_details,
        },
        "video": {
            "minimum_source_psnr_db": min(
                (row["minimum_sample_psnr_db"] for row in video_results), default=0.0
            ),
            "fully_decoded": len(video_results),
            "details": video_results,
        },
        "official_loader_sample_shapes": loader_shapes,
        "sha256_manifest": str(manifest_path),
    }
    write_json_atomic(report_path, report)
    print(
        f"Hard gate {report['status']}: episodes={len(episodes)} frames={expected_global_index} "
        f"videos={len(video_results)} errors={len(errors)}"
    )
    print(f"Report: {report_path}")
    print(f"SHA-256 manifest: {manifest_path}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
