#!/usr/bin/env python3
"""Select and materialize a diverse GR00T training subset.

The selector prevents chronological distribution drift from being lost by splitting
the collection into contiguous strata.  Inside each stratum it uses deterministic
farthest-point (k-center) sampling over time-normalized right-arm XYZ, orientation,
gripper, duration, and Cartesian path length features.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


AGIBOT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = AGIBOT_ROOT / "data/xichong_right_single_grasp"
DEFAULT_SOURCE = AGIBOT_ROOT / "gr00t_data/xichong_right_single_grasp"
DEFAULT_OUTPUT = AGIBOT_ROOT / "gr00t_data/xichong_right_single_grasp_300"
DEFAULT_REPORT = AGIBOT_ROOT / "reports/xichong_right_single_grasp_selection_300.json"
CAMERAS = ("head_color", "hand_right")
DEFAULT_MAX_CAMERA_SPAN_MS = 100.0
DEFAULT_MAX_HEAD_STATE_OFFSET_MS = 100.0
DEFAULT_MAX_HAND_STATE_OFFSET_MS = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dataset", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--strata", type=int, default=12)
    parser.add_argument("--resample-points", type=int, default=32)
    parser.add_argument(
        "--no-strict-quality-filter",
        action="store_true",
        help="Disable the raw quality/timing hard gate (not recommended for training).",
    )
    parser.add_argument(
        "--max-camera-span-ms", type=float, default=DEFAULT_MAX_CAMERA_SPAN_MS
    )
    parser.add_argument(
        "--max-head-state-offset-ms",
        type=float,
        default=DEFAULT_MAX_HEAD_STATE_OFFSET_MS,
    )
    parser.add_argument(
        "--max-hand-state-offset-ms",
        type=float,
        default=DEFAULT_MAX_HAND_STATE_OFFSET_MS,
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def allocate_quotas(sizes: list[int], total: int) -> list[int]:
    """Allocate an exact total proportionally, with deterministic remainders."""
    raw = np.asarray(sizes, dtype=np.float64) * (total / sum(sizes))
    quotas = np.floor(raw).astype(int)
    remainder_order = sorted(
        range(len(sizes)), key=lambda i: (-(raw[i] - quotas[i]), i)
    )
    for i in remainder_order[: total - int(quotas.sum())]:
        quotas[i] += 1
    if any(quota > size for quota, size in zip(quotas, sizes)):
        raise ValueError("a stratum quota exceeds its available episodes")
    return quotas.tolist()


def interpolate_episode(
    arrays_path: Path,
    grid: np.ndarray,
    orientation_reference: Rotation,
) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(arrays_path) as arrays:
        states = arrays["states"].astype(np.float64)
        timestamps = arrays["timestamps_monotonic"].astype(np.float64)

    xyz = states[:, 8:11]
    quaternion = states[:, 11:15]
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    gripper = states[:, 15]
    source_grid = np.linspace(0.0, 1.0, len(states))

    xyz_resampled = np.stack(
        [np.interp(grid, source_grid, xyz[:, axis]) for axis in range(3)], axis=1
    )
    grip_resampled = np.interp(grid, source_grid, gripper)[:, None]
    rotations = Slerp(source_grid, Rotation.from_quat(quaternion))(grid)
    relative_rotvec = (orientation_reference.inv() * rotations).as_rotvec()

    duration = float(timestamps[-1] - timestamps[0])
    path_length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())

    # Each trajectory group contributes its mean squared normalized distance.
    # Physical scales: 20 mm translation ~= 10 deg rotation ~= 0.2 gripper units.
    feature = np.concatenate(
        [
            (xyz_resampled / 0.020).reshape(-1) / math.sqrt(xyz_resampled.size),
            (relative_rotvec / np.deg2rad(10.0)).reshape(-1)
            / math.sqrt(relative_rotvec.size),
            (grip_resampled / 0.20).reshape(-1) / math.sqrt(grip_resampled.size),
            np.asarray([duration / 2.0, path_length / 0.10]),
        ]
    )
    summary = {
        "frames": len(states),
        "duration_s": duration,
        "path_length_m": path_length,
        "start_xyz": xyz[0].tolist(),
        "end_xyz": xyz[-1].tolist(),
        "resampled_xyz": xyz_resampled,
    }
    return feature, summary


def pairwise_squared(features: np.ndarray) -> np.ndarray:
    norms = np.sum(features * features, axis=1)
    return np.maximum(norms[:, None] + norms[None, :] - 2.0 * features @ features.T, 0.0)


def kcenter_indices(features: np.ndarray, count: int) -> list[int]:
    """Return deterministic representative/farthest-point indices."""
    if count >= len(features):
        return list(range(len(features)))
    distances = pairwise_squared(features)
    selected = [int(np.argmin(distances.mean(axis=1)))]
    nearest = distances[selected[0]].copy()
    while len(selected) < count:
        nearest[selected] = -1.0
        candidate = int(np.argmax(nearest))
        selected.append(candidate)
        nearest = np.minimum(nearest, distances[candidate])
    return selected


def quantiles(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        key: round(float(value), 6)
        for key, value in zip(("min", "p05", "p25", "p50", "p75", "p95", "max"), points)
    }


def raw_quality_gate(
    episode_dir: Path,
    max_camera_span_ms: float,
    max_head_state_offset_ms: float,
    max_hand_state_offset_ms: float,
) -> tuple[bool, dict[str, Any]]:
    """Apply a conservative gate on top of the collector's accepted contract.

    The collector retrieves the head and wrist images sequentially and stamps the
    robot state after image retrieval. The offsets are therefore expected to be
    negative, but an isolated retrieval/control stall can associate an image with
    the wrong 10 Hz control step. The limits below remove those tail events.
    """
    quality = json.loads((episode_dir / "quality_report.json").read_text())
    frames = read_jsonl(episode_dir / "frames.jsonl")
    max_offsets: dict[str, float] = {}
    for camera in CAMERAS:
        offsets = [
            abs(
                float(frame["image_meta"][camera]["software_midpoint_monotonic"])
                - float(frame["timestamp_monotonic"])
            )
            * 1000.0
            for frame in frames
        ]
        max_offsets[camera] = max(offsets, default=float("inf"))

    duplicate_counts = quality["images"]["duplicate_previous_counts"]
    checks = {
        "collector_contract_ok": bool(quality["contract"]["ok"]),
        "collector_warnings_empty": not quality["contract"]["warnings"],
        "zero_late_frames": int(quality["timing"]["late_frame_count"]) == 0,
        "zero_duplicate_images": sum(int(v) for v in duplicate_counts.values()) == 0,
        "camera_span_within_limit": float(
            quality["images"]["software_retrieval_sync"]["max_span_ms"]
        )
        <= max_camera_span_ms,
        "head_state_offset_within_limit": max_offsets["head_color"]
        <= max_head_state_offset_ms,
        "hand_state_offset_within_limit": max_offsets["hand_right"]
        <= max_hand_state_offset_ms,
    }
    details = {
        "checks": checks,
        "collector_warnings": quality["contract"]["warnings"],
        "late_frame_count": int(quality["timing"]["late_frame_count"]),
        "max_camera_span_ms": float(
            quality["images"]["software_retrieval_sync"]["max_span_ms"]
        ),
        "max_abs_camera_state_offset_ms": max_offsets,
        "duplicate_previous_counts": duplicate_counts,
    }
    return all(checks.values()), details


def coverage_report(
    summaries: list[dict[str, Any]], selected: list[int], reserve: list[int]
) -> dict[str, Any]:
    all_xyz = np.stack([row["resampled_xyz"] for row in summaries])
    selected_xyz = all_xyz[selected]
    reserve_xyz = all_xyz[reserve]
    flat_selected = selected_xyz.reshape(len(selected_xyz), -1)
    flat_reserve = reserve_xyz.reshape(len(reserve_xyz), -1)
    d2 = (
        np.sum(flat_reserve * flat_reserve, axis=1)[:, None]
        + np.sum(flat_selected * flat_selected, axis=1)[None, :]
        - 2.0 * flat_reserve @ flat_selected.T
    )
    nearest_rms_mm = np.sqrt(
        np.maximum(d2.min(axis=1), 0.0) / (all_xyz.shape[1] * all_xyz.shape[2])
    ) * 1000.0

    def group(indices: list[int]) -> dict[str, Any]:
        frames = np.asarray([summaries[i]["frames"] for i in indices])
        durations = np.asarray([summaries[i]["duration_s"] for i in indices])
        paths = np.asarray([summaries[i]["path_length_m"] for i in indices])
        endpoints = np.asarray([summaries[i]["end_xyz"] for i in indices])
        return {
            "episodes": len(indices),
            "frames": int(frames.sum()),
            "duration_s": quantiles(durations),
            "path_length_m": quantiles(paths),
            "end_xyz_mean": np.mean(endpoints, axis=0).round(6).tolist(),
            "end_xyz_std": np.std(endpoints, axis=0).round(6).tolist(),
            "end_xyz_min": np.min(endpoints, axis=0).round(6).tolist(),
            "end_xyz_max": np.max(endpoints, axis=0).round(6).tolist(),
        }

    return {
        "all": group(list(range(len(summaries)))),
        "selected": group(selected),
        "reserve": group(reserve),
        "reserve_to_selected_nearest_xyz_rms_mm": quantiles(nearest_rms_mm),
    }


def materialize_subset(
    source: Path,
    destination: Path,
    selected: list[int],
    source_map: dict[int, dict[str, Any]],
) -> None:
    if destination.exists():
        raise FileExistsError(
            f"output already exists: {destination}; remove or rename it explicitly before rerunning"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        source_meta = source / "meta"
        episodes = read_jsonl(source_meta / "episodes.jsonl")
        tasks = read_jsonl(source_meta / "tasks.jsonl")
        info = json.loads((source_meta / "info.json").read_text())
        (temporary / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_meta / "modality.json", temporary / "meta/modality.json")
        write_jsonl(temporary / "meta/tasks.jsonl", tasks)

        output_episodes: list[dict[str, Any]] = []
        output_map: list[dict[str, Any]] = []
        global_index = 0
        for target_index, source_index in enumerate(selected):
            source_episode = episodes[source_index]
            length = int(source_episode["length"])
            source_parquet = (
                source / "data/chunk-000" / f"episode_{source_index:06d}.parquet"
            )
            dataframe = pd.read_parquet(source_parquet)
            if len(dataframe) != length:
                raise RuntimeError(f"{source_parquet}: unexpected row count")
            dataframe["episode_index"] = np.full(length, target_index, dtype=np.int64)
            dataframe["index"] = np.arange(
                global_index, global_index + length, dtype=np.int64
            )
            target_parquet = (
                temporary / "data/chunk-000" / f"episode_{target_index:06d}.parquet"
            )
            target_parquet.parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_parquet(target_parquet, index=False)

            for camera in CAMERAS:
                source_video = (
                    source
                    / "videos/chunk-000"
                    / f"observation.images.{camera}"
                    / f"episode_{source_index:06d}.mp4"
                )
                target_video = (
                    temporary
                    / "videos/chunk-000"
                    / f"observation.images.{camera}"
                    / f"episode_{target_index:06d}.mp4"
                )
                target_video.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source_video, target_video)
                except OSError:
                    shutil.copy2(source_video, target_video)

            output_episodes.append(
                {
                    "episode_index": target_index,
                    "tasks": source_episode["tasks"],
                    "length": length,
                }
            )
            mapping = source_map[source_index]
            output_map.append(
                {
                    "source_episode": mapping["source_episode"],
                    "full_dataset_episode_index": source_index,
                    "subset_episode_index": target_index,
                    "length": length,
                }
            )
            global_index += length

        info["total_episodes"] = len(selected)
        info["total_frames"] = global_index
        info["total_videos"] = len(selected) * len(CAMERAS)
        info["total_chunks"] = math.ceil(len(selected) / int(info["chunks_size"]))
        info["splits"] = {"train": f"0:{len(selected)}"}
        write_json(temporary / "meta/info.json", info)
        write_jsonl(temporary / "meta/episodes.jsonl", output_episodes)
        write_json(
            temporary / "meta/source_episode_map.json",
            {
                "source_dataset": str(source.resolve()),
                "episodes": output_map,
            },
        )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    source = args.source_dataset.resolve()
    destination = args.output_dataset.resolve()
    report_path = args.report.resolve()
    if args.count <= 0 or args.strata <= 0 or args.resample_points < 2:
        raise ValueError("count/strata/resample-points must be positive")

    source_map_data = json.loads((source / "meta/source_episode_map.json").read_text())
    source_map = {
        int(row["target_episode_index"]): row for row in source_map_data["episodes"]
    }
    episode_indices = sorted(source_map)
    if episode_indices != list(range(len(episode_indices))):
        raise ValueError("full dataset episode indices are not contiguous")

    quality_audit: dict[int, dict[str, Any]] = {}
    eligible_indices: list[int] = []
    for episode_index in episode_indices:
        if args.no_strict_quality_filter:
            eligible_indices.append(episode_index)
            continue
        episode_dir = raw_root / source_map[episode_index]["source_episode"]
        accepted, details = raw_quality_gate(
            episode_dir,
            args.max_camera_span_ms,
            args.max_head_state_offset_ms,
            args.max_hand_state_offset_ms,
        )
        quality_audit[episode_index] = details
        if accepted:
            eligible_indices.append(episode_index)
    if args.count > len(eligible_indices):
        raise ValueError(
            f"requested {args.count} from only {len(eligible_indices)} quality-eligible episodes"
        )

    first_arrays = raw_root / source_map[0]["source_episode"] / "arrays.npz"
    with np.load(first_arrays) as arrays:
        reference = Rotation.from_quat(arrays["states"][0, 11:15].astype(np.float64))
    grid = np.linspace(0.0, 1.0, args.resample_points)
    features: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        arrays_path = raw_root / source_map[episode_index]["source_episode"] / "arrays.npz"
        feature, summary = interpolate_episode(arrays_path, grid, reference)
        features.append(feature)
        summaries.append(summary)
    feature_array = np.stack(features)

    # Keep chronological boundaries based on the full collection; filtering must
    # not silently shift later episodes into earlier collection strata.
    full_strata = [part.tolist() for part in np.array_split(episode_indices, args.strata)]
    eligible_set = set(eligible_indices)
    strata = [[index for index in part if index in eligible_set] for part in full_strata]
    if any(not part for part in strata):
        raise ValueError("strict filtering left an empty chronological stratum")
    quotas = allocate_quotas([len(part) for part in strata], args.count)
    selected: list[int] = []
    stratum_report: list[dict[str, Any]] = []
    for stratum_index, (members, quota) in enumerate(zip(strata, quotas)):
        local = kcenter_indices(feature_array[members], quota)
        chosen = sorted(members[i] for i in local)
        selected.extend(chosen)
        stratum_report.append(
            {
                "stratum": stratum_index,
                "source_index_start": full_strata[stratum_index][0],
                "source_index_end": full_strata[stratum_index][-1],
                "available_before_quality_filter": len(full_strata[stratum_index]),
                "available": len(members),
                "selected": len(chosen),
                "selected_source_indices": chosen,
            }
        )
    selected = sorted(selected)
    reserve = sorted(eligible_set - set(selected))
    if len(selected) != args.count:
        raise RuntimeError(f"selected {len(selected)} episodes, expected {args.count}")

    materialize_subset(source, destination, selected, source_map)
    report = {
        "source_dataset": str(source),
        "output_dataset": str(destination),
        "algorithm": {
            "name": "chronological_stratified_kcenter",
            "count": args.count,
            "strata": args.strata,
            "resample_points": args.resample_points,
            "trajectory_scales": {
                "xyz_m": 0.020,
                "orientation_deg": 10.0,
                "gripper": 0.20,
                "duration_s": 2.0,
                "path_length_m": 0.10,
            },
        },
        "quality_filter": {
            "enabled": not args.no_strict_quality_filter,
            "eligible_episodes": len(eligible_indices),
            "excluded_episodes": len(episode_indices) - len(eligible_indices),
            "thresholds": {
                "max_camera_span_ms": args.max_camera_span_ms,
                "max_head_state_offset_ms": args.max_head_state_offset_ms,
                "max_hand_state_offset_ms": args.max_hand_state_offset_ms,
                "collector_warnings": 0,
                "late_frames": 0,
                "duplicate_images": 0,
            },
            "excluded": [
                {
                    "full_dataset_episode_index": index,
                    "source_episode": source_map[index]["source_episode"],
                    **quality_audit[index],
                }
                for index in episode_indices
                if index not in eligible_set
            ],
        },
        "strata": stratum_report,
        "selected_full_dataset_indices": selected,
        "selected_source_episodes": [source_map[i]["source_episode"] for i in selected],
        "reserve_full_dataset_indices": reserve,
        "coverage": coverage_report(summaries, selected, reserve),
    }
    write_json(report_path, report)
    print(
        f"Selected {len(selected)} of {len(episode_indices)} episodes; "
        f"frames={report['coverage']['selected']['frames']}"
    )
    print(f"Dataset: {destination}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
