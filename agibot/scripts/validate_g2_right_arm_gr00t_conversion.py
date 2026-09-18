#!/usr/bin/env python3
"""Independently validate a converted G2 right-arm GR00T/LeRobot dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm


CAMERAS = ("head_color", "hand_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minimum-psnr-db", type=float, default=25.0)
    parser.add_argument(
        "--all-source-frames",
        action="store_true",
        help="Compare every sequential decoded frame with its source JPEG",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def rot6d(rotation: Rotation) -> np.ndarray:
    return rotation.as_matrix()[..., :2, :].reshape(-1, 6)


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right)), initial=0.0))


def validate_video(
    path: Path,
    source_images: Path,
    camera: str,
    expected_frames: int,
    expected_fps: float,
    minimum_psnr_db: float,
    all_source_frames: bool = False,
) -> dict[str, Any]:
    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    stream = json.loads(probe)["streams"][0]
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
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

    numerator, denominator = stream["avg_frame_rate"].split("/")
    actual_fps = float(numerator) / float(denominator)
    checks = {
        "codec_h264": stream.get("codec_name") == "h264",
        "size_640x480": (int(stream["width"]), int(stream["height"])) == (640, 480),
        "pixel_format_yuv420p": stream.get("pix_fmt") == "yuv420p",
        "fps": math.isclose(actual_fps, expected_fps, abs_tol=1e-9),
        "fully_decoded_frame_count": int(stream.get("nb_read_frames", -1)) == expected_frames,
    }

    capture = cv2.VideoCapture(str(path))
    psnr_values: list[float] = []
    indices = (
        range(expected_frames)
        if all_source_frames
        else sorted({0, expected_frames // 2, expected_frames - 1})
    )
    for frame_index in indices:
        if not all_source_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, decoded = capture.read()
        source = cv2.imread(str(source_images / f"{camera}_{frame_index:06d}.jpg"))
        if not ok or decoded is None or source is None or decoded.shape != source.shape:
            psnr_values.append(0.0)
            continue
        mse = float(np.mean((decoded.astype(np.float32) - source.astype(np.float32)) ** 2))
        psnr_values.append(float("inf") if mse == 0 else 10 * math.log10(255**2 / mse))
    capture.release()
    minimum_sample_psnr = min(psnr_values, default=0.0)
    checks["source_frame_correspondence"] = minimum_sample_psnr >= minimum_psnr_db
    return {
        "path": str(path),
        "checks": checks,
        "minimum_sample_psnr_db": minimum_sample_psnr,
        "source_frames_compared": len(psnr_values),
    }


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    dataset = args.dataset.resolve()
    manifest = [
        line.strip()
        for line in args.episode_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    info = read_json(dataset / "meta/info.json")
    modality = read_json(dataset / "meta/modality.json")
    episodes = read_jsonl(dataset / "meta/episodes.jsonl")
    tasks = {row["task_index"]: row["task"] for row in read_jsonl(dataset / "meta/tasks.jsonl")}
    source_map = read_json(dataset / "meta/source_episode_map.json")["episodes"]
    errors: list[str] = []
    if not manifest or len(manifest) != len(set(manifest)):
        errors.append("empty or duplicate episode manifest")
    parquet_count = len(list(dataset.glob("data/*/*.parquet")))
    video_count = len(list(dataset.glob("videos/*/*/*.mp4")))
    if parquet_count != len(manifest) or video_count != 2 * len(manifest):
        errors.append("extra or missing Parquet/video files relative to manifest")
    maxima = {
        "state": 0.0,
        "action": 0.0,
        "timestamp_seconds": 0.0,
        "state_rot6d_orthogonality": 0.0,
        "action_rot6d_orthogonality": 0.0,
    }

    if not (len(manifest) == len(episodes) == len(source_map) == info["total_episodes"]):
        errors.append("manifest, metadata, source map, and info episode counts differ")
    if [row["source_episode"] for row in source_map] != manifest:
        errors.append("source episode map does not preserve manifest order")
    if info.get("fps") != 10.0:
        errors.append(f"dataset fps is {info.get('fps')}, expected 10.0")
    if modality.get("state", {}).get("right_eef") != {"start": 0, "end": 9}:
        errors.append("right EEF state slice is not [0:9]")
    if modality.get("action", {}).get("right_gripper") != {"start": 9, "end": 10}:
        errors.append("right gripper action slice is not [9:10]")
    for required_stats in ("stats.json", "relative_stats.json"):
        if not (dataset / "meta" / required_stats).is_file():
            errors.append(f"missing meta/{required_stats}")

    expected_global_index = 0
    video_jobs: list[tuple[Path, Path, str, int, float, float, bool]] = []
    identity = np.eye(2, dtype=np.float64)
    for target_index, (episode_meta, mapping) in enumerate(
        tqdm(zip(episodes, source_map), total=len(episodes), desc="Auditing Parquet")
    ):
        source_dir = raw_root / mapping["source_episode"]
        frames = read_jsonl(source_dir / "frames.jsonl")
        length = len(frames)
        chunk = target_index // int(info["chunks_size"])
        parquet = dataset / f"data/chunk-{chunk:03d}/episode_{target_index:06d}.parquet"
        dataframe = pd.read_parquet(parquet)
        if length != len(dataframe) or length != int(episode_meta["length"]):
            errors.append(f"episode {target_index}: raw/parquet/metadata length mismatch")
            continue

        state = np.stack(dataframe["observation.state"].to_numpy()).astype(np.float64)
        action = np.stack(dataframe["action"].to_numpy()).astype(np.float64)
        if state.shape != (length, 10) or action.shape != (length, 10):
            errors.append(f"episode {target_index}: state/action shape mismatch")
            continue
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            errors.append(f"episode {target_index}: state/action has non-finite values")

        current_pose = np.asarray([frame["right_ee_pose"] for frame in frames])
        delta_action = np.asarray([frame["action_right_7d"] for frame in frames])
        current_rotation = Rotation.from_quat(current_pose[:, 3:])
        target_rotation = Rotation.from_rotvec(delta_action[:, 3:6]) * current_rotation
        expected_state = np.concatenate(
            (
                current_pose[:, :3],
                rot6d(current_rotation),
                np.asarray([[frame["right_gripper"]["position"]] for frame in frames]),
            ),
            axis=1,
        )
        expected_action = np.concatenate(
            (
                current_pose[:, :3] + delta_action[:, :3],
                rot6d(target_rotation),
                delta_action[:, 6:7],
            ),
            axis=1,
        )
        state_error = max_abs(state, expected_state)
        action_error = max_abs(action, expected_action)
        maxima["state"] = max(maxima["state"], state_error)
        maxima["action"] = max(maxima["action"], action_error)
        if state_error > 2e-6 or action_error > 2e-6:
            errors.append(
                f"episode {target_index}: converted values differ from source "
                f"(state={state_error:.3g}, action={action_error:.3g})"
            )

        for label, values in (("state", state[:, 3:9]), ("action", action[:, 3:9])):
            rows = values.reshape(-1, 2, 3)
            gram = rows @ np.swapaxes(rows, 1, 2)
            error = max_abs(gram, identity)
            maxima[f"{label}_rot6d_orthogonality"] = max(
                maxima[f"{label}_rot6d_orthogonality"], error
            )
            if error > 2e-6:
                errors.append(f"episode {target_index}: {label} Rot6D rows are not orthonormal")

        expected_timestamp = np.arange(length, dtype=np.float64) / float(info["fps"])
        timestamp_error = max_abs(dataframe["timestamp"].to_numpy(), expected_timestamp)
        maxima["timestamp_seconds"] = max(maxima["timestamp_seconds"], timestamp_error)
        if timestamp_error > 1e-6:
            errors.append(f"episode {target_index}: timestamp grid mismatch")
        if not np.array_equal(dataframe["frame_index"].to_numpy(), np.arange(length)):
            errors.append(f"episode {target_index}: frame_index is not contiguous")
        if not np.array_equal(
            dataframe["index"].to_numpy(),
            np.arange(expected_global_index, expected_global_index + length),
        ):
            errors.append(f"episode {target_index}: global index is not contiguous")
        if not np.all(dataframe["episode_index"].to_numpy() == target_index):
            errors.append(f"episode {target_index}: episode_index column mismatch")
        task_indices = np.unique(dataframe["task_index"].to_numpy())
        if len(task_indices) != 1 or tasks.get(int(task_indices[0])) not in episode_meta["tasks"]:
            errors.append(f"episode {target_index}: task mapping mismatch")
        expected_global_index += length

        for camera in CAMERAS:
            video_jobs.append(
                (
                    dataset
                    / f"videos/chunk-{chunk:03d}/observation.images.{camera}"
                    / f"episode_{target_index:06d}.mp4",
                    source_dir / "images",
                    camera,
                    length,
                    float(info["fps"]),
                    args.minimum_psnr_db,
                    args.all_source_frames,
                )
            )

    if expected_global_index != int(info["total_frames"]):
        errors.append("audited frame total disagrees with info.total_frames")

    video_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(validate_video, *job): job[0] for job in video_jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Decoding videos"):
            path = futures[future]
            try:
                result = future.result()
                video_results.append(result)
                failed = [key for key, passed in result["checks"].items() if not passed]
                if failed:
                    errors.append(f"{path}: failed video checks {failed}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {type(exc).__name__}: {exc}")

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "dataset": str(dataset),
        "raw_root": str(raw_root),
        "episode_manifest": str(args.episode_manifest.resolve()),
        "counts": {
            "episodes": len(episodes),
            "frames": expected_global_index,
            "parquet": parquet_count,
            "videos": len(video_results),
        },
        "maximum_conversion_errors": maxima,
        "minimum_video_source_psnr_db": min(
            (row["minimum_sample_psnr_db"] for row in video_results), default=0.0
        ),
        "source_comparison_scope": "all_frames" if args.all_source_frames else "first_middle_last",
        "source_frames_compared": sum(row["source_frames_compared"] for row in video_results),
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(
        f"Conversion audit {report['status']}: episodes={len(episodes)} "
        f"frames={expected_global_index} videos={len(video_results)} errors={len(errors)}"
    )
    print(f"Report: {args.report}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
