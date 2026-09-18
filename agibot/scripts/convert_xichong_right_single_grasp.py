#!/usr/bin/env python3
"""Validate and convert the G2A raw right-arm dataset to GR00T LeRobot v2.

The raw action is a base-frame/world-frame one-step delta:
    [delta_xyz_world, delta_rotvec_world, next_gripper_absolute]

The converter reconstructs the absolute target EEF pose and stores both state
and action as absolute XYZ+Rot6D. GR00T's RELATIVE EEF processor then computes
its expected local-frame relative action without double-differencing the raw
delta.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm


CAMERAS = ("head_color", "hand_right")
IMAGE_SIZE = (640, 480)
STATE_DIM = 10
ACTION_DIM = 10


@dataclass
class EpisodeValidation:
    source_episode: str
    source_index: int
    valid: bool
    length: int = 0
    prompt: str = ""
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root / "agibot/data/xichong_right_single_grasp",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "agibot/gr00t_data/xichong_right_single_grasp",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "agibot/reports/xichong_right_single_grasp_validation.json",
    )
    parser.add_argument(
        "--episode-manifest",
        type=Path,
        default=None,
        help="Optional newline-delimited source episode list; only these episodes are used",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--preset",
        default="fast",
        help="libx264 preset used for H.264 encoding",
    )
    parser.add_argument(
        "--skip-image-decode",
        action="store_true",
        help="Only check image presence/naming; default fully decodes every selected JPEG",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Write the validation report without creating the GR00T dataset",
    )
    parser.add_argument(
        "--quarantine-invalid",
        action="store_true",
        help="Move invalid accepted episode directories into .quality_quarantine",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional development limit; omitted means all accepted episodes",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_frames(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
    return frames


def finite_vector(value: Any, expected_dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (expected_dim,):
        raise ValueError(f"{name} shape {array.shape}, expected ({expected_dim},)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return array


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise ValueError(f"format={image.format}, expected JPEG")
        if image.mode != "RGB":
            raise ValueError(f"mode={image.mode}, expected RGB")
        if image.size != IMAGE_SIZE:
            raise ValueError(f"size={image.size}, expected {IMAGE_SIZE}")
        image.load()


def validate_episode(episode_dir: Path, decode_images: bool) -> EpisodeValidation:
    try:
        source_index = int(episode_dir.name.removeprefix("episode_"))
    except ValueError:
        return EpisodeValidation(episode_dir.name, -1, False, errors=["invalid episode name"])

    result = EpisodeValidation(episode_dir.name, source_index, False)
    errors = result.errors
    assert errors is not None

    required = (
        episode_dir / "arrays.npz",
        episode_dir / "frames.jsonl",
        episode_dir / "meta_info.json",
        episode_dir / "quality_report.json",
        episode_dir / "images",
    )
    for path in required:
        if not path.exists():
            errors.append(f"missing {path.name}")
    if errors:
        return result

    try:
        meta = load_json(episode_dir / "meta_info.json")
        quality = load_json(episode_dir / "quality_report.json")
        frames = load_frames(episode_dir / "frames.jsonl")
        result.length = len(frames)
        result.prompt = json.loads(meta["text"])["description"]

        if meta.get("annotations", {}).get("success") != "y":
            errors.append("success label is not y")
        if meta.get("annotations", {}).get("failure_reason") != "none":
            errors.append("failure_reason is not none")
        if not meta.get("quality", {}).get("data_validate", False):
            errors.append("meta quality.data_validate is false")
        if meta.get("quality", {}).get("errors"):
            errors.append(f"meta quality errors: {meta['quality']['errors']}")
        if not quality.get("contract", {}).get("ok", False):
            errors.append("quality contract is not ok")
        if not quality.get("xichong_terminal", {}).get("ok", False):
            errors.append("xichong terminal quality is not ok")
        if meta.get("selected", {}).get("pose_frame") != "base_link_tf":
            errors.append("pose_frame is not base_link_tf")
        if meta.get("selected", {}).get("action_mode") != "next_delta":
            errors.append("action_mode is not next_delta")
        if not result.prompt:
            errors.append("empty prompt")

        with np.load(episode_dir / "arrays.npz") as arrays:
            expected_shapes = {
                "states": (result.length, 16),
                "actions": (result.length, 14),
                "ee_poses": (result.length, 14),
                "grippers": (result.length, 2),
                "timestamps_monotonic": (result.length,),
            }
            for key, shape in expected_shapes.items():
                if key not in arrays:
                    errors.append(f"arrays.npz missing {key}")
                    continue
                if arrays[key].shape != shape:
                    errors.append(f"{key} shape={arrays[key].shape}, expected={shape}")
                elif not np.isfinite(arrays[key]).all():
                    errors.append(f"{key} contains NaN/Inf")
            if not errors and result.length > 1:
                if not np.all(np.diff(arrays["timestamps_monotonic"]) > 0):
                    errors.append("arrays timestamps are not strictly increasing")

        previous_timestamp = -math.inf
        for expected_frame_index, frame in enumerate(frames):
            prefix = f"frame {expected_frame_index}"
            if frame.get("frame_index") != expected_frame_index:
                errors.append(f"{prefix}: non-contiguous frame_index")
                break
            timestamp = float(frame["timestamp_monotonic"])
            if not math.isfinite(timestamp) or timestamp <= previous_timestamp:
                errors.append(f"{prefix}: timestamp is not strictly increasing")
                break
            previous_timestamp = timestamp

            current_pose = finite_vector(frame["right_ee_pose"], 7, f"{prefix} right_ee_pose")
            raw_action = finite_vector(frame["action_right_7d"], 7, f"{prefix} action")
            next_pose = finite_vector(frame["next_right_ee_pose"], 7, f"{prefix} next pose")
            next_gripper = float(frame["next_right_gripper"])
            if not math.isfinite(next_gripper):
                errors.append(f"{prefix}: next gripper is not finite")
                break
            if abs(np.linalg.norm(current_pose[3:]) - 1.0) > 1e-5:
                errors.append(f"{prefix}: current quaternion is not normalized")
                break
            if abs(np.linalg.norm(next_pose[3:]) - 1.0) > 1e-5:
                errors.append(f"{prefix}: next quaternion is not normalized")
                break

            target_xyz = current_pose[:3] + raw_action[:3]
            if not np.allclose(target_xyz, next_pose[:3], atol=2e-5, rtol=0):
                errors.append(f"{prefix}: delta translation does not reconstruct next pose")
                break
            current_rotation = Rotation.from_quat(current_pose[3:])
            target_rotation = Rotation.from_rotvec(raw_action[3:6]) * current_rotation
            next_rotation = Rotation.from_quat(next_pose[3:])
            angular_error = (target_rotation.inv() * next_rotation).magnitude()
            if angular_error > 2e-5:
                errors.append(
                    f"{prefix}: delta rotation does not reconstruct next pose ({angular_error})"
                )
                break
            if not math.isclose(raw_action[6], next_gripper, abs_tol=2e-5, rel_tol=0):
                errors.append(f"{prefix}: action gripper does not equal next gripper")
                break

            image_map = frame.get("images", {})
            for camera in CAMERAS:
                expected_relpath = f"images/{camera}_{expected_frame_index:06d}.jpg"
                if image_map.get(camera) != expected_relpath:
                    errors.append(f"{prefix}: bad {camera} image mapping")
                    break
                image_path = episode_dir / expected_relpath
                if not image_path.is_file():
                    errors.append(f"{prefix}: missing {camera} image")
                    break
                if decode_images:
                    try:
                        validate_image(image_path)
                    except Exception as exc:  # noqa: BLE001 - report malformed source data
                        errors.append(f"{prefix}: invalid {camera} image: {exc}")
                        break
            if errors:
                break

        result.valid = not errors
    except Exception as exc:  # noqa: BLE001 - validation must report every bad episode
        errors.append(f"{type(exc).__name__}: {exc}")
        result.valid = False
    return result


def rot6d(rotation: Rotation) -> np.ndarray:
    """Match GR00T EndEffectorPose: first two rotation-matrix rows, flattened."""
    return rotation.as_matrix()[:2, :].reshape(6).astype(np.float32)


def convert_low_dimensional_data(
    source_dir: Path,
    parquet_path: Path,
    target_episode_index: int,
    global_start_index: int,
    task_index: int,
    fps: float,
) -> int:
    frames = load_frames(source_dir / "frames.jsonl")
    states = np.empty((len(frames), STATE_DIM), dtype=np.float32)
    actions = np.empty((len(frames), ACTION_DIM), dtype=np.float32)
    # LeRobot videos are encoded on a fixed-rate frame grid.  Keep the tabular
    # timestamps on that exact same grid instead of carrying over acquisition
    # jitter from the robot clock.
    timestamps = np.arange(len(frames), dtype=np.float32) / np.float32(fps)

    for i, frame in enumerate(frames):
        current_pose = np.asarray(frame["right_ee_pose"], dtype=np.float64)
        raw_action = np.asarray(frame["action_right_7d"], dtype=np.float64)

        current_rotation = Rotation.from_quat(current_pose[3:])
        target_xyz = current_pose[:3] + raw_action[:3]
        target_rotation = Rotation.from_rotvec(raw_action[3:6]) * current_rotation

        states[i, :3] = current_pose[:3]
        states[i, 3:9] = rot6d(current_rotation)
        states[i, 9] = float(frame["right_gripper"]["position"])

        actions[i, :3] = target_xyz
        actions[i, 3:9] = rot6d(target_rotation)
        actions[i, 9] = raw_action[6]

    dataframe = pd.DataFrame(
        {
            "action": [row for row in actions],
            "observation.state": [row for row in states],
            "timestamp": timestamps,
            "frame_index": np.arange(len(frames), dtype=np.int64),
            "episode_index": np.full(len(frames), target_episode_index, dtype=np.int64),
            "index": np.arange(
                global_start_index,
                global_start_index + len(frames),
                dtype=np.int64,
            ),
            "task_index": np.full(len(frames), task_index, dtype=np.int64),
        }
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = parquet_path.with_suffix(".tmp.parquet")
    dataframe.to_parquet(temporary_path, index=False)
    temporary_path.replace(parquet_path)
    return len(frames)


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    output = subprocess.check_output(command, text=True)
    return json.loads(output)["streams"][0]


def encode_video(
    source_dir: Path,
    output_path: Path,
    camera: str,
    length: int,
    fps: float,
    crf: int,
    preset: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        stream = probe_video(output_path)
        if (
            stream.get("codec_name") == "h264"
            and int(stream["width"]) == IMAGE_SIZE[0]
            and int(stream["height"]) == IMAGE_SIZE[1]
            and int(stream.get("nb_frames", -1)) == length
        ):
            return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.mp4")
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-start_number",
        "0",
        "-i",
        str(source_dir / "images" / f"{camera}_%06d.jpg"),
        "-frames:v",
        str(length),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]
    subprocess.run(command, check=True)
    stream = probe_video(temporary_path)
    if stream.get("codec_name") != "h264":
        raise RuntimeError(f"{temporary_path}: codec is not h264")
    if int(stream["width"]) != IMAGE_SIZE[0] or int(stream["height"]) != IMAGE_SIZE[1]:
        raise RuntimeError(f"{temporary_path}: unexpected video size")
    if int(stream.get("nb_frames", -1)) != length:
        raise RuntimeError(
            f"{temporary_path}: frame count {stream.get('nb_frames')} != expected {length}"
        )
    temporary_path.replace(output_path)


def convert_episode(
    source_dir: Path,
    output_dir: Path,
    target_episode_index: int,
    global_start_index: int,
    task_index: int,
    length: int,
    fps: float,
    crf: int,
    preset: str,
    overwrite: bool,
) -> dict[str, Any]:
    chunk_index = target_episode_index // 1000
    parquet_path = (
        output_dir
        / f"data/chunk-{chunk_index:03d}/episode_{target_episode_index:06d}.parquet"
    )
    converted_length = convert_low_dimensional_data(
        source_dir,
        parquet_path,
        target_episode_index,
        global_start_index,
        task_index,
        fps,
    )
    if converted_length != length:
        raise RuntimeError(f"converted length {converted_length} != validated length {length}")

    for camera in CAMERAS:
        video_path = (
            output_dir
            / f"videos/chunk-{chunk_index:03d}"
            / f"observation.images.{camera}"
            / f"episode_{target_episode_index:06d}.mp4"
        )
        encode_video(
            source_dir,
            video_path,
            camera,
            length,
            fps,
            crf,
            preset,
            overwrite,
        )
    return {
        "source_episode": source_dir.name,
        "target_episode_index": target_episode_index,
        "length": length,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temporary_path.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def write_metadata(
    output_dir: Path,
    valid_results: list[EpisodeValidation],
    task_to_index: dict[str, int],
    fps: float,
) -> None:
    total_frames = sum(result.length for result in valid_results)
    state_names = [
        "right_eef.x",
        "right_eef.y",
        "right_eef.z",
        "right_eef.rot6d.r00",
        "right_eef.rot6d.r01",
        "right_eef.rot6d.r02",
        "right_eef.rot6d.r10",
        "right_eef.rot6d.r11",
        "right_eef.rot6d.r12",
        "right_gripper.position",
    ]
    video_feature = {
        "dtype": "video",
        "shape": [IMAGE_SIZE[1], IMAGE_SIZE[0], 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": IMAGE_SIZE[1],
            "video.width": IMAGE_SIZE[0],
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    scalar_features = {
        key: {"dtype": dtype, "shape": [1], "names": None}
        for key, dtype in {
            "timestamp": "float32",
            "frame_index": "int64",
            "episode_index": "int64",
            "index": "int64",
            "task_index": "int64",
        }.items()
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "agibot_g2a_right_arm_eef",
        "total_episodes": len(valid_results),
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(valid_results)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": state_names},
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": state_names,
            },
            "observation.images.head_color": dict(video_feature),
            "observation.images.hand_right": dict(video_feature),
            **scalar_features,
        },
        "total_chunks": (len(valid_results) + 999) // 1000,
        "total_videos": len(valid_results) * len(CAMERAS),
    }
    modality = {
        "state": {
            "right_eef": {"start": 0, "end": 9},
            "right_gripper": {"start": 9, "end": 10},
        },
        "action": {
            "right_eef": {"start": 0, "end": 9},
            "right_gripper": {"start": 9, "end": 10},
        },
        "video": {
            "head_color": {"original_key": "observation.images.head_color"},
            "hand_right": {"original_key": "observation.images.hand_right"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }
    tasks = [
        {"task_index": task_index, "task": task}
        for task, task_index in sorted(task_to_index.items(), key=lambda item: item[1])
    ]
    episodes = [
        {
            "episode_index": target_index,
            "tasks": [result.prompt],
            "length": result.length,
        }
        for target_index, result in enumerate(valid_results)
    ]
    write_json(output_dir / "meta/info.json", info)
    write_json(output_dir / "meta/modality.json", modality)
    write_jsonl(output_dir / "meta/tasks.jsonl", tasks)
    write_jsonl(output_dir / "meta/episodes.jsonl", episodes)


def quarantine_invalid(source_root: Path, invalid: list[EpisodeValidation]) -> list[dict[str, str]]:
    moved: list[dict[str, str]] = []
    quarantine_root = source_root / ".quality_quarantine"
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    for result in invalid:
        source = source_root / result.source_episode
        if not source.exists():
            continue
        destination = quarantine_root / f"{result.source_episode}-gr00t-invalid-{timestamp}"
        suffix = 1
        while destination.exists():
            destination = quarantine_root / (
                f"{result.source_episode}-gr00t-invalid-{timestamp}-{suffix:02d}"
            )
            suffix += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved.append({"source": str(source), "destination": str(destination)})
    return moved


def main() -> int:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    report_path = args.report.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    manifest_order: dict[str, int] | None = None
    if args.episode_manifest is not None:
        episode_manifest = args.episode_manifest.resolve()
        episode_names = [
            line.strip()
            for line in episode_manifest.read_text().splitlines()
            if line.strip()
        ]
        if len(episode_names) != len(set(episode_names)):
            raise ValueError(f"duplicate entries in {episode_manifest}")
        if any(
            not name.startswith("episode_") or len(name) != len("episode_000000")
            for name in episode_names
        ):
            raise ValueError(f"invalid episode name in {episode_manifest}")
        episode_dirs = [source_root / name for name in episode_names]
        missing = [path.name for path in episode_dirs if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"manifest episode directories missing: {missing[:5]}")
        manifest_order = {name: index for index, name in enumerate(episode_names)}
    else:
        episode_dirs = sorted(
            path
            for path in source_root.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]")
            if path.is_dir()
        )
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]
    if not episode_dirs:
        raise RuntimeError(f"no accepted episode directories found in {source_root}")

    results: list[EpisodeValidation] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                validate_episode,
                episode_dir,
                not args.skip_image_decode,
            ): episode_dir
            for episode_dir in episode_dirs
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Validating accepted episodes",
        ):
            results.append(future.result())
    if manifest_order is None:
        results.sort(key=lambda result: result.source_index)
    else:
        results.sort(key=lambda result: manifest_order[result.source_episode])
    valid = [result for result in results if result.valid]
    invalid = [result for result in results if not result.valid]

    moved: list[dict[str, str]] = []
    if invalid and args.quarantine_invalid:
        moved = quarantine_invalid(source_root, invalid)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": str(source_root),
        "episode_manifest": (
            str(args.episode_manifest.resolve()) if args.episode_manifest else None
        ),
        "output": str(output_root),
        "selected_cameras": list(CAMERAS),
        "image_decode_validation": not args.skip_image_decode,
        "checked_episodes": len(results),
        "valid_episodes": len(valid),
        "invalid_episodes": len(invalid),
        "valid_frames": sum(result.length for result in valid),
        "invalid": [asdict(result) for result in invalid],
        "quarantined": moved,
        "episodes": [asdict(result) for result in results],
    }
    write_json(report_path, report)
    print(
        f"Validation: checked={len(results)} valid={len(valid)} "
        f"invalid={len(invalid)} valid_frames={report['valid_frames']}"
    )
    print(f"Validation report: {report_path}")

    if args.validate_only:
        return 0 if not invalid else 2
    if not valid:
        raise RuntimeError("no valid episodes to convert")

    unique_prompts = sorted({result.prompt for result in valid})
    task_to_index = {prompt: index for index, prompt in enumerate(unique_prompts)}
    global_starts: list[int] = []
    running_index = 0
    for result in valid:
        global_starts.append(running_index)
        running_index += result.length

    conversions: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {}
        for target_index, (result, global_start) in enumerate(zip(valid, global_starts)):
            future = executor.submit(
                convert_episode,
                source_root / result.source_episode,
                output_root,
                target_index,
                global_start,
                task_to_index[result.prompt],
                result.length,
                args.fps,
                args.crf,
                args.preset,
                args.overwrite,
            )
            futures[future] = result.source_episode
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Converting GR00T episodes",
        ):
            try:
                conversions.append(future.result())
            except Exception as exc:  # noqa: BLE001
                source_episode = futures[future]
                print(f"Conversion failed for {source_episode}: {exc}", file=sys.stderr)
                raise
    conversions.sort(key=lambda row: row["target_episode_index"])
    write_metadata(output_root, valid, task_to_index, args.fps)
    write_json(
        output_root / "meta/source_episode_map.json",
        {
            "source_root": str(source_root),
            "pose_frame": "base_link_tf",
            "raw_action_mode": "next_delta_world_frame",
            "converted_action_mode": "absolute_target_xyz_rot6d",
            "episodes": conversions,
        },
    )
    print(f"Converted dataset: {output_root}")
    print(f"Episodes={len(valid)} frames={running_index} tasks={len(task_to_index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
