#!/usr/bin/env python3
"""Verify official loading, H16 EEF round trips and train-only normalization.

Independent matrix arithmetic is compared to the official state/action routines.
No policy weights, GPU inference, source hashes or robot access are needed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import runpy
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from convert_xichong_right_single_grasp import write_json
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.state_action.action_chunking import EndEffectorActionChunk
from gr00t.data.state_action.pose import EndEffectorPose
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.types import ActionFormat, EmbodimentTag


def matrices(values):
    rows = np.asarray(values, dtype=np.float64).reshape(-1, 2, 3)
    a = rows[:, 0] / np.linalg.norm(rows[:, 0], axis=1, keepdims=True)
    b = rows[:, 1] - (a * rows[:, 1]).sum(axis=1, keepdims=True) * a
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], axis=1)


def relative_windows(state, action):
    """All complete H16 windows, with each window referenced to its first state."""
    sr, ar = matrices(state[:, 3:9]), matrices(action[:, 3:9])
    indices = np.arange(len(state) - 15)[:, None] + np.arange(16)
    ref = sr[: len(indices)]
    xyz = np.einsum("whi,wij->whj", action[indices, :3] - state[: len(indices), None, :3], ref)
    rotation = np.swapaxes(ref, -1, -2)[:, None] @ ar[indices]
    return np.concatenate([xyz, rotation[..., :2, :].reshape(-1, 16, 6)], axis=-1).astype(
        np.float32
    )


def check_statistics(actual, data):
    expected = {
        "mean": np.mean(data, axis=0),
        "std": np.std(data, axis=0),
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
        "q01": np.quantile(data, 0.01, axis=0),
        "q99": np.quantile(data, 0.99, axis=0),
    }
    maximum = 0.0
    for key, target in expected.items():
        value = np.asarray(actual[key])
        if value.shape != target.shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid statistic {key}: {value.shape} != {target.shape}")
        error = float(np.max(np.abs(value - target)))
        if error > 2e-5:
            raise ValueError(f"Statistic {key} differs by {error}")
        maximum = max(maximum, error)
    return maximum


_WORKER_LOADER = None
_WORKER_PROCESSOR = None
_WORKER_PROMPT = None


def initialize_worker(dataset, config, normalization_bounds):
    global _WORKER_LOADER, _WORKER_PROCESSOR, _WORKER_PROMPT
    _WORKER_LOADER = LeRobotEpisodeLoader(dataset, config, decoder_kwargs={"num_ffmpeg_threads": 1})
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    _WORKER_PROCESSOR = StateActionProcessor(
        {tag: config},
        {tag: _WORKER_LOADER.get_dataset_statistics()},
        use_percentiles=normalization_bounds == "percentile",
        clip_outliers=True,
        use_relative_action=True,
    )
    mapping = json.loads((Path(dataset) / "meta/source_episode_map.json").read_text())
    _WORKER_PROMPT = mapping["task_mapping"]["canonical_prompt"]


def verify_episode(index):
    loader, processor, prompt = _WORKER_LOADER, _WORKER_PROCESSOR, _WORKER_PROMPT
    tag, fmt = EmbodimentTag.NEW_EMBODIMENT.value, ActionFormat.XYZ_ROT6D
    max_relative = max_roundtrip = normalized_error = processor_roundtrip = 0.0
    count = {"official_video_frames": 0}
    clipping = {
        "eef_values": 0,
        "eef_outside_effective_bounds": 0,
        "jaw_values": 0,
        "jaw_outside_effective_bounds": 0,
        "max_jaw_change_from_production_normalization": 0.0,
        "max_xyz_change_from_production_normalization_m": 0.0,
    }
    frame = loader[index]
    state = np.concatenate(
        [np.stack(frame["state.right_eef"]), np.stack(frame["state.right_gripper"])], axis=1
    )
    action = np.concatenate(
        [np.stack(frame["action.right_eef"]), np.stack(frame["action.right_gripper"])],
        axis=1,
    )
    if len(frame) < 16 or not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Short or nonfinite episode")
    if set(frame["language.annotation.human.task_description"]) != {prompt}:
        raise ValueError("Official language mapping mismatch")
    raw_state = {"right_eef": state[:, :9], "right_gripper": state[:, 9:]}
    normalized_state = processor.apply_state(raw_state, tag)
    restored_state = processor.unapply_state(normalized_state, tag)
    clipping["state_values"] = state.size
    clipping["state_outside_effective_bounds"] = 0
    for key, values in raw_state.items():
        params = processor.norm_params[tag]["state"][key]
        clipping["state_outside_effective_bounds"] += int(
            ((values < params["min"]) | (values > params["max"])).sum()
        )
        if not np.isfinite(normalized_state[key]).all():
            raise ValueError("Official normalized state is nonfinite")
    clipping["max_state_xyz_change_from_normalization_m"] = float(
        np.max(np.linalg.norm(restored_state["right_eef"][:, :3] - state[:, :3], axis=1))
    )
    for camera in ("head_color", "hand_right"):
        for image in frame[f"video.{camera}"]:
            array = np.asarray(image)
            if array.shape != (480, 640, 3) or array.dtype != np.uint8:
                raise ValueError("Official video decoder shape/dtype mismatch")
        count["official_video_frames"] += len(frame)
    relative = relative_windows(state, action)
    for start, expected in enumerate(relative):
        ref = EndEffectorPose.from_action_format(state[start, :9], fmt)
        chunk = EndEffectorActionChunk.from_array(action[start : start + 16, :9], fmt)
        official = chunk.relative_chunking(reference_frame=ref)
        restored = official.to_absolute_chunking(reference_frame=ref).to(fmt)
        err = float(np.max(np.abs(official.to(fmt) - expected)))
        rt = float(np.max(np.abs(restored - action[start : start + 16, :9])))
        max_relative, max_roundtrip = max(max_relative, err), max(max_roundtrip, rt)
        if max(err, rt) > 2e-6:
            raise ValueError(f"EEF math mismatch: relative={err}, roundtrip={rt}")
        # The actual production processor: quantile clipping is not
        # invertible. Verify its result against independently implemented
        # normalization/denormalization, and report lost outlier extent.
        state_dict = {
            "right_eef": state[start : start + 1, :9],
            "right_gripper": state[start : start + 1, 9:],
        }
        action_dict = {
            "right_eef": action[start : start + 16, :9],
            "right_gripper": action[start : start + 16, 9:],
        }
        normalized = processor.apply_action(action_dict, tag, state_dict)
        decoded = processor.unapply_action(normalized, tag, state_dict)
        for key, values in normalized.items():
            if not np.isfinite(values).all():
                raise ValueError("Official normalized action is nonfinite")
            params = processor.norm_params[tag]["action"][key]
            lo, hi = params["min"], params["max"]
            source_values = official.to(fmt) if key == "right_eef" else action_dict[key]
            mask = ~np.isclose(hi, lo)
            ratio = np.zeros_like(source_values)
            np.divide(source_values - lo, hi - lo, out=ratio, where=mask)
            unclipped = np.where(mask, 2 * ratio - 1, 0)
            expected_normalized = np.clip(unclipped, -1, 1)
            norm_error = float(np.max(np.abs(values - expected_normalized)))
            normalized_error = max(normalized_error, norm_error)
            if norm_error > 2e-5:
                raise ValueError(f"Official normalize mismatch {key}: {norm_error}")
            expected_decoded = (expected_normalized + 1) / 2 * (hi - lo) + lo
            if key == "right_eef":
                absolute_rotation = ref.rotation_matrix @ matrices(expected_decoded[:, 3:])
                expected_decoded = np.concatenate(
                    [
                        expected_decoded[:, :3] @ ref.rotation_matrix.T + ref.translation,
                        absolute_rotation[:, :2, :].reshape(16, 6),
                    ],
                    axis=1,
                )
                loss = float(
                    np.max(np.linalg.norm(decoded[key][:, :3] - action_dict[key][:, :3], axis=1))
                )
                clipping["max_xyz_change_from_production_normalization_m"] = max(
                    clipping["max_xyz_change_from_production_normalization_m"], loss
                )
            else:
                loss = float(np.max(np.abs(decoded[key] - action_dict[key])))
                clipping["max_jaw_change_from_production_normalization"] = max(
                    clipping["max_jaw_change_from_production_normalization"], loss
                )
            error = float(np.max(np.abs(decoded[key] - expected_decoded)))
            processor_roundtrip = max(processor_roundtrip, error)
            if error > 2e-5:
                raise ValueError(f"Official processor round trip {key}: {error}")
            label = "eef" if key == "right_eef" else "jaw"
            clipping[label + "_values"] += values.size
            clipping[label + "_outside_effective_bounds"] += int((np.abs(unclipped) > 1).sum())

    return {
        "state": state.astype(np.float32),
        "action": action.astype(np.float32),
        "relative": relative,
        "clipping": clipping,
        "count": count,
        "max_relative": max_relative,
        "max_roundtrip": max_roundtrip,
        "normalized_error": normalized_error,
        "processor_roundtrip": processor_roundtrip,
    }


def verify(
    train: Path,
    heldout: Path,
    config_path: Path,
    workers: int = 4,
    normalization_bounds: str = "minmax",
) -> dict:
    if normalization_bounds not in ("minmax", "percentile"):
        raise ValueError("Unknown normalization bounds")
    runpy.run_path(str(config_path))
    config = MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
    if config["action"].delta_indices != list(range(16)):
        raise ValueError("Expected the H16 placement modality")
    all_state, all_action, all_relative, counts = [], [], [], {}
    max_relative, max_roundtrip = 0.0, 0.0
    processor_roundtrip = 0.0
    normalized_error = 0.0
    processor_counts = {}
    source_sets = []
    for split, dataset in (("train", train), ("heldout", heldout)):
        mapping = json.loads((dataset / "meta/source_episode_map.json").read_text())
        names = [r["source_episode"] for r in mapping["episodes"]]
        source_sets.append(set(names))
        if len(names) != len(set(names)):
            raise ValueError("Duplicate source episodes")
        loader = LeRobotEpisodeLoader(dataset, config, decoder_kwargs={"num_ffmpeg_threads": 1})
        clipping = {
            "eef_values": 0,
            "eef_outside_effective_bounds": 0,
            "jaw_values": 0,
            "jaw_outside_effective_bounds": 0,
            "max_jaw_change_from_production_normalization": 0.0,
            "max_xyz_change_from_production_normalization_m": 0.0,
        }
        count = {
            "episodes": len(loader),
            "frames": 0,
            "complete_h16_windows": 0,
            "official_video_frames": 0,
        }
        # Separate processes avoid the Python-heavy pose-object/GIL bottleneck.
        # map preserves episode order, so the initial reference remains reproducible.
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=initialize_worker,
            initargs=(dataset, config, normalization_bounds),
        ) as pool:
            for result in tqdm(
                pool.map(verify_episode, range(len(loader))),
                total=len(loader),
                desc=f"Official loader + math: {split}",
            ):
                state, action, relative = result["state"], result["action"], result["relative"]
                if split == "train":
                    all_state.append(state)
                    all_action.append(action)
                    all_relative.append(relative)
                max_relative = max(max_relative, result["max_relative"])
                max_roundtrip = max(max_roundtrip, result["max_roundtrip"])
                normalized_error = max(normalized_error, result["normalized_error"])
                processor_roundtrip = max(processor_roundtrip, result["processor_roundtrip"])
                for key, value in result["clipping"].items():
                    clipping[key] = (
                        max(clipping.get(key, 0), value)
                        if key.startswith("max_")
                        else clipping.get(key, 0) + value
                    )
                count["frames"] += len(state)
                count["complete_h16_windows"] += len(relative)
                count["official_video_frames"] += result["count"]["official_video_frames"]
        counts[split] = count
        processor_counts[split] = clipping
    if source_sets[0] & source_sets[1]:
        raise ValueError("Split leakage")
    stats = json.loads((train / "meta/stats.json").read_text())
    rel_stats = json.loads((train / "meta/relative_stats.json").read_text())
    stat_errors = {
        "state": check_statistics(stats["observation.state"], np.concatenate(all_state)),
        "action": check_statistics(stats["action"], np.concatenate(all_action)),
        "relative_right_eef": check_statistics(
            rel_stats["right_eef"], np.concatenate(all_relative)
        ),
    }
    for filename in ("stats.json", "relative_stats.json"):
        # Equality of small JSON metadata, not a source/model SHA scan.
        if json.loads((train / "meta" / filename).read_text()) != json.loads(
            (heldout / "meta" / filename).read_text()
        ):
            raise ValueError("Held-out normalization is not identical to training normalization")
    initial = np.stack([values[0] for values in all_state])
    initial_rotations = Rotation.from_matrix(matrices(initial[:, 3:9]))
    center = np.median(initial[:, :3], axis=0)
    angles = (initial_rotations.mean().inv() * initial_rotations).magnitude()
    scores = np.linalg.norm(initial[:, :3] - center, axis=1) / 0.02 + angles / np.deg2rad(10)
    reference_index = int(np.argmin(scores))
    train_map = json.loads((train / "meta/source_episode_map.json").read_text())["episodes"]
    initial_reference = {
        "source": "nearest actual training start to median XYZ and mean orientation; not a robot reset command",
        "source_episode": train_map[reference_index]["source_episode"],
        "converted_episode_index": reference_index,
        "pose_frame": "base_link_tf",
        "xyz_quaternion_xyzw": np.r_[
            initial[reference_index, :3], initial_rotations[reference_index].as_quat()
        ].tolist(),
        "right_gripper_native_radians": float(initial[reference_index, 9]),
        "train_start_xyz_min": initial[:, :3].min(axis=0).tolist(),
        "train_start_xyz_max": initial[:, :3].max(axis=0).tolist(),
        "train_start_xyz_median": center.tolist(),
        "robot_or_gdk_calibration_done": False,
    }
    write_json(train / "meta/initial_pose_reference.json", initial_reference)
    preprocessing = {
        "normalization_bounds": normalization_bounds,
        "use_percentiles": normalization_bounds == "percentile",
        "clip_outliers": True,
        "use_relative_action": True,
        "fitted_on": "train_only",
        "official_finetune_flag": "--use-percentiles"
        if normalization_bounds == "percentile"
        else "--no-use-percentiles",
    }
    for dataset in (train, heldout):
        write_json(dataset / "meta/training_preprocessing.json", preprocessing)
    return {
        "status": "PASS",
        "counts": counts,
        "max_relative_error": max_relative,
        "max_absolute_roundtrip_error": max_roundtrip,
        "stats_max_errors": stat_errors,
        "official_processor_vs_independent_denormalization_max_error": processor_roundtrip,
        "official_processor_vs_independent_normalization_max_error": normalized_error,
        "production_normalization": {
            "use_percentiles": normalization_bounds == "percentile",
            "clip_outliers": True,
            "use_relative_action": True,
        },
        "effective_action_bounds": {
            "right_eef": "official relative_stats per-horizon min/max (overrides absolute percentile params)",
            "right_gripper": "training q01/q99"
            if normalization_bounds == "percentile"
            else "training min/max",
        },
        "production_clipping_diagnostics": processor_counts,
        "heldout_used_to_fit_stats": False,
        "initial_pose_reference": initial_reference,
        "scope": "official_loader_and_eef_math_not_model_forward",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--modality-config", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--normalization-bounds", choices=("minmax", "percentile"), default="minmax")
    args = p.parse_args()
    try:
        report = verify(
            args.train, args.heldout, args.modality_config, args.workers, args.normalization_bounds
        )
    except Exception as exc:
        write_json(args.report, {"status": "FAIL", "error": str(exc)})
        raise
    write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
