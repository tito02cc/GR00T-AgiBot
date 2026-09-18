#!/usr/bin/env python3
"""Deterministic open-loop evaluation for Xichong right-arm GR00T checkpoints.

The evaluator re-plans every 16 steps from recorded observations, matching the
intended deployment execution horizon.  It reports task-space metrics instead
of relying only on a scale-mixed 10D MSE: XYZ error, rotation geodesic error,
gripper error, closure timing, and PyTorch inference latency.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.utils.determinism import seed_everything


GRIPPER_TRAINING_RANGE = 0.785
GRIPPER_G2_RANGE_MM = 120.0
GRIPPER_CLOSED_THRESHOLD = -0.55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--heldout-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-episode-ids", type=int, nargs="+", default=[0, 75, 150, 225, 299])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument(
        "--skip-heldout",
        action="store_true",
        help="Evaluate only the requested training episodes (inference smoke/fit check).",
    )
    return parser.parse_args()


def read_source_map(dataset: Path) -> dict[int, int]:
    path = dataset / "meta/source_episode_map.json"
    rows = json.loads(path.read_text())["episodes"]
    return {
        int(row["subset_episode_index"]): int(row["full_dataset_episode_index"])
        for row in rows
    }


def rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64).reshape(-1, 2, 3)
    row0_norm = np.linalg.norm(rows[:, 0], axis=1, keepdims=True)
    if np.any(row0_norm < 1e-8):
        raise ValueError("Rot6D first row has near-zero norm")
    row0 = rows[:, 0] / row0_norm
    residual = rows[:, 1] - np.sum(row0 * rows[:, 1], axis=1, keepdims=True) * row0
    row1_norm = np.linalg.norm(residual, axis=1, keepdims=True)
    if np.any(row1_norm < 1e-8):
        raise ValueError("Rot6D rows are near-collinear")
    row1 = residual / row1_norm
    row2 = np.cross(row0, row1)
    return np.stack((row0, row1, row2), axis=1)


def rotation_errors_deg(gt_rot6d: np.ndarray, pred_rot6d: np.ndarray) -> np.ndarray:
    gt = rot6d_to_matrix(gt_rot6d)
    pred = rot6d_to_matrix(pred_rot6d)
    relative = pred @ np.swapaxes(gt, 1, 2)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def extract_column(trajectory: Any, key: str) -> np.ndarray:
    return np.vstack([np.asarray(value) for value in trajectory[key]])


def prepare_observation(
    trajectory: Any,
    step: int,
    loader: LeRobotEpisodeLoader,
    embodiment: EmbodimentTag,
) -> dict[str, Any]:
    modalities = deepcopy(loader.modality_configs)
    modalities.pop("action")
    point = extract_step_data(trajectory, step, modalities, embodiment)
    observation: dict[str, Any] = {}
    for key, value in point.states.items():
        observation[f"state.{key}"] = value
    for key, value in point.images.items():
        observation[f"video.{key}"] = np.asarray(value)
    for key in loader.modality_configs["language"].modality_keys:
        observation[key] = point.text
    return parse_observation_gr00t(observation, loader.modality_configs)


def first_closed_index(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values).reshape(-1) >= GRIPPER_CLOSED_THRESHOLD)
    return int(indices[0]) if len(indices) else None


def threshold_transitions(values: np.ndarray) -> tuple[int, int]:
    """Return closed onsets and reopenings for the configured gripper threshold."""
    closed = np.asarray(values).reshape(-1) >= GRIPPER_CLOSED_THRESHOLD
    previous = np.concatenate(([False], closed[:-1]))
    onsets = int(np.count_nonzero(closed & ~previous))
    reopenings = int(np.count_nonzero(~closed & previous))
    return onsets, reopenings


def scalar_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def compute_metrics(
    episodes: list[dict[str, Any]], inference_times: list[float]
) -> dict[str, Any]:
    gt_eef = np.concatenate([row["gt_eef"] for row in episodes])
    pred_eef = np.concatenate([row["pred_eef"] for row in episodes])
    gt_gripper = np.concatenate([row["gt_gripper"] for row in episodes])
    pred_gripper = np.concatenate([row["pred_gripper"] for row in episodes])
    gt_all = np.concatenate((gt_eef, gt_gripper), axis=1)
    pred_all = np.concatenate((pred_eef, pred_gripper), axis=1)

    xyz_error_m = np.linalg.norm(pred_eef[:, :3] - gt_eef[:, :3], axis=1)
    rotation_error_deg = rotation_errors_deg(gt_eef[:, 3:9], pred_eef[:, 3:9])
    gripper_error = np.abs(pred_gripper[:, 0] - gt_gripper[:, 0])
    gripper_error_mm = gripper_error * GRIPPER_G2_RANGE_MM / GRIPPER_TRAINING_RANGE

    closure_errors: list[int] = []
    closure_signed_errors: list[int] = []
    closure_misses = 0
    episode_metrics: list[dict[str, Any]] = []
    for row in episodes:
        episode_xyz_error_m = np.linalg.norm(
            row["pred_eef"][:, :3] - row["gt_eef"][:, :3], axis=1
        )
        episode_rotation_error_deg = rotation_errors_deg(
            row["gt_eef"][:, 3:9], row["pred_eef"][:, 3:9]
        )
        episode_gripper_error = np.abs(
            row["pred_gripper"][:, 0] - row["gt_gripper"][:, 0]
        )
        gt_close = first_closed_index(row["gt_gripper"])
        pred_close = first_closed_index(row["pred_gripper"])
        gt_onsets, gt_reopenings = threshold_transitions(row["gt_gripper"])
        pred_onsets, pred_reopenings = threshold_transitions(row["pred_gripper"])
        timing_signed = None
        timing_abs = None
        if gt_close is None or pred_close is None:
            closure_misses += 1
        else:
            timing_signed = pred_close - gt_close
            timing_abs = abs(timing_signed)
            closure_signed_errors.append(timing_signed)
            closure_errors.append(timing_abs)
        episode_metrics.append(
            {
                "subset_episode_id": row["subset_episode_id"],
                "full_episode_id": row["full_episode_id"],
                "action_steps": int(len(row["gt_eef"])),
                "xyz_l2_mm": {
                    key: value * 1000.0
                    for key, value in scalar_summary(episode_xyz_error_m).items()
                },
                "rotation_geodesic_deg": scalar_summary(episode_rotation_error_deg),
                "gripper_abs_error_training_units": scalar_summary(
                    episode_gripper_error
                ),
                "gt_first_closed_step": gt_close,
                "pred_first_closed_step": pred_close,
                "closure_timing_signed_error_steps": timing_signed,
                "closure_timing_abs_error_steps": timing_abs,
                "gt_closed_onsets": gt_onsets,
                "gt_reopenings": gt_reopenings,
                "pred_closed_onsets": pred_onsets,
                "pred_reopenings": pred_reopenings,
            }
        )

    timed = np.asarray(inference_times[1:] if len(inference_times) > 1 else inference_times)
    result: dict[str, Any] = {
        "episodes": len(episodes),
        "action_steps": int(len(gt_all)),
        "mse_10d": float(np.mean((pred_all - gt_all) ** 2)),
        "mae_10d": float(np.mean(np.abs(pred_all - gt_all))),
        "xyz_l2_mm": {key: value * 1000.0 for key, value in scalar_summary(xyz_error_m).items()},
        "rotation_geodesic_deg": scalar_summary(rotation_error_deg),
        "gripper_abs_error_training_units": scalar_summary(gripper_error),
        "gripper_abs_error_equivalent_mm": scalar_summary(gripper_error_mm),
        "closure_threshold": GRIPPER_CLOSED_THRESHOLD,
        "closure_timing_abs_error_steps_mean": (
            float(np.mean(closure_errors)) if closure_errors else None
        ),
        "closure_timing_abs_error_steps": (
            scalar_summary(np.asarray(closure_errors)) if closure_errors else None
        ),
        "closure_timing_signed_error_steps": (
            scalar_summary(np.asarray(closure_signed_errors))
            if closure_signed_errors
            else None
        ),
        "closure_timing_misses": closure_misses,
        "episode_metrics": episode_metrics,
        "inference_calls": len(inference_times),
        "inference_latency_seconds": scalar_summary(timed),
    }
    return result


def plot_episode(row: dict[str, Any], output: Path) -> None:
    gt_eef = row["gt_eef"]
    pred_eef = row["pred_eef"]
    gt_gripper = row["gt_gripper"][:, 0]
    pred_gripper = row["pred_gripper"][:, 0]
    xyz_error_mm = np.linalg.norm(pred_eef[:, :3] - gt_eef[:, :3], axis=1) * 1000.0
    rotation_error = rotation_errors_deg(gt_eef[:, 3:9], pred_eef[:, 3:9])
    x = np.arange(len(gt_eef))

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    labels = ("X", "Y", "Z")
    for dim, label in enumerate(labels):
        axes[0].plot(x, gt_eef[:, dim], label=f"GT {label}")
        axes[0].plot(x, pred_eef[:, dim], "--", label=f"Pred {label}")
    axes[0].set_ylabel("EEF XYZ (m)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(x, xyz_error_mm)
    axes[1].set_ylabel("XYZ L2 error (mm)")
    axes[2].plot(x, rotation_error)
    axes[2].set_ylabel("Rotation error (deg)")
    axes[3].plot(x, gt_gripper, label="GT")
    axes[3].plot(x, pred_gripper, "--", label="Pred")
    axes[3].axhline(GRIPPER_CLOSED_THRESHOLD, color="black", alpha=0.4)
    axes[3].set_ylabel("Gripper")
    axes[3].set_xlabel("Recorded action step")
    axes[3].legend()
    fig.suptitle(
        f"{row['split']} subset={row['subset_episode_id']} full={row['full_episode_id']} seed={row['seed']}"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)


def evaluate_split(
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    split: str,
    episode_ids: list[int],
    source_map: dict[int, int],
    seed: int,
    steps: int,
    execution_horizon: int,
    plot_dir: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    embodiment = EmbodimentTag.NEW_EMBODIMENT
    episode_results: list[dict[str, Any]] = []
    inference_times: list[float] = []

    for episode_id in episode_ids:
        full_episode_id = source_map[episode_id]
        episode_seed = seed + full_episode_id * 1009
        seed_everything(episode_seed)
        trajectory = loader[episode_id]
        actual_steps = min(steps, len(trajectory))
        predicted_eef: list[np.ndarray] = []
        predicted_gripper: list[np.ndarray] = []

        for step in range(0, actual_steps, execution_horizon):
            observation = prepare_observation(trajectory, step, loader, embodiment)
            torch.cuda.synchronize()
            started = time.perf_counter()
            action, _ = policy.get_action(observation)
            torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - started)
            predicted_eef.append(np.asarray(action["right_eef"])[0, :execution_horizon])
            predicted_gripper.append(np.asarray(action["right_gripper"])[0, :execution_horizon])

        pred_eef = np.concatenate(predicted_eef, axis=0)[:actual_steps].astype(np.float64)
        pred_gripper = np.concatenate(predicted_gripper, axis=0)[:actual_steps].astype(np.float64)
        gt_eef = extract_column(trajectory, "action.right_eef")[:actual_steps].astype(np.float64)
        gt_gripper = extract_column(trajectory, "action.right_gripper")[:actual_steps].astype(np.float64)
        if pred_eef.shape != gt_eef.shape or pred_gripper.shape != gt_gripper.shape:
            raise ValueError(
                f"shape mismatch for {split}/{episode_id}: "
                f"eef {pred_eef.shape}/{gt_eef.shape}, gripper {pred_gripper.shape}/{gt_gripper.shape}"
            )
        for name, values in (
            ("pred_eef", pred_eef),
            ("pred_gripper", pred_gripper),
            ("gt_eef", gt_eef),
            ("gt_gripper", gt_gripper),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite {name} for {split}/{episode_id}")

        row = {
            "split": split,
            "subset_episode_id": episode_id,
            "full_episode_id": full_episode_id,
            "seed": seed,
            "gt_eef": gt_eef,
            "pred_eef": pred_eef,
            "gt_gripper": gt_gripper,
            "pred_gripper": pred_gripper,
        }
        episode_results.append(row)
        if plot_dir is not None:
            plot_episode(row, plot_dir / f"{split}_full_{full_episode_id:06d}.png")
        print(
            f"PASS split={split} seed={seed} subset={episode_id} "
            f"full={full_episode_id} steps={actual_steps}",
            flush=True,
        )

    metrics = compute_metrics(episode_results, inference_times)
    metrics["episode_ids"] = episode_ids
    metrics["full_episode_ids"] = [source_map[index] for index in episode_ids]
    return metrics, episode_results


def summarize_seeds(runs: list[dict[str, Any]], split: str) -> dict[str, Any]:
    paths = {
        "mse_10d": ("mse_10d",),
        "mae_10d": ("mae_10d",),
        "xyz_l2_mean_mm": ("xyz_l2_mm", "mean"),
        "xyz_l2_p90_mm": ("xyz_l2_mm", "p90"),
        "rotation_mean_deg": ("rotation_geodesic_deg", "mean"),
        "rotation_p90_deg": ("rotation_geodesic_deg", "p90"),
        "gripper_mae_training_units": ("gripper_abs_error_training_units", "mean"),
        "gripper_mae_equivalent_mm": ("gripper_abs_error_equivalent_mm", "mean"),
        "closure_timing_abs_error_steps_mean": ("closure_timing_abs_error_steps_mean",),
        "inference_latency_mean_seconds": ("inference_latency_seconds", "mean"),
        "inference_latency_p90_seconds": ("inference_latency_seconds", "p90"),
    }
    summary: dict[str, Any] = {}
    for output_key, path in paths.items():
        values: list[float] = []
        for run in runs:
            value: Any = run["splits"][split]
            for key in path:
                value = value[key]
            if value is not None:
                values.append(float(value))
        summary[output_key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
    summary["closure_timing_misses"] = [
        run["splits"][split]["closure_timing_misses"] for run in runs
    ]
    return summary


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.execution_horizon <= 0 or args.steps <= 0:
        raise ValueError("steps and execution horizon must be positive")

    match = re.search(r"checkpoint-(\d+)", str(checkpoint))
    global_step = int(match.group(1)) if match else None
    started = time.perf_counter()
    policy = Gr00tPolicy("new_embodiment", str(checkpoint), device="cuda")
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    model_load_seconds = time.perf_counter() - started
    modality = policy.get_modality_config()
    if len(modality["action"].delta_indices) < args.execution_horizon:
        raise ValueError("execution horizon exceeds checkpoint action horizon")

    train_dataset = args.train_dataset.resolve()
    heldout_dataset = args.heldout_dataset.resolve()
    train_loader = LeRobotEpisodeLoader(train_dataset, modality)
    train_map = read_source_map(train_dataset)
    heldout_loader = None
    heldout_map = None
    if not args.skip_heldout:
        heldout_loader = LeRobotEpisodeLoader(heldout_dataset, modality)
        heldout_map = read_source_map(heldout_dataset)
        if set(heldout_map.values()) & set(train_map.values()):
            raise ValueError("held-out dataset overlaps the training dataset")

    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        split_metrics: dict[str, Any] = {}
        split_specs = [("train", train_loader, args.train_episode_ids, train_map)]
        if heldout_loader is not None and heldout_map is not None:
            split_specs.append(
                ("heldout", heldout_loader, list(range(len(heldout_loader))), heldout_map)
            )
        for split, loader, ids, mapping in split_specs:
            plot_dir = output_dir / "plots_seed42" if seed == args.seeds[0] else None
            metrics, _ = evaluate_split(
                policy,
                loader,
                split,
                ids,
                mapping,
                seed,
                args.steps,
                args.execution_horizon,
                plot_dir,
            )
            split_metrics[split] = metrics
        runs.append({"seed": seed, "splits": split_metrics})

    def quality_counts(split: str) -> dict[str, int]:
        episode_metrics = [
            episode
            for run in runs
            for episode in run["splits"][split]["episode_metrics"]
        ]
        return {
            "episode_seed_pairs": len(episode_metrics),
            "xyz_max_over_100mm_pairs": sum(
                row["xyz_l2_mm"]["max"] > 100.0 for row in episode_metrics
            ),
            "rotation_max_over_10deg_pairs": sum(
                row["rotation_geodesic_deg"]["max"] > 10.0
                for row in episode_metrics
            ),
            "closure_abs_error_over_5steps_pairs": sum(
                row["closure_timing_abs_error_steps"] is not None
                and row["closure_timing_abs_error_steps"] > 5
                for row in episode_metrics
            ),
            "extra_predicted_closure_cycle_pairs": sum(
                row["pred_closed_onsets"] > row["gt_closed_onsets"]
                or row["pred_reopenings"] > row["gt_reopenings"]
                for row in episode_metrics
            ),
        }

    summaries = {"train": summarize_seeds(runs, "train")}
    quality_observations = {"train": quality_counts("train")}
    generalization_ratios = None
    heldout_degraded = False
    if heldout_loader is not None:
        summaries["heldout"] = summarize_seeds(runs, "heldout")
        quality_observations["heldout"] = quality_counts("heldout")
        generalization_ratios = {
            key: summaries["heldout"][key]["mean"] / summaries["train"][key]["mean"]
            for key in (
                "mse_10d",
                "mae_10d",
                "xyz_l2_mean_mm",
                "xyz_l2_p90_mm",
                "rotation_mean_deg",
                "rotation_p90_deg",
                "gripper_mae_equivalent_mm",
                "closure_timing_abs_error_steps_mean",
            )
        }
        heldout_degraded = (
            generalization_ratios["xyz_l2_mean_mm"] > 1.5
            or generalization_ratios["xyz_l2_p90_mm"] > 1.5
            or generalization_ratios["rotation_p90_deg"] > 1.5
        )

    warning_keys = (
        "xyz_max_over_100mm_pairs",
        "rotation_max_over_10deg_pairs",
        "closure_abs_error_over_5steps_pairs",
        "extra_predicted_closure_cycle_pairs",
    )
    train_warnings = any(
        quality_observations["train"][key] > 0 for key in warning_keys
    )
    heldout_warnings = heldout_loader is not None and any(
        quality_observations["heldout"][key] > 0 for key in warning_keys
    )
    if heldout_loader is not None and heldout_degraded:
        status = "EXECUTION_PASS_HELDOUT_DEGRADED"
    elif heldout_warnings:
        status = "EXECUTION_PASS_HELDOUT_PASS_WITH_OUTLIERS"
    elif train_warnings:
        status = "EXECUTION_PASS_TRAIN_FIT_WARNINGS"
    else:
        status = "TRAIN_FIT_PASS"

    report = {
        "status": status,
        "assessment": {
            "offline_inference_execution": "PASS",
            "training_fit_sanity": "PASS_WITH_WARNINGS" if train_warnings else "PASS",
            "heldout_generalization": (
                "NOT_EVALUATED"
                if heldout_loader is None
                else "DEGRADED_WITH_WARNINGS"
                if heldout_degraded
                else "PASS_WITH_OUTLIERS"
                if heldout_warnings
                else "PASS"
            ),
            "robot_deployment_readiness": "NOT_EVALUATED",
            "quality_observations": quality_observations,
            "heldout_to_train_ratios": generalization_ratios,
            "note": (
                "The 100 mm, 10 degree, and 5-step counters are descriptive review "
                "thresholds, not task-success guarantees. Held-out degradation is flagged "
                "when XYZ mean/P90 or rotation P90 exceeds 1.5x the training result."
            ),
        },
        "checkpoint": str(checkpoint),
        "global_step": global_step,
        "model_load_seconds": model_load_seconds,
        "protocol": {
            "execution_horizon": args.execution_horizon,
            "maximum_steps_per_episode": args.steps,
            "denoising_steps": args.denoising_steps,
            "seeds": args.seeds,
            "deterministic_pairing": "episode seed = base seed + full episode id * 1009",
            "train_subset_episode_ids": args.train_episode_ids,
            "heldout_subset_episode_ids": (
                list(range(len(heldout_loader))) if heldout_loader is not None else []
            ),
        },
        "datasets": {
            "train": str(train_dataset),
            "heldout": str(heldout_dataset) if heldout_loader is not None else None,
            "overlap": [],
        },
        "runs": runs,
        "summary": summaries,
    }
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"REPORT {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
