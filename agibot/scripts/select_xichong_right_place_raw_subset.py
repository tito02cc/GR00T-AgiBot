#!/usr/bin/env python3
"""Audit and select raw right-place episodes before downloading full images.

The script only needs each episode's arrays, JSON/JSONL metadata, and quality
report.  It applies collector, synchronization, lineage, and place/release/
withdraw semantic gates, then performs chronological-stratified k-center
sampling over the right-arm trajectory.  It writes deterministic train,
held-out, candidate, and reserve manifests; it never modifies raw episodes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from select_xichong_training_subset import (
    allocate_quotas,
    coverage_report,
    interpolate_episode,
    kcenter_indices,
    quantiles,
    raw_quality_gate,
    write_json,
)


AGIBOT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = AGIBOT_ROOT / "data/xichong_right_place_r0002_job01"
DEFAULT_REPORT = (
    AGIBOT_ROOT
    / "reports/xichong_right_place_r0002_selection_600_train_100_heldout.json"
)
EXPECTED_BATCH = "xichong_right_place_r0002_job01"
EXPECTED_TASK = "operator_xichong_right_place_right_place_release_withdraw_r0002"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--train-count", type=int, default=600)
    parser.add_argument("--heldout-count", type=int, default=100)
    parser.add_argument("--strata", type=int, default=20)
    parser.add_argument("--resample-points", type=int, default=32)
    parser.add_argument("--transfer-shards", type=int, default=4)
    parser.add_argument("--max-camera-span-ms", type=float, default=100.0)
    parser.add_argument("--max-head-state-offset-ms", type=float, default=100.0)
    parser.add_argument("--max-hand-state-offset-ms", type=float, default=50.0)
    parser.add_argument("--min-withdraw-m", type=float, default=0.10)
    parser.add_argument("--max-terminal-stability-m", type=float, default=0.005)
    parser.add_argument("--max-inactive-left-displacement-m", type=float, default=0.005)
    parser.add_argument("--min-pre-release-frames", type=int, default=15)
    parser.add_argument("--min-post-release-frames", type=int, default=10)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def episode_number(path: Path) -> int:
    return int(path.name.removeprefix("episode_"))


def place_quality_gate(
    episode_dir: Path, args: argparse.Namespace
) -> tuple[bool, dict[str, Any]]:
    sync_ok, sync_details = raw_quality_gate(
        episode_dir,
        args.max_camera_span_ms,
        args.max_head_state_offset_ms,
        args.max_hand_state_offset_ms,
    )
    meta = load_json(episode_dir / "meta_info.json")
    quality = load_json(episode_dir / "quality_report.json")
    config = load_json(episode_dir / "parameters/collector_config.json")["args"]
    manifest = load_json(episode_dir / "parameters/manifest_row.json")
    receipt = load_json(episode_dir / "parameters/episode_receipt.json")

    semantic = quality.get("semantic_terminal") or quality.get("xichong_terminal") or {}
    metrics = semantic.get("metrics", {})
    saved_frames = int(metrics.get("saved_frames", 0))
    release_anchor = metrics.get("release_anchor_index")
    release_anchor = int(release_anchor) if release_anchor is not None else None
    withdraw = metrics.get("withdraw_m", {}).get("right")
    stability = metrics.get("terminal_stability_radius_m", {}).get("right")
    inactive_left = metrics.get("inactive_displacement_m", {}).get("left")

    annotations = meta.get("annotations", {})
    checks = {
        "sync_gate_ok": sync_ok,
        "batch_matches": config.get("operator_batch_id") == EXPECTED_BATCH,
        "task_matches": manifest.get("task_name") == EXPECTED_TASK,
        "right_arm": manifest.get("arm_mode") == "right",
        "manifest_quality_ok": str(manifest.get("quality_ok")).lower() == "true",
        "annotated_success": manifest.get("success") == "y",
        "no_collision": annotations.get("collision") == "n",
        "no_slip": annotations.get("slip") == "n",
        "no_manual_correction": annotations.get("manual_correction") == "n",
        "no_person_visible": annotations.get("person_visible") == "n",
        "semantic_right_place_ok": bool(semantic.get("applicable"))
        and semantic.get("phase") == "right_place"
        and bool(semantic.get("ok")),
        "initial_right_gripper_held": bool(
            metrics.get("initial_held", {}).get("right")
        ),
        "terminal_right_gripper_open": bool(
            metrics.get("initial_open", {}).get("right") is False
            and metrics.get("terminal_state") == "open"
        ),
        "release_anchor_present": release_anchor is not None,
        "enough_pre_release_frames": release_anchor is not None
        and release_anchor >= args.min_pre_release_frames,
        "enough_post_release_frames": release_anchor is not None
        and saved_frames - 1 - release_anchor >= args.min_post_release_frames,
        "withdraw_distance_ok": withdraw is not None
        and float(withdraw) >= args.min_withdraw_m,
        "terminal_stability_ok": stability is not None
        and float(stability) <= args.max_terminal_stability_m,
        "inactive_left_arm_ok": inactive_left is not None
        and float(inactive_left) <= args.max_inactive_left_displacement_m,
        "receipt_identity_ok": receipt.get("episode") == episode_dir.name
        and receipt.get("episode_uuid") == meta.get("episode_uuid"),
    }
    details = {
        "checks": checks,
        "sync": sync_details,
        "metrics": {
            "saved_frames": saved_frames,
            "release_anchor_index": release_anchor,
            "withdraw_m": withdraw,
            "terminal_stability_radius_m": stability,
            "inactive_left_displacement_m": inactive_left,
            "receipt_root_sha256": receipt.get("root_sha256"),
        },
        "failed_checks": [name for name, ok in checks.items() if not ok],
    }
    return all(checks.values()), details


def write_manifest(path: Path, episodes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{episode}\n" for episode in episodes))


def semantic_metric_summary(
    audit: dict[int, dict[str, Any]], indices: list[int]
) -> dict[str, Any]:
    def values(key: str) -> np.ndarray:
        return np.asarray(
            [float(audit[index]["metrics"][key]) for index in indices],
            dtype=np.float64,
        )

    saved_frames = values("saved_frames")
    release_anchor = values("release_anchor_index")
    return {
        "episodes": len(indices),
        "saved_frames": quantiles(saved_frames),
        "release_progress": quantiles(release_anchor / saved_frames),
        "withdraw_m": quantiles(values("withdraw_m")),
        "terminal_stability_radius_m": quantiles(
            values("terminal_stability_radius_m")
        ),
        "inactive_left_displacement_m": quantiles(
            values("inactive_left_displacement_m")
        ),
    }


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    report_path = args.report.resolve()
    candidate_count = args.train_count + args.heldout_count
    if min(
        args.train_count,
        args.heldout_count,
        args.strata,
        args.resample_points,
        args.transfer_shards,
    ) <= 0:
        raise ValueError("counts, strata, and resample-points must be positive")

    episode_dirs = sorted(raw_root.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if not episode_dirs:
        raise FileNotFoundError(f"no episodes under {raw_root}")
    episode_indices = [episode_number(path) for path in episode_dirs]
    if episode_indices != list(range(episode_indices[-1] + 1)):
        raise ValueError("episode indices are not contiguous from zero")

    required = (
        "arrays.npz",
        "frames.jsonl",
        "meta_info.json",
        "quality_report.json",
        "parameters/collector_config.json",
        "parameters/episode_receipt.json",
        "parameters/manifest_row.json",
    )
    incomplete = [
        path.name
        for path in episode_dirs
        if any(not (path / relative).is_file() for relative in required)
    ]
    if incomplete:
        raise RuntimeError(
            f"metadata sync incomplete for {len(incomplete)} episodes; "
            f"first={incomplete[:5]}"
        )

    first_arrays = episode_dirs[0] / "arrays.npz"
    with np.load(first_arrays) as arrays:
        reference = Rotation.from_quat(arrays["states"][0, 11:15].astype(np.float64))
    grid = np.linspace(0.0, 1.0, args.resample_points)

    eligible: list[int] = []
    audit: dict[int, dict[str, Any]] = {}
    features: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    for index, episode_dir in enumerate(episode_dirs):
        accepted, details = place_quality_gate(episode_dir, args)
        audit[index] = details
        if accepted:
            eligible.append(index)
        feature, summary = interpolate_episode(
            episode_dir / "arrays.npz", grid, reference
        )
        semantic_metrics = details["metrics"]
        feature = np.concatenate(
            [
                feature,
                np.asarray(
                    [
                        float(semantic_metrics["release_anchor_index"] or 0)
                        / max(float(semantic_metrics["saved_frames"]), 1.0),
                        float(semantic_metrics["withdraw_m"] or 0.0) / 0.10,
                    ]
                ),
            ]
        )
        features.append(feature)
        summaries.append(summary)

    if candidate_count > len(eligible):
        raise ValueError(
            f"requested {candidate_count} candidates from {len(eligible)} eligible episodes"
        )
    feature_array = np.stack(features)
    all_strata = [part.tolist() for part in np.array_split(episode_indices, args.strata)]
    eligible_set = set(eligible)
    strata = [[index for index in part if index in eligible_set] for part in all_strata]
    if any(not members for members in strata):
        raise ValueError("quality filtering left an empty chronological stratum")

    train_quotas = allocate_quotas([len(members) for members in strata], args.train_count)
    train_by_stratum: list[list[int]] = []
    for members, quota in zip(strata, train_quotas):
        local = kcenter_indices(feature_array[members], quota)
        train_by_stratum.append(sorted(members[position] for position in local))
    train = sorted(index for group in train_by_stratum for index in group)
    train_set = set(train)

    heldout_quotas = allocate_quotas(
        [len([index for index in members if index not in train_set]) for members in strata],
        args.heldout_count,
    )
    heldout_by_stratum: list[list[int]] = []
    for members, quota in zip(strata, heldout_quotas):
        remaining = [index for index in members if index not in train_set]
        local = kcenter_indices(feature_array[remaining], quota)
        heldout_by_stratum.append(
            sorted(remaining[position] for position in local)
        )
    heldout = sorted(index for group in heldout_by_stratum for index in group)
    candidates = sorted(train + heldout)
    reserve = sorted(eligible_set - set(candidates))
    if len(train) != args.train_count or len(heldout) != args.heldout_count:
        raise RuntimeError("train/held-out allocation did not produce exact counts")

    manifest_dir = report_path.parent / report_path.stem
    names = [path.name for path in episode_dirs]
    write_manifest(manifest_dir / "candidate_episodes.txt", [names[i] for i in candidates])
    write_manifest(manifest_dir / "train_episodes.txt", [names[i] for i in train])
    write_manifest(manifest_dir / "heldout_episodes.txt", [names[i] for i in heldout])
    write_manifest(manifest_dir / "reserve_episodes.txt", [names[i] for i in reserve])
    transfer_shards = [
        [int(index) for index in shard]
        for shard in np.array_split(np.asarray(candidates), args.transfer_shards)
    ]
    for shard_index, shard in enumerate(transfer_shards):
        write_manifest(
            manifest_dir / f"candidate_shard_{shard_index:02d}.txt",
            [names[index] for index in shard],
        )

    report = {
        "raw_root": str(raw_root),
        "algorithm": {
            "name": "raw_place_gate_chronological_stratified_kcenter",
            "train_count": args.train_count,
            "heldout_count": args.heldout_count,
            "candidate_count": candidate_count,
            "strata": args.strata,
            "resample_points": args.resample_points,
        },
        "thresholds": {
            "max_camera_span_ms": args.max_camera_span_ms,
            "max_head_state_offset_ms": args.max_head_state_offset_ms,
            "max_hand_state_offset_ms": args.max_hand_state_offset_ms,
            "min_withdraw_m": args.min_withdraw_m,
            "max_terminal_stability_m": args.max_terminal_stability_m,
            "max_inactive_left_displacement_m": args.max_inactive_left_displacement_m,
            "min_pre_release_frames": args.min_pre_release_frames,
            "min_post_release_frames": args.min_post_release_frames,
        },
        "source": {
            "episodes": len(episode_dirs),
            "first_episode": episode_dirs[0].name,
            "last_episode": episode_dirs[-1].name,
            "eligible": len(eligible),
            "excluded": len(episode_dirs) - len(eligible),
        },
        "exclusion_reason_counts": dict(
            Counter(
                reason
                for index in episode_indices
                if index not in eligible_set
                for reason in audit[index]["failed_checks"]
            ).most_common()
        ),
        "manifests": {
            "directory": str(manifest_dir),
            "candidate": "candidate_episodes.txt",
            "train": "train_episodes.txt",
            "heldout": "heldout_episodes.txt",
            "reserve": "reserve_episodes.txt",
            "transfer_shards": [
                f"candidate_shard_{index:02d}.txt"
                for index in range(args.transfer_shards)
            ],
        },
        "candidate_episodes": [names[i] for i in candidates],
        "train_episodes": [names[i] for i in train],
        "heldout_episodes": [names[i] for i in heldout],
        "excluded": [
            {"episode": names[index], **audit[index]}
            for index in episode_indices
            if index not in eligible_set
        ],
        "strata": [
            {
                "index": stratum_index,
                "source_start": names[all_strata[stratum_index][0]],
                "source_end": names[all_strata[stratum_index][-1]],
                "eligible": len(strata[stratum_index]),
                "candidates": len(train_by_stratum[stratum_index])
                + len(heldout_by_stratum[stratum_index]),
                "train": len(train_by_stratum[stratum_index]),
                "heldout": len(heldout_by_stratum[stratum_index]),
            }
            for stratum_index in range(args.strata)
        ],
        "coverage": {
            "candidates": coverage_report(summaries, candidates, reserve),
            "train_vs_heldout": coverage_report(summaries, train, heldout),
            "heldout_vs_train": coverage_report(summaries, heldout, train),
            "semantic_metrics": {
                "eligible": semantic_metric_summary(audit, eligible),
                "train": semantic_metric_summary(audit, train),
                "heldout": semantic_metric_summary(audit, heldout),
                "reserve": semantic_metric_summary(audit, reserve),
            },
        },
    }
    write_json(report_path, report)
    print(
        f"source={len(episode_dirs)} eligible={len(eligible)} "
        f"train={len(train)} heldout={len(heldout)} reserve={len(reserve)}"
    )
    print(f"report={report_path}")
    print(f"manifests={manifest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
