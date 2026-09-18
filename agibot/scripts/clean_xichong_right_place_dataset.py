#!/usr/bin/env python3
"""Finalize the right-place train/held-out manifests after strict raw-data cleaning.

The script never modifies raw episodes. It validates the selected source data,
removes trajectory/action outliers, validates explicitly chosen replacements,
keeps the held-out split isolated, and writes deterministic content receipts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from convert_xichong_right_single_grasp import EpisodeValidation, validate_episode


AGIBOT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = AGIBOT_ROOT / "data/xichong_right_place_r0002_job01"
DEFAULT_SELECTION_DIR = (
    AGIBOT_ROOT
    / "reports/xichong_right_place_r0002_selection_600_train_100_heldout"
)
DEFAULT_OUTPUT_DIR = AGIBOT_ROOT / "reports/xichong_right_place_r0002_final_clean"
EXPECTED_TASK = "operator_xichong_right_place_right_place_release_withdraw_r0002"
EXPECTED_PROMPT = (
    "The right arm moves the gripped workpiece to the Xichong placement target, "
    "opens the right gripper to release the workpiece, then retracts the right arm "
    "to a safe pose."
)
CAMERAS = ("head_color", "hand_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-step-translation-m", type=float, default=0.08)
    parser.add_argument("--max-step-rotation-rad", type=float, default=0.15)
    parser.add_argument("--max-gripper-backtrack", type=float, default=0.05)
    parser.add_argument(
        "--skip-image-decode",
        action="store_true",
        help="Skip full JPEG decoding. This should not be used for a final receipt.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate entries in {path}")
    return values


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(f"{value}\n" for value in values))
    temporary.replace(path)


def load_frames(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dynamic_metrics(episode_dir: Path) -> dict[str, float | int]:
    frames = load_frames(episode_dir / "frames.jsonl")
    actions = np.asarray([frame["action_right_7d"] for frame in frames], dtype=np.float64)
    gripper = np.asarray(
        [frame["right_gripper"]["position"] for frame in frames], dtype=np.float64
    )
    translation = np.linalg.norm(actions[:, :3], axis=1)
    rotation = np.linalg.norm(actions[:, 3:6], axis=1)
    gripper_backtrack = gripper - np.minimum.accumulate(gripper)
    return {
        "frames": len(frames),
        "max_step_translation_m": float(translation.max(initial=0.0)),
        "p99_step_translation_m": float(np.quantile(translation, 0.99)),
        "max_step_rotation_rad": float(rotation.max(initial=0.0)),
        "p99_step_rotation_rad": float(np.quantile(rotation, 0.99)),
        "max_gripper_backtrack": float(gripper_backtrack.max(initial=0.0)),
        "initial_gripper": float(gripper[0]),
        "terminal_gripper": float(gripper[-1]),
    }


def task_checks(episode_dir: Path) -> tuple[dict[str, bool], dict[str, Any]]:
    meta = read_json(episode_dir / "meta_info.json")
    quality = read_json(episode_dir / "quality_report.json")
    manifest = read_json(episode_dir / "parameters/manifest_row.json")
    receipt = read_json(episode_dir / "parameters/episode_receipt.json")
    prompt = json.loads(meta["text"])["description"]
    semantic = quality.get("semantic_terminal") or quality.get("xichong_terminal") or {}
    metrics = semantic.get("metrics", {})
    checks = {
        "task_matches": manifest.get("task_name") == EXPECTED_TASK,
        "right_arm": manifest.get("arm_mode") == "right",
        "prompt_matches": prompt == EXPECTED_PROMPT,
        "right_place_semantics": bool(semantic.get("ok"))
        and semantic.get("phase") == "right_place",
        "initial_right_gripper_held": bool(metrics.get("initial_held", {}).get("right")),
        "terminal_right_gripper_open": metrics.get("terminal_state") == "open",
        "release_anchor_present": metrics.get("release_anchor_index") is not None,
        "receipt_identity": receipt.get("episode") == episode_dir.name
        and receipt.get("episode_uuid") == meta.get("episode_uuid"),
    }
    return checks, {
        "episode_uuid": meta.get("episode_uuid"),
        "receipt_root_sha256": receipt.get("root_sha256"),
        "prompt": prompt,
        "release_anchor_index": metrics.get("release_anchor_index"),
    }


def validate_all(
    raw_root: Path,
    episodes: list[str],
    workers: int,
    decode_images: bool,
) -> dict[str, EpisodeValidation]:
    results: dict[str, EpisodeValidation] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(validate_episode, raw_root / episode, decode_images): episode
            for episode in episodes
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            episode = futures[future]
            results[episode] = future.result()
            if completed % 50 == 0 or completed == len(futures):
                print(f"strict_validation={completed}/{len(futures)}", flush=True)
    return results


def audit_episode(
    raw_root: Path,
    episode: str,
    strict: EpisodeValidation,
    args: argparse.Namespace,
) -> dict[str, Any]:
    checks, details = task_checks(raw_root / episode)
    metrics = dynamic_metrics(raw_root / episode)
    gates = {
        "strict_converter_validation": strict.valid,
        **checks,
        "step_translation_within_limit": (
            metrics["max_step_translation_m"] <= args.max_step_translation_m
        ),
        "step_rotation_within_limit": (
            metrics["max_step_rotation_rad"] <= args.max_step_rotation_rad
        ),
        "gripper_backtrack_within_limit": (
            metrics["max_gripper_backtrack"] <= args.max_gripper_backtrack
        ),
    }
    return {
        "episode": episode,
        "valid": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "gates": gates,
        "strict_errors": strict.errors,
        "metrics": metrics,
        **details,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_receipt(raw_root: Path, episodes: list[str]) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    dataset_digest = hashlib.sha256()
    for completed, episode in enumerate(episodes, start=1):
        episode_dir = raw_root / episode
        files = [
            episode_dir / "arrays.npz",
            episode_dir / "frames.jsonl",
            episode_dir / "meta_info.json",
            episode_dir / "quality_report.json",
        ]
        for camera in CAMERAS:
            files.extend(sorted((episode_dir / "images").glob(f"{camera}_*.jpg")))
        episode_digest = hashlib.sha256()
        total_bytes = 0
        for path in files:
            relative = path.relative_to(raw_root).as_posix()
            file_digest = sha256_file(path)
            total_bytes += path.stat().st_size
            episode_digest.update(relative.encode())
            episode_digest.update(b"\0")
            episode_digest.update(file_digest.encode())
            episode_digest.update(b"\0")
        digest = episode_digest.hexdigest()
        dataset_digest.update(episode.encode())
        dataset_digest.update(b"\0")
        dataset_digest.update(digest.encode())
        dataset_digest.update(b"\0")
        rows.append(
            {
                "episode": episode,
                "sha256": digest,
                "file_count": len(files),
                "bytes": total_bytes,
            }
        )
        if completed % 100 == 0 or completed == len(episodes):
            print(f"content_receipt={completed}/{len(episodes)}", flush=True)
    return rows, dataset_digest.hexdigest()


def summary(audits: list[dict[str, Any]]) -> dict[str, Any]:
    def quantiles(key: str) -> dict[str, float]:
        values = np.asarray([row["metrics"][key] for row in audits], dtype=np.float64)
        points = np.quantile(values, [0.0, 0.5, 0.95, 0.99, 1.0])
        return {
            name: round(float(value), 8)
            for name, value in zip(("min", "p50", "p95", "p99", "max"), points)
        }

    return {
        "episodes": len(audits),
        "frames": sum(int(row["metrics"]["frames"]) for row in audits),
        "max_step_translation_m": quantiles("max_step_translation_m"),
        "max_step_rotation_rad": quantiles("max_step_rotation_rad"),
        "max_gripper_backtrack": quantiles("max_gripper_backtrack"),
    }


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    selection_dir = args.selection_dir.resolve()
    output_dir = args.output_dir.resolve()

    train = read_lines(selection_dir / "train_episodes.txt")
    heldout = read_lines(selection_dir / "heldout_episodes.txt")
    replacement_map = read_json(selection_dir / "replacement_map.json")
    replacements = list(replacement_map.values())
    all_checked = sorted(set(train) | set(heldout) | set(replacements))
    if len(train) != 600 or len(heldout) != 100:
        raise ValueError("expected the original 600 train / 100 held-out split")
    if set(train) & set(heldout):
        raise ValueError("original train and held-out manifests overlap")
    if set(replacement_map) - set(train):
        raise ValueError("replacement_map contains a rejected episode outside train")
    if set(replacements) & (set(train) | set(heldout)):
        raise ValueError("replacement episodes already occur in the original candidate set")

    strict_results = validate_all(
        raw_root,
        all_checked,
        args.workers,
        not args.skip_image_decode,
    )
    audits = {
        episode: audit_episode(raw_root, episode, strict_results[episode], args)
        for episode in all_checked
    }
    rejected_train = sorted(episode for episode in train if not audits[episode]["valid"])
    rejected_heldout = sorted(
        episode for episode in heldout if not audits[episode]["valid"]
    )
    if set(rejected_train) != set(replacement_map):
        raise RuntimeError(
            "observed train rejects do not exactly match replacement_map: "
            f"observed={rejected_train} map={sorted(replacement_map)}"
        )
    if rejected_heldout:
        raise RuntimeError(f"held-out split contains invalid episodes: {rejected_heldout}")
    invalid_replacements = [episode for episode in replacements if not audits[episode]["valid"]]
    if invalid_replacements:
        raise RuntimeError(f"replacement episodes failed cleaning: {invalid_replacements}")

    final_train = sorted((set(train) - set(rejected_train)) | set(replacements))
    final_heldout = sorted(heldout)
    final_all = sorted(final_train + final_heldout)
    if len(final_train) != 600 or len(final_heldout) != 100 or len(final_all) != 700:
        raise RuntimeError("final split counts are not exactly 600 / 100 / 700")
    if set(final_train) & set(final_heldout):
        raise RuntimeError("final train and held-out manifests overlap")

    final_audits = [audits[episode] for episode in final_all]
    uuids = [row["episode_uuid"] for row in final_audits]
    receipt_hashes = [row["receipt_root_sha256"] for row in final_audits]
    if len(set(uuids)) != len(uuids) or len(set(receipt_hashes)) != len(receipt_hashes):
        raise RuntimeError("duplicate UUID or source receipt hash in final data")

    receipts, dataset_sha256 = content_receipt(raw_root, final_all)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_lines(output_dir / "train_episodes.txt", final_train)
    write_lines(output_dir / "heldout_episodes.txt", final_heldout)
    write_lines(output_dir / "all_episodes.txt", final_all)
    write_lines(output_dir / "rejected_episodes.txt", rejected_train)
    write_json(output_dir / "replacement_map.json", replacement_map)
    write_lines(
        output_dir / "content.sha256",
        [f"{row['sha256']}  {row['episode']}" for row in receipts]
        + [f"{dataset_sha256}  FINAL_DATASET"],
    )

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "PASS",
        "raw_root": str(raw_root),
        "image_decode_validation": not args.skip_image_decode,
        "thresholds": {
            "max_step_translation_m": args.max_step_translation_m,
            "max_step_rotation_rad": args.max_step_rotation_rad,
            "max_gripper_backtrack": args.max_gripper_backtrack,
        },
        "counts": {
            "original_train": len(train),
            "rejected_train": len(rejected_train),
            "replacements": len(replacements),
            "final_train": len(final_train),
            "final_heldout": len(final_heldout),
            "final_all": len(final_all),
        },
        "rejected": [audits[episode] for episode in rejected_train],
        "replacement_map": replacement_map,
        "replacements": [audits[episode] for episode in replacements],
        "final_summary": summary(final_audits),
        "final_train_summary": summary([audits[episode] for episode in final_train]),
        "final_heldout_summary": summary([audits[episode] for episode in final_heldout]),
        "dataset_sha256": dataset_sha256,
        "content": receipts,
        "episodes": final_audits,
    }
    write_json(output_dir / "audit.json", report)
    print(
        f"PASS final_train={len(final_train)} final_heldout={len(final_heldout)} "
        f"frames={report['final_summary']['frames']} sha256={dataset_sha256}"
    )
    print(f"report={output_dir / 'audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
