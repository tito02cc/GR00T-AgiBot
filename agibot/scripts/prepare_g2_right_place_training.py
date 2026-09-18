#!/usr/bin/env python3
"""Prepare explicit raw right-place splits without modifying or deleting source data.

Reuses the validated legacy converter's numeric/video implementation, but replaces
grasp-specific admission with the independent placement auditor. No source hashes,
smoothing, clipping, endpoint snapping, frame dropping, or inferred task aliases.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from audit_g2_right_place_raw import audit_episode
from convert_xichong_right_single_grasp import (
    CAMERAS,
    EpisodeValidation,
    convert_episode,
    load_frames,
    validate_image,
    write_json,
    write_metadata,
)
from tqdm import tqdm


REPO = Path(__file__).resolve().parents[2]


def read_manifest(path: Path) -> list[str]:
    names = [s.strip() for s in path.read_text().splitlines() if s.strip()]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Empty/duplicate manifest: {path}")
    if any(re.fullmatch(r"episode_\d{6}", name) is None for name in names):
        raise ValueError(f"Invalid episode names: {path}")
    return names


def check_splits(train: list[str], heldout: list[str]) -> None:
    if set(train) & set(heldout):
        raise ValueError("Training and held-out episodes overlap")


def prescreen_one(job: tuple) -> dict:
    root, name, profile = job
    row = audit_episode(root / name, profile)
    if row["status"] == "candidate":
        frames = load_frames(root / name / "frames.jsonl")
        try:
            for frame in frames:
                for camera in CAMERAS:
                    validate_image(root / name / frame["images"][camera])
        except Exception as exc:
            row["status"] = "invalid"
            row["invalid_reasons"].append({"code": "local_image_decode", "detail": str(exc)})
    return row


def convert_split(
    raw: Path, output: Path, names: list[str], profile: dict, rows: dict, workers: int, crf: int
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to reuse/overwrite dataset: {output}")
    output.mkdir(parents=True)
    results = []
    for name in names:
        frames = load_frames(raw / name / "frames.jsonl")
        results.append(
            EpisodeValidation(name, int(name[8:]), True, len(frames), profile["canonical_prompt"])
        )
    starts, running = [], 0
    for row in results:
        starts.append(running)
        running += row.length

    def convert(job):
        index, row, start = job
        result = convert_episode(
            raw / row.source_episode, output, index, start, 0, row.length, 10.0, crf, "fast", False
        )
        result["identity"] = rows[row.source_episode]["identity"]
        # Retain acquisition time separately from the fixed 10 Hz training grid.
        result["timestamps_monotonic"] = [
            f["timestamp_monotonic"] for f in load_frames(raw / row.source_episode / "frames.jsonl")
        ]
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        mapping = list(
            tqdm(
                pool.map(convert, zip(range(len(results)), results, starts)),
                total=len(results),
                desc=f"Convert {output.name}",
            )
        )
    write_metadata(output, results, {profile["canonical_prompt"]: 0}, 10.0)
    write_json(
        output / "meta/source_episode_map.json",
        {
            "source_root": str(raw),
            "pose_frame": "base_link_tf",
            "raw_action_mode": "next_delta_world_frame",
            "converted_action_mode": "absolute_target_xyz_rot6d",
            "task_mapping": profile,
            "episodes": mapping,
            "encoding": {
                "codec": "h264",
                "crf": crf,
                "preset": "fast",
                "fps": 10,
                "pixel_format": "yuv420p",
                "source_frames_preserved": True,
            },
        },
    )
    write_json(output / "meta/conversion_validation.json", [asdict(r) for r in results])
    return {"episodes": len(results), "frames": running, "dataset": str(output)}


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--heldout-manifest", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--modality-config", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--stage", choices=("all", "convert", "verify"), default="all")
    p.add_argument("--expected-train-count", type=int)
    p.add_argument("--expected-heldout-count", type=int)
    p.add_argument("--normalization-bounds", choices=("minmax", "percentile"), default="minmax")
    args = p.parse_args()
    status_path = args.report_dir.resolve() / "preparation_status.json"
    write_json(
        status_path,
        {
            "status": "RUNNING",
            "stage": args.stage,
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )
    try:
        prepare(args)
    except BaseException as exc:
        write_json(
            status_path,
            {
                "status": "FAIL",
                "stage": args.stage,
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )
        raise
    if args.stage == "convert":
        write_json(
            status_path,
            {
                "status": "CONVERSION_COMPLETE_VALIDATION_PENDING",
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )


def prepare(args) -> None:
    if args.workers < 1 or not 0 <= args.crf <= 51:
        raise ValueError("workers must be positive; CRF must be in [0,51]")
    raw, output, reports = (x.resolve() for x in (args.raw_root, args.output_root, args.report_dir))
    train, heldout = read_manifest(args.train_manifest), read_manifest(args.heldout_manifest)
    check_splits(train, heldout)
    for expected, actual in (
        (args.expected_train_count, len(train)),
        (args.expected_heldout_count, len(heldout)),
    ):
        if expected is not None and expected != actual:
            raise ValueError(f"Manifest count {actual} != configured {expected}")
    profile = json.loads(args.profile.read_text())
    train_dir, heldout_dir = output / "train", output / "heldout"
    reports.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "convert"):
        if train_dir.exists() or heldout_dir.exists():
            raise FileExistsError(
                "Use a fresh output directory; verify-only can audit existing data"
            )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(
                tqdm(
                    pool.map(prescreen_one, [(raw, n, profile) for n in train + heldout]),
                    total=len(train + heldout),
                    desc="Local raw + full JPEG audit",
                )
            )
        write_json(reports / "local_raw_audit.json", {"episodes": rows})
        rejected = [r["episode"] for r in rows if r["status"] != "candidate"]
        if rejected:
            raise ValueError(f"Selected raw episodes failed fresh admission: {rejected}")
        uuid = [r["identity"]["episode_uuid"] for r in rows]
        if not all(uuid) or len(uuid) != len(set(uuid)):
            raise ValueError("Missing/duplicate episode UUID across splits")
        rows = {r["episode"]: r for r in rows}
        # Portable manifests are the sole episode selection; do not glob the raw root.
        for name, values in (("train", train), ("heldout", heldout)):
            (reports / f"{name}.txt").write_text("\n".join(values) + "\n")
        summary = {
            "train": convert_split(raw, train_dir, train, profile, rows, args.workers, args.crf),
            "heldout": convert_split(
                raw, heldout_dir, heldout, profile, rows, args.workers, args.crf
            ),
            "source_unchanged": True,
            "content_hashes_computed": False,
        }
        write_json(reports / "conversion_summary.json", summary)
    if args.stage in ("all", "verify"):
        # Official stats run on TRAIN only. Held-out uses exactly those statistics.
        run(
            [
                sys.executable,
                "gr00t/data/stats.py",
                "--dataset-path",
                str(train_dir),
                "--embodiment-tag",
                "NEW_EMBODIMENT",
                "--modality-config-path",
                str(args.modality_config.resolve()),
            ]
        )
        for filename in ("stats.json", "relative_stats.json"):
            shutil.copyfile(train_dir / "meta" / filename, heldout_dir / "meta" / filename)
        for split, dataset in (("train", train_dir), ("heldout", heldout_dir)):
            write_json(
                dataset / "meta/normalization_provenance.json",
                {
                    "normalization_source": "train",
                    "train_episodes": train,
                    "heldout_used_to_fit": False,
                },
            )
            manifest = args.train_manifest if split == "train" else args.heldout_manifest
            run(
                [
                    sys.executable,
                    "agibot/scripts/validate_g2_right_arm_gr00t_conversion.py",
                    "--raw-root",
                    str(raw),
                    "--dataset",
                    str(dataset),
                    "--episode-manifest",
                    str(manifest.resolve()),
                    "--report",
                    str(reports / f"{split}_conversion_audit.json"),
                    "--workers",
                    str(args.workers),
                    "--all-source-frames",
                    "--minimum-psnr-db",
                    "30",
                ]
            )
        run(
            [
                sys.executable,
                "agibot/scripts/verify_g2_training_math.py",
                "--train",
                str(train_dir),
                "--heldout",
                str(heldout_dir),
                "--modality-config",
                str(args.modality_config.resolve()),
                "--report",
                str(reports / "official_loading_and_math.json"),
                "--workers",
                str(args.workers),
                "--normalization-bounds",
                args.normalization_bounds,
            ]
        )
        write_json(
            reports / "preparation_status.json",
            {
                "updated_at": datetime.now().astimezone().isoformat(),
                "status": "AUTOMATED_DATA_CHECKS_PASS",
                "train": str(train_dir),
                "heldout": str(heldout_dir),
                "physical_placement_visual_review": "not_certified",
                "model_training_started": False,
                "normalization_bounds": args.normalization_bounds,
            },
        )


if __name__ == "__main__":
    main()
