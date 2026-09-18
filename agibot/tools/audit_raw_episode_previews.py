#!/usr/bin/env python3
"""Audit first/middle/last raw episode previews and build visual review sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("head_color", "hand_right")
STAGES = ("first", "middle", "last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--episode-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-page", type=int, default=8)
    parser.add_argument("--cell-width", type=int, default=240)
    parser.add_argument("--cell-height", type=int, default=180)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_metrics(image: np.ndarray) -> dict[str, float | list[int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "shape": list(image.shape),
        "mean": round(float(gray.mean()), 4),
        "std": round(float(gray.std()), 4),
        "p01": round(float(np.percentile(gray, 1)), 4),
        "p99": round(float(np.percentile(gray, 99)), 4),
        "laplacian_variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4),
    }


def labelled_cell(
    image: np.ndarray,
    label: str,
    width: int,
    height: int,
) -> np.ndarray:
    cell = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(cell, (0, height - 22), (width, height), (0, 0, 0), -1)
    cv2.putText(
        cell,
        label,
        (5, height - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cell


def placeholder(label: str, width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.putText(
        image,
        label,
        (8, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (70, 70, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    episode_list = args.episode_list.resolve()
    output = args.output.resolve()
    if args.episodes_per_page < 1 or args.cell_width < 32 or args.cell_height < 32:
        raise ValueError("invalid page or cell dimensions")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    episodes = [line.strip() for line in episode_list.read_text().splitlines() if line.strip()]
    if len(episodes) != len(set(episodes)):
        raise ValueError("episode list contains duplicates")

    records: list[dict] = []
    episode_rows: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None

    for episode in episodes:
        episode_dir = raw_root / episode
        hard_issues: list[str] = []
        warnings: list[str] = []
        previews: dict[str, np.ndarray] = {}
        preview_records: dict[str, dict] = {}
        camera_hashes: dict[str, list[str]] = {camera: [] for camera in CAMERAS}

        for camera in CAMERAS:
            for stage in STAGES:
                key = f"{stage}_{camera}"
                path = episode_dir / f"preview_{stage}_{camera}.jpg"
                if not path.is_file():
                    hard_issues.append(f"missing:{path.name}")
                    continue
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    hard_issues.append(f"undecodable:{path.name}")
                    continue
                previews[key] = image
                shape = tuple(image.shape)
                if expected_shape is None:
                    expected_shape = shape
                elif shape != expected_shape:
                    hard_issues.append(f"shape:{path.name}:{shape}")
                metrics = image_metrics(image)
                dynamic_range = float(metrics["p99"]) - float(metrics["p01"])
                if dynamic_range < 5.0:
                    hard_issues.append(f"near_constant:{path.name}")
                elif float(metrics["mean"]) < 10.0 or float(metrics["mean"]) > 245.0:
                    warnings.append(f"extreme_brightness:{path.name}")
                if float(metrics["laplacian_variance"]) < 10.0:
                    warnings.append(f"low_detail:{path.name}")
                digest = file_sha256(path)
                camera_hashes[camera].append(digest)
                preview_records[key] = {
                    "path": str(path.relative_to(raw_root)),
                    "sha256": digest,
                    **metrics,
                }

        temporal_deltas: dict[str, dict[str, float]] = {}
        for camera in CAMERAS:
            hashes = camera_hashes[camera]
            if len(hashes) == len(STAGES) and len(set(hashes)) < len(hashes):
                hard_issues.append(f"exact_temporal_duplicate:{camera}")
            deltas: dict[str, float] = {}
            for left, right in zip(STAGES, STAGES[1:]):
                left_image = previews.get(f"{left}_{camera}")
                right_image = previews.get(f"{right}_{camera}")
                if left_image is None or right_image is None or left_image.shape != right_image.shape:
                    continue
                delta = float(
                    np.mean(
                        np.abs(left_image.astype(np.int16) - right_image.astype(np.int16))
                    )
                )
                deltas[f"{left}_to_{right}"] = round(delta, 4)
                if delta < 0.25:
                    warnings.append(f"very_low_temporal_change:{camera}:{left}_to_{right}")
            temporal_deltas[camera] = deltas

        title_height = 30
        row = np.full(
            (title_height + args.cell_height, len(CAMERAS) * len(STAGES) * args.cell_width, 3),
            18,
            dtype=np.uint8,
        )
        status = "FAIL" if hard_issues else ("WARN" if warnings else "OK")
        title = f"{episode} | {status} | hard={len(hard_issues)} warn={len(warnings)}"
        cv2.putText(
            row,
            title,
            (7, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 80, 255) if hard_issues else (80, 210, 255) if warnings else (120, 235, 120),
            1,
            cv2.LINE_AA,
        )
        column = 0
        for camera in CAMERAS:
            for stage in STAGES:
                key = f"{stage}_{camera}"
                image = previews.get(key)
                cell = (
                    labelled_cell(image, f"{camera} / {stage}", args.cell_width, args.cell_height)
                    if image is not None
                    else placeholder(f"MISSING {key}", args.cell_width, args.cell_height)
                )
                x0 = column * args.cell_width
                row[title_height:, x0 : x0 + args.cell_width] = cell
                column += 1
        episode_rows.append(row)
        records.append(
            {
                "episode": episode,
                "status": status,
                "hard_issues": hard_issues,
                "warnings": warnings,
                "temporal_mean_absolute_deltas": temporal_deltas,
                "previews": preview_records,
            }
        )

    pages: list[dict] = []
    for start in range(0, len(episode_rows), args.episodes_per_page):
        page_rows = episode_rows[start : start + args.episodes_per_page]
        page = np.vstack(page_rows)
        page_index = start // args.episodes_per_page
        page_path = output / f"review_page_{page_index:03d}.jpg"
        if not cv2.imwrite(str(page_path), page, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {page_path}")
        pages.append(
            {
                "page": page_path.name,
                "episodes": episodes[start : start + args.episodes_per_page],
            }
        )

    status_counts = {
        status: sum(record["status"] == status for record in records)
        for status in ("OK", "WARN", "FAIL")
    }
    report = {
        "raw_root": str(raw_root),
        "episode_list": str(episode_list),
        "episode_count": len(episodes),
        "required_cameras": list(CAMERAS),
        "required_stages": list(STAGES),
        "expected_image_shape": list(expected_shape) if expected_shape else None,
        "status_counts": status_counts,
        "pages": pages,
        "episodes": records,
    }
    (output / "preview_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"episode_count": len(episodes), "status_counts": status_counts}))
    print(f"report: {output / 'preview_audit.json'}")
    print(f"review pages: {len(pages)}")
    return 1 if status_counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
