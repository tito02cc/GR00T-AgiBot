#!/usr/bin/env python3
"""Read-only full JPEG decoding of named raw episodes; no hashes or source edits.

This validates image encoding, dimensions and per-frame paths, not physical task
success. Left images can be preserved in transfer without being policy inputs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from PIL import Image


def decode_episode(root: str, name: str, cameras: tuple[str, ...]) -> dict:
    result = {"episode": name, "status": "pass", "decoded_images": 0, "errors": []}
    directory = Path(root) / name
    try:
        frames = [
            json.loads(line)
            for line in (directory / "frames.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if not frames:
            raise ValueError("empty episode")
        for index, frame in enumerate(frames):
            if frame["frame_index"] != index:
                raise ValueError(f"frame {index}: index mismatch")
            for camera in cameras:
                relative = f"images/{camera}_{index:06d}.jpg"
                if frame["images"].get(camera) != relative:
                    raise ValueError(f"frame {index}: {camera} image mapping mismatch")
                path = directory / relative
                if not path.resolve().is_relative_to(directory.resolve()):
                    raise ValueError(f"image points outside episode: {relative}")
                with Image.open(path) as image:
                    if image.format != "JPEG" or image.mode != "RGB" or image.size != (640, 480):
                        raise ValueError(
                            f"{relative}: expected JPEG RGB 640x480, "
                            f"got {image.format} {image.mode} {image.size}"
                        )
                    image.load()
                result["decoded_images"] += 1
        result["frames"] = len(frames)
    except Exception as error:  # One malformed episode must not stop the batch.
        result["status"] = "fail"
        result["errors"].append(f"{type(error).__name__}: {error}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episode-list", type=Path)
    selection.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cameras", nargs="+", default=["head_color", "hand_right"])
    args = parser.parse_args()
    root, destination = args.raw_root.resolve(), args.report.resolve()
    if destination.is_relative_to(root):
        parser.error("report must be outside raw-root")
    if args.workers < 1 or args.workers > 8:
        parser.error("workers must be between 1 and 8")
    cameras = tuple(args.cameras)
    if len(set(cameras)) != len(cameras) or not set(cameras) <= {
        "head_color",
        "hand_right",
        "hand_left",
    }:
        parser.error("invalid or duplicate cameras")
    if args.all_episodes:
        names = sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and re.fullmatch(r"episode_\d{6}", path.name)
        )
    else:
        names = [
            line.strip() for line in args.episode_list.read_text().splitlines() if line.strip()
        ]
    if not names or len(names) != len(set(names)):
        parser.error("episode-list must be nonempty and unique")
    if any(re.fullmatch(r"episode_\d{6}", name) is None for name in names):
        parser.error("invalid episode directory name")
    rows = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(decode_episode, str(root), name, cameras): name for name in names}
        for done, future in enumerate(as_completed(jobs), 1):
            row = future.result()
            rows[row["episode"]] = row
            if done % 20 == 0 or done == len(names):
                print(f"decoded_episodes={done}/{len(names)}", flush=True)
    failures = sum(row["status"] == "fail" for row in rows.values())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "full JPEG decoding and frame mapping; not visual task-success review",
        "source_root": str(root),
        "episode_list": str(args.episode_list.resolve()) if args.episode_list else None,
        "selection": "all_episodes" if args.all_episodes else "episode_list",
        "cameras": cameras,
        "counts": {
            "episodes": len(names),
            "passed": len(names) - failures,
            "failed": failures,
            "decoded_images": sum(row["decoded_images"] for row in rows.values()),
        },
        "episodes": [rows[name] for name in names],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(report["counts"]), flush=True)
    print(f"report={destination}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
