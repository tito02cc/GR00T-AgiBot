#!/usr/bin/env python3
"""Generate dual-camera temporal storyboards for xichong visual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


AGIBOT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = AGIBOT_ROOT / "gr00t_data/xichong_right_single_grasp_300"
DEFAULT_RAW = AGIBOT_ROOT / "data/xichong_right_single_grasp"
CAMERAS = ("head_color", "hand_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--episodes-per-page", type=int, default=5)
    parser.add_argument("--cell-width", type=int, default=160)
    parser.add_argument("--cell-height", type=int, default=120)
    parser.add_argument(
        "--columns",
        type=int,
        default=12,
        help="Sample columns per temporal block; additional samples wrap below",
    )
    parser.add_argument(
        "--episode-indices",
        help="Optional comma-separated subset episode indices (for targeted review)",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sampled_indices(length: int, count: int) -> list[int]:
    return np.rint(np.linspace(0, length - 1, min(length, count))).astype(int).tolist()


def labelled_frame(path: Path, width: int, height: int, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, height - 19), (width, height), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (4, height - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def episode_board(
    raw_root: Path,
    source_episode: str,
    subset_index: int,
    length: int,
    sample_count: int,
    cell_width: int,
    cell_height: int,
    columns: int,
) -> tuple[np.ndarray, dict]:
    source_dir = raw_root / source_episode
    indices = sampled_indices(length, sample_count)
    with np.load(source_dir / "arrays.npz") as arrays:
        states = arrays["states"]
        lift = float(states[-1, 10] - states[0, 10])
        max_lift = float(states[:, 10].max() - states[0, 10])
        grip_start = float(states[0, 15])
        grip_end = float(states[-1, 15])
    title_height = 34
    column_count = min(len(indices), columns)
    block_count = (len(indices) + columns - 1) // columns
    width = column_count * cell_width
    board = np.full(
        (title_height + block_count * 2 * cell_height, width, 3),
        17,
        dtype=np.uint8,
    )
    title = (
        f"subset {subset_index:03d} | {source_episode} | {length} frames | "
        f"final dz={lift:+.3f}m max dz={max_lift:+.3f}m | grip {grip_start:+.3f}->{grip_end:+.3f}"
    )
    cv2.putText(
        board,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (232, 237, 245),
        1,
        cv2.LINE_AA,
    )
    for sample_position, frame_index in enumerate(indices):
        block = sample_position // columns
        column = sample_position % columns
        for camera_row, camera in enumerate(CAMERAS):
            label = f"{camera} f={frame_index} t={frame_index / 10:.1f}s"
            frame = labelled_frame(
                source_dir / "images" / f"{camera}_{frame_index:06d}.jpg",
                cell_width,
                cell_height,
                label,
            )
            y0 = title_height + (block * 2 + camera_row) * cell_height
            x0 = column * cell_width
            board[y0 : y0 + cell_height, x0 : x0 + cell_width] = frame
    return board, {
        "subset_episode_index": subset_index,
        "source_episode": source_episode,
        "length": length,
        "sampled_frame_indices": indices,
        "final_lift_m": lift,
        "max_lift_m": max_lift,
        "gripper_start": grip_start,
        "gripper_end": grip_end,
    }


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    raw_root = args.raw_root.resolve()
    output = args.output.resolve()
    if args.samples < 3 or args.episodes_per_page < 1 or args.columns < 1:
        raise ValueError(
            "samples must be >= 3; episodes-per-page and columns must be positive"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    mappings = read_json(dataset / "meta/source_episode_map.json")["episodes"]
    episodes = [
        json.loads(line)
        for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if args.episode_indices:
        selected_indices = []
        for token in args.episode_indices.split(","):
            subset_index = int(token.strip())
            if not 0 <= subset_index < len(episodes):
                raise ValueError(f"episode index out of range: {subset_index}")
            if subset_index not in selected_indices:
                selected_indices.append(subset_index)
    else:
        selected_indices = list(range(len(episodes)))
    records = []
    page_records = []
    for page_start in range(0, len(selected_indices), args.episodes_per_page):
        boards = []
        page_episodes = []
        page_indices = selected_indices[page_start : page_start + args.episodes_per_page]
        for subset_index in page_indices:
            board, record = episode_board(
                raw_root,
                mappings[subset_index]["source_episode"],
                subset_index,
                int(episodes[subset_index]["length"]),
                args.samples,
                args.cell_width,
                args.cell_height,
                args.columns,
            )
            boards.append(board)
            records.append(record)
            page_episodes.append(subset_index)
        page = np.vstack(boards)
        page_index = page_start // args.episodes_per_page
        page_path = output / f"review_page_{page_index:03d}.jpg"
        if not cv2.imwrite(str(page_path), page, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"failed to write {page_path}")
        page_records.append({"page": page_path.name, "episodes": page_episodes})
    manifest = {
        "dataset": str(dataset),
        "raw_root": str(raw_root),
        "samples_per_episode": args.samples,
        "episodes_per_page": args.episodes_per_page,
        "pages": page_records,
        "episodes": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(page_records)} pages for {len(records)} episodes: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
