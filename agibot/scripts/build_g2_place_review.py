#!/usr/bin/env python3
"""Build raw release/withdraw contact sheets (not an automatic success classifier)."""

import argparse
import html
import json
from pathlib import Path

from convert_xichong_right_single_grasp import load_frames, write_json
import numpy as np
from PIL import Image, ImageDraw
from prepare_g2_right_place_training import read_manifest


def pick_risk_and_time(names, rows, count):
    chosen = []
    for key, reverse in (
        ("step_translation_max_m", True),
        ("step_rotation_max_rad", True),
        ("terminal_stability_m", True),
        ("withdraw_m", False),
    ):
        name = sorted(names, key=lambda n: rows[n]["metrics"][key], reverse=reverse)[0]
        if name not in chosen:
            chosen.append(name)
    # Include the last collection time before filling duplicate slots.
    indices = list(np.linspace(0, len(names) - 1, max(2, count - len(chosen)), dtype=int))
    indices.extend(range(len(names)))
    for index in indices:
        if names[index] not in chosen:
            chosen.append(names[index])
        if len(chosen) >= count:
            break
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--heldout-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = {r["episode"]: r for r in json.loads(args.audit.read_text())["episodes"]}
    train, heldout = read_manifest(args.train_manifest), read_manifest(args.heldout_manifest)
    sample = pick_risk_and_time(sorted(train), rows, 12) + pick_risk_and_time(
        sorted(heldout), rows, 6
    )
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = []
    # All selected episodes get a sheet, not just the 18-episode inspection sample.
    for name in train + heldout:
        frames = load_frames(args.raw_root / name / "frames.jsonl")
        release = rows[name]["release_index"]
        indices = [
            0,
            max(0, release - 3),
            max(0, release - 1),
            release,
            min(len(frames) - 1, release + 5),
            len(frames) - 1,
        ]
        canvas = Image.new("RGB", (1440, 424), "#15202b")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 4),
            f"{name} / {'train' if name in train else 'heldout'} / release={release}",
            fill="white",
        )
        for col, index in enumerate(indices):
            frame = frames[index]
            draw.text(
                (col * 240 + 4, 24),
                f"f{index:03d} jaw={frame['right_gripper']['position']:.3f}",
                fill="white",
            )
            for row, camera in enumerate(("head_color", "hand_right")):
                with Image.open(args.raw_root / name / frame["images"][camera]) as im:
                    canvas.paste(im.resize((240, 180)), (col * 240, 44 + row * 190))
        canvas.save(args.output / f"{name}.jpg", quality=92)
        evidence.append(
            {
                "episode": name,
                "split": "train" if name in train else "heldout",
                "indices": indices,
                "sampled_for_inspection": name in sample,
                "visual_verdict": "not_reviewed",
            }
        )
    for page in range(0, len(sample), 3):
        canvas = Image.new("RGB", (1440, 1272))
        for row, name in enumerate(sample[page : page + 3]):
            with Image.open(args.output / f"{name}.jpg") as im:
                canvas.paste(im, (0, row * 424))
        canvas.save(args.output / f"sample_{page // 3:02d}.jpg", quality=94)
    body = "\n".join(
        f'<section><h2>{html.escape(n)}</h2><img loading="lazy" width="1440" src="{n}.jpg"></section>'
        for n in train + heldout
    )
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Placement source review</title>'
        "<style>body{background:#15202b;color:white;font-family:sans-serif}img{max-width:100%;height:auto}</style>"
        "<h1>Source release / withdraw review</h1><p>Head row, wrist row. "
        "Contact sheets do not automatically prove placement success.</p>" + body
    )
    write_json(
        args.output / "review_manifest.json",
        {
            "sample": sample,
            "episodes": evidence,
            "selection": "within each split: motion/stability/withdraw extrema plus chronological coverage",
            "review_status": "awaiting_visual_inspection",
        },
    )


if __name__ == "__main__":
    main()
