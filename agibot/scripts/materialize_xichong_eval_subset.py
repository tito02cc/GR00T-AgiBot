#!/usr/bin/env python3
"""Materialize the fixed, leakage-free Xichong held-out evaluation subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from select_xichong_training_subset import materialize_subset


AGIBOT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = AGIBOT_ROOT / "gr00t_data/xichong_right_single_grasp"
DEFAULT_OUTPUT = AGIBOT_ROOT / "gr00t_data/xichong_right_single_grasp_eval_heldout12"
DEFAULT_SELECTION_REPORT = AGIBOT_ROOT / "reports/xichong_right_single_grasp_selection_300.json"
DEFAULT_REPORT = AGIBOT_ROOT / "reports/xichong_right_single_grasp_eval_heldout12.json"

# Closest quality-eligible reserve episode to the midpoint of each of the 12
# chronological collection strata.  None occurs in the 300-episode training set.
HELD_OUT_FULL_INDICES = [27, 79, 132, 185, 238, 292, 343, 396, 450, 503, 556, 607]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    selection = json.loads(args.selection_report.resolve().read_text())
    selected = set(selection["selected_full_dataset_indices"])
    reserve = set(selection["reserve_full_dataset_indices"])
    excluded = {
        row["full_dataset_episode_index"] for row in selection["quality_filter"]["excluded"]
    }
    held_out = set(HELD_OUT_FULL_INDICES)

    assert len(held_out) == 12
    assert not held_out & selected, f"held-out/training leakage: {sorted(held_out & selected)}"
    assert not held_out & excluded, f"held-out quality exclusions: {sorted(held_out & excluded)}"
    assert held_out <= reserve, f"held-out indices outside reserve: {sorted(held_out - reserve)}"

    source_map_data = json.loads((source / "meta/source_episode_map.json").read_text())
    source_map = {
        int(row["target_episode_index"]): row for row in source_map_data["episodes"]
    }
    materialize_subset(source, output, HELD_OUT_FULL_INDICES, source_map)

    # The loader requires these files, but normalization at inference is taken
    # from the checkpoint processor.  Copying the source statistics keeps this
    # small evaluation dataset structurally complete without fitting on held-out data.
    for name in ("stats.json", "relative_stats.json"):
        shutil.copy2(source / "meta" / name, output / "meta" / name)

    rows = json.loads((output / "meta/source_episode_map.json").read_text())["episodes"]
    report = {
        "status": "PASS",
        "selection": "nearest quality-eligible training-reserve episode to each chronological-stratum midpoint",
        "chronological_stratum_midpoints": [26, 79, 132, 185, 238, 291, 344, 397, 450, 503, 556, 609],
        "full_dataset_episode_indices": HELD_OUT_FULL_INDICES,
        "subset_episode_indices": list(range(12)),
        "training_overlap": [],
        "quality_excluded_overlap": [],
        "episodes": rows,
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
