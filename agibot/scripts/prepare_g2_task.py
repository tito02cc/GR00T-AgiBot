#!/usr/bin/env python3
"""Run a task's data preparation config; relative paths are repository-relative."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
PATH_KEYS = (
    "raw_root",
    "train_manifest",
    "heldout_manifest",
    "profile",
    "modality_config",
    "output_root",
    "report_dir",
)
VALUE_KEYS = (
    "workers",
    "crf",
    "expected_train_count",
    "expected_heldout_count",
    "normalization_bounds",
)


def build_command(config: dict, stage: str) -> list[str]:
    command = [sys.executable, str(REPO / "agibot/scripts/prepare_g2_right_place_training.py")]
    if config.get("pipeline") != "g2_right_place":
        raise ValueError(
            "This entry point implements g2_right_place, not another task/arm contract"
        )
    for key in PATH_KEYS:
        value = Path(config[key]).expanduser()
        if not value.is_absolute():
            value = REPO / value
        command.extend(["--" + key.replace("_", "-"), str(value.resolve())])
    for key in VALUE_KEYS:
        if key in config:
            command.extend(["--" + key.replace("_", "-"), str(config[key])])
    return command + ["--stage", stage]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-config", type=Path, required=True)
    p.add_argument("--stage", choices=("all", "convert", "verify"), default="all")
    args = p.parse_args()
    config = json.loads(args.task_config.read_text())
    subprocess.run(build_command(config, args.stage), cwd=REPO, check=True)


if __name__ == "__main__":
    main()
