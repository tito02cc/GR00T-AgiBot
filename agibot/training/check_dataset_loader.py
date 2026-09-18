#!/usr/bin/env python3
"""Load a task modality and verify a converted dataset with the official loader."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--embodiment-tag", required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("task_modality_config", args.modality_config)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import modality config: {args.modality_config}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        modality = MODALITY_CONFIGS[args.embodiment_tag.lower()]
    except KeyError as exc:
        raise KeyError(
            f"modality did not register embodiment {args.embodiment_tag!r}; "
            f"available={sorted(MODALITY_CONFIGS)}"
        ) from exc
    loader = LeRobotEpisodeLoader(args.dataset.resolve(), modality)
    if len(loader) != args.expected_episodes:
        raise ValueError(
            f"loader episodes={len(loader)}, expected={args.expected_episodes}"
        )
    expected_stats = {"action", "relative_action", "state"}
    actual_stats = set(loader.get_dataset_statistics())
    if actual_stats != expected_stats:
        raise ValueError(f"dataset statistics keys={actual_stats}, expected={expected_stats}")
    print(f"official loader episodes={len(loader)} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
