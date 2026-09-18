#!/usr/bin/env python3
"""Reload a saved processor and compare its live bounds to the training dataset."""

import argparse
import json
from pathlib import Path

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--embodiment", default="new_embodiment")
    args = parser.parse_args()
    expected = json.loads((args.dataset / "meta/training_preprocessing.json").read_text())
    processor = Gr00tN1d7Processor.from_pretrained(
        args.model, transformers_loading_kwargs={"local_files_only": True}
    )
    for key in ("use_percentiles", "use_relative_action", "clip_outliers"):
        if (
            getattr(processor, key) != expected[key]
            or getattr(processor.state_action_processor, key) != expected[key]
        ):
            raise ValueError(f"Reloaded processor conflicts with dataset: {key}")
    loader = LeRobotEpisodeLoader(args.dataset, processor.modality_configs[args.embodiment])
    stats = loader.get_dataset_statistics()
    actual = processor.state_action_processor.norm_params[args.embodiment]
    source_bounds = ("q01", "q99") if expected["use_percentiles"] else ("min", "max")
    for key in processor.modality_configs[args.embodiment]["state"].modality_keys:
        for target, source in zip(("min", "max"), source_bounds, strict=True):
            np.testing.assert_allclose(
                actual["state"][key][target], stats["state"][key][source], rtol=0, atol=1e-7
            )
    print(
        json.dumps(
            {
                "status": "PASS",
                "reloaded_processor_use_percentiles": processor.use_percentiles,
                "live_state_bounds_match_training_stats": True,
                "dataset_episodes": len(loader),
                "scope": "processor reload and live state bounds; no model forward",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
