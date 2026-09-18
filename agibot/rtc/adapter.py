"""Convert committed G2 commands back to physical-space GR00T prefix fields."""

from __future__ import annotations

import json
from pathlib import Path

from agibot.tools.g2_gr00t_shadow_adapter import build_policy_observation, make_right_eef_state
import numpy as np
from PIL import Image


def targets_to_prefix(targets: np.ndarray) -> dict[str, np.ndarray]:
    """No filtering, clipping, gripper inversion or re-normalization here.

    These are physical absolute commands. The RTC policy encodes them relative
    to the NEW observation using the checkpoint's own state/action processor.
    """
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 8 or not 0 < len(targets) < 16:
        raise ValueError("prefix must contain 1..15 G2 XYZ+XYZW+gripper rows")
    if not np.isfinite(targets).all():
        raise ValueError("non-finite prefix")
    if np.any(targets[:, 7] < -0.7851) or np.any(targets[:, 7] > 0.0001):
        raise ValueError("prefix jaw target outside native-radian command range")
    return {
        "right_eef": np.stack([make_right_eef_state(row[:7]) for row in targets])[None],
        "right_gripper": targets[None, :, 7:8].astype(np.float32),
    }


def load_saved_observation(directory: Path, prompt: str) -> dict:
    """Read a previously captured observation only; never connect to a robot."""
    metadata = json.loads((directory / "metadata.json").read_text())
    images = []
    for key in ("head_color", "hand_right"):
        candidates = [directory / f"{key}{suffix}" for suffix in (".jpg", ".png")]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"missing saved {key} image")
        with Image.open(path) as source:
            images.append(np.asarray(source.convert("RGB"), dtype=np.uint8))
    return build_policy_observation(
        *images,
        metadata["right_eef_xyz_quaternion_xyzw"],
        metadata["right_gripper"]["training_position"],
        prompt,
    )
