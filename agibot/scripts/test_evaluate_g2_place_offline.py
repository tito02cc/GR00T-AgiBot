"""Pure offline checks; no server or robot is started."""

from agibot.scripts.evaluate_g2_place_offline import (
    check_decoded,
    check_live_observation_adapter,
    first_open,
)
from agibot.tools.g2_gr00t_shadow_adapter import build_policy_observation
import numpy as np
import pytest


def action():
    return {
        "right_eef": np.tile([0.5, -0.2, 0.9, 1, 0, 0, 0, 1, 0], (1, 16, 1)),
        "right_gripper": np.linspace(0, -0.785, 16).reshape(1, 16, 1),
    }


def test_complete_chunk_preserves_every_in_range_gripper_value():
    data = action()
    result = check_decoded(data)
    np.testing.assert_allclose(result[:, 7], data["right_gripper"][0, :, 0], atol=1e-7)


def test_short_chunk_rejected():
    data = action()
    data["right_eef"] = data["right_eef"][:, :8]
    with pytest.raises(ValueError, match="H16"):
        check_decoded(data)


def test_nan_rejected():
    data = action()
    data["right_gripper"][0, 3, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        check_decoded(data)


def test_open_timing_is_not_closed_at_start():
    assert first_open([0, -0.1, -0.6, -0.72, -0.785]) == 3
    assert first_open([0, -0.6]) is None


def test_native_live_input_matches_dataset_input_contract():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = build_policy_observation(image, image, [0.5, -0.2, 0.9, 0, 0, 0, 1], -0.3, "place")
    check_live_observation_adapter(obs)
