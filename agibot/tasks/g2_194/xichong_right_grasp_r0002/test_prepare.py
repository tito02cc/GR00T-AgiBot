import copy
import importlib.util
from pathlib import Path
import sys

import pytest

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
spec = importlib.util.spec_from_file_location("grasp_prepare", TASK / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def frames():
    return [{"frame_index": i, "prompt": "grasp and lift", "images": {
        c: f"images/{c}_{i:06d}.jpg" for c in prepare.CAMERAS
    }} for i in range(2)]


def test_correct_sequence_and_prompt():
    prepare.check_frame_contract(frames(), "grasp and lift")


@pytest.mark.parametrize("mutation", ["index", "prompt", "camera"])
def test_no_silent_reindex_relabel_or_camera_swap(mutation):
    data = copy.deepcopy(frames())
    if mutation == "index":
        data[1]["frame_index"] = 2
    elif mutation == "prompt":
        data[1]["prompt"] = "place and release"
    else:
        data[1]["images"]["hand_right"] = "images/hand_left_000001.jpg"
    with pytest.raises(ValueError):
        prepare.check_frame_contract(data, "grasp and lift")


def test_manifest_overlap_is_rejected():
    with pytest.raises(ValueError):
        prepare.check_splits(["episode_000001"], ["episode_000001"])


def test_empty_frames_rejected():
    with pytest.raises(ValueError):
        prepare.check_frame_contract([], "grasp and lift")


@pytest.mark.parametrize("mutation", [None, "freq", "frame", "gripper"])
def test_collection_encoding_checked(mutation):
    config = {
        "args": {"freq": 10., "img_w": 640, "img_h": 480, "arm_mode": "right"},
        "frames": {"pose_frame": "base_link_tf", "action_mode": "next_delta"},
        "gripper_mapping": {"mode": "official_norm_open_negative_closed0",
                            "official_norm_open": -0.785, "official_norm_closed": 0.0},
    }
    if mutation is None:
        prepare.check_capture_config(config)
        return
    if mutation == "freq":
        config["args"]["freq"] = 20.
    elif mutation == "frame":
        config["frames"]["pose_frame"] = "tool"
    else:
        config["gripper_mapping"]["official_norm_closed"] = 120.
    with pytest.raises(ValueError):
        prepare.check_capture_config(config)
