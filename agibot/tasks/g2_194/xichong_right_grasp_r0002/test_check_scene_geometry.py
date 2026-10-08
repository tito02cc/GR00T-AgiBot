import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("grasp_scene_geometry", HERE / "check_scene_geometry.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
REFERENCE = json.loads((HERE / "scene_geometry_reference.json").read_text())


def synthetic_scene(spacing: int) -> np.ndarray:
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for tag, x in ((3, 200), (4, 200 + spacing)):
        marker = cv2.aruco.generateImageMarker(dictionary, tag, 40)
        image[160:200, x:x + 40] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return image


def test_scene_geometry_is_advisory_and_never_a_robot_command():
    reference = MODULE.inspect_image(synthetic_scene(170), REFERENCE)
    distant = MODULE.inspect_image(synthetic_scene(159), REFERENCE)
    assert reference["available"] and not reference["large_scene_scale_difference"]
    assert distant["available"] and distant["large_scene_scale_difference"]
    assert distant["relative_deviation"] < 0
    assert distant["diagnostic_only"] and not distant["robot_motion_changed"]


def test_missing_tags_cannot_be_treated_as_matched_scene():
    result = MODULE.inspect_image(np.full((480, 640, 3), 255, dtype=np.uint8), REFERENCE)
    assert not result["available"]
    assert result["missing_tag_ids"] == [3, 4]
