#!/usr/bin/env python3
"""Compare a head-camera image with this task's training scene geometry.

This reports only the relative image geometry of two visible rack markers.
It never connects to GDK, changes the model input, or commands the robot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REFERENCE = Path(__file__).with_name("scene_geometry_reference.json")


def inspect_image(image: np.ndarray, reference: dict) -> dict:
    expected_shape = (reference["image_height"], reference["image_width"])
    if image is None or image.shape[:2] != expected_shape:
        raise ValueError(f"expected {expected_shape} head-camera image")
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, reference["dictionary"])
    )
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    found = {} if ids is None else {
        int(tag): corner.reshape(4, 2) for corner, tag in zip(corners, ids.flatten())
    }
    wanted = [int(tag) for tag in reference["tag_ids"]]
    missing = [tag for tag in wanted if tag not in found]
    if missing:
        return {"available": False, "missing_tag_ids": missing,
                "diagnostic_only": True, "robot_motion_changed": False}

    centers = {tag: found[tag].mean(axis=0) for tag in wanted}
    spacing = float(np.linalg.norm(centers[wanted[0]] - centers[wanted[1]]))
    median = float(reference["training_spacing_px"]["median"])
    deviation = spacing / median - 1.0
    return {
        "available": True,
        "tag_ids": wanted,
        "tag_centers_px": {str(tag): centers[tag].tolist() for tag in wanted},
        "spacing_px": spacing,
        "training_median_spacing_px": median,
        "relative_deviation": deviation,
        "large_scene_scale_difference": abs(deviation) > reference["large_deviation_fraction"],
        "diagnostic_only": True,
        "robot_motion_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path,
                        help="640x480 head_color preflight JPEG")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    result = inspect_image(image, reference)
    result["image"] = str(args.image.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
