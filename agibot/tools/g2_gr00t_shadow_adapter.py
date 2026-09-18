#!/usr/bin/env python3
"""Pure, non-actuating Agibot G2 <-> GR00T N1.7 protocol adapter.

This module deliberately contains no ``agibot_gdk`` import and no robot command
call.  It is the shared, testable boundary used before a live G2 client is
allowed to exist:

* G2 base-link XYZ + XYZW quaternion -> GR00T XYZ + row-major Rot6D state.
* G2 gripper feedback millimetres -> the training convention [-0.785, 0].
* GR00T absolute physical-space action chunks -> G2 XYZ + XYZW targets.
* Strict construction of the PolicyClient observation batch/time dimensions.

GR00T N1.7's processor already denormalizes actions and converts configured
relative EEF predictions back to absolute targets.  Consequently
``decode_action_chunk`` must receive the PolicyClient output, not raw model
latents or normalized actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


GRIPPER_TRAINING_OPEN = -0.785
GRIPPER_TRAINING_CLOSED = 0.0
GRIPPER_G2_RAW_CLOSED_MM = 120.0


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} shape={array.shape}, expected={shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return array


def normalize_quaternion_xyzw(quaternion: Any) -> np.ndarray:
    """Return a finite unit quaternion in SciPy/G2 XYZW order."""
    quat = _finite_array(quaternion, (4,), "quaternion_xyzw")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError("quaternion norm is zero")
    return (quat / norm).astype(np.float32)


def quaternion_xyzw_to_rot6d(quaternion: Any) -> np.ndarray:
    """Convert XYZW quaternion to GR00T's first-two-matrix-rows Rot6D."""
    quat = normalize_quaternion_xyzw(quaternion)
    return Rotation.from_quat(quat).as_matrix()[:2, :].reshape(6).astype(np.float32)


def rot6d_to_rotation_matrix(rot6d: Any) -> np.ndarray:
    """Convert GR00T row-major Rot6D to the nearest proper rotation matrix."""
    rows = _finite_array(rot6d, (6,), "rot6d").reshape(2, 3)
    row0_norm = float(np.linalg.norm(rows[0]))
    if row0_norm < 1e-8:
        raise ValueError("rot6d first row has zero norm")
    row0 = rows[0] / row0_norm
    row1_residual = rows[1] - np.dot(row0, rows[1]) * row0
    row1_norm = float(np.linalg.norm(row1_residual))
    if row1_norm < 1e-8:
        raise ValueError("rot6d rows are collinear")
    row1 = row1_residual / row1_norm
    row2 = np.cross(row0, row1)
    matrix = np.stack((row0, row1, row2), axis=0)
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("rot6d reconstruction is not orthogonal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-6):
        raise ValueError("rot6d reconstruction is not a proper rotation")
    return matrix.astype(np.float32)


def rot6d_to_quaternion_xyzw(rot6d: Any) -> np.ndarray:
    """Convert GR00T row-major Rot6D to a normalized G2 XYZW quaternion."""
    matrix = rot6d_to_rotation_matrix(rot6d)
    return normalize_quaternion_xyzw(Rotation.from_matrix(matrix).as_quat())


def g2_gripper_raw_mm_to_training(position_raw_mm: float) -> np.float32:
    """Map G2 feedback: 0 mm=open, 120 mm=closed, to [-0.785, 0]."""
    value = float(position_raw_mm)
    if not math.isfinite(value):
        raise ValueError("gripper raw feedback contains NaN/Inf")
    closed_fraction = np.clip(value, 0.0, GRIPPER_G2_RAW_CLOSED_MM) / GRIPPER_G2_RAW_CLOSED_MM
    normalized = GRIPPER_TRAINING_OPEN * (1.0 - closed_fraction)
    return np.float32(normalized)


def clip_training_gripper_command(position: float) -> np.float32:
    """Fail on non-finite output and clip only to the demonstrated actuator range."""
    value = float(position)
    if not math.isfinite(value):
        raise ValueError("gripper action contains NaN/Inf")
    return np.float32(np.clip(value, GRIPPER_TRAINING_OPEN, GRIPPER_TRAINING_CLOSED))


def make_right_eef_state(xyz_quaternion_xyzw: Any) -> np.ndarray:
    """Create the 9D right EEF state used by this N1.7 modality config."""
    pose = _finite_array(xyz_quaternion_xyzw, (7,), "xyz_quaternion_xyzw")
    return np.concatenate((pose[:3], quaternion_xyzw_to_rot6d(pose[3:]))).astype(np.float32)


def _validate_rgb_image(image: Any, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"{name} dtype={array.dtype}, expected=uint8")
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} shape={array.shape}, expected=(H,W,3)")
    if array.shape[:2] != (480, 640):
        raise ValueError(f"{name} shape={array.shape}, expected=(480,640,3)")
    return np.ascontiguousarray(array)


def build_policy_observation(
    head_color_rgb: Any,
    hand_right_rgb: Any,
    right_pose_xyz_quaternion_xyzw: Any,
    right_gripper_training: float,
    prompt: str,
) -> dict[str, Any]:
    """Build one strict (B=1,T=1) observation for ``PolicyClient.get_action``."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    gripper = float(right_gripper_training)
    if not math.isfinite(gripper):
        raise ValueError("right_gripper_training contains NaN/Inf")
    if not GRIPPER_TRAINING_OPEN - 1e-6 <= gripper <= GRIPPER_TRAINING_CLOSED + 1e-6:
        raise ValueError(
            f"right_gripper_training={gripper} is outside "
            f"[{GRIPPER_TRAINING_OPEN}, {GRIPPER_TRAINING_CLOSED}]"
        )
    head = _validate_rgb_image(head_color_rgb, "head_color_rgb")
    wrist = _validate_rgb_image(hand_right_rgb, "hand_right_rgb")
    eef = make_right_eef_state(right_pose_xyz_quaternion_xyzw)
    return {
        "video": {
            "head_color": head[None, None, ...],
            "hand_right": wrist[None, None, ...],
        },
        "state": {
            "right_eef": eef[None, None, ...],
            "right_gripper": np.asarray([[[gripper]]], dtype=np.float32),
        },
        "language": {
            "annotation.human.task_description": [[prompt]],
        },
    }


def _action_array(action: dict[str, Any], key: str, dim: int) -> np.ndarray:
    value = action.get(key, action.get(f"action.{key}"))
    if value is None:
        raise KeyError(f"missing action key '{key}'")
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"{key} batch={array.shape[0]}, only batch=1 is supported")
        array = array[0]
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"{key} shape={array.shape}, expected=(T,{dim}) or (1,T,{dim})")
    if not np.isfinite(array).all():
        raise ValueError(f"{key} contains NaN/Inf")
    return array


def decode_action_chunk(action: dict[str, Any]) -> np.ndarray:
    """Decode physical absolute GR00T output to G2 ``XYZ,XYZW,gripper`` targets.

    Returns an array of shape ``(T, 8)``.  The gripper is clipped to the range
    demonstrated during data collection.  EEF pose values are not workspace or
    velocity clipped here; live execution must pass an independently configured
    fail-closed safety layer using limits approved for the specific robot cell.
    """
    eef = _action_array(action, "right_eef", 9)
    gripper = _action_array(action, "right_gripper", 1)
    if eef.shape[0] != gripper.shape[0]:
        raise ValueError(f"action horizon mismatch: right_eef={eef.shape[0]}, gripper={gripper.shape[0]}")
    targets = np.empty((eef.shape[0], 8), dtype=np.float32)
    targets[:, :3] = eef[:, :3]
    for index, row in enumerate(eef):
        targets[index, 3:7] = rot6d_to_quaternion_xyzw(row[3:9])
    targets[:, 7] = [clip_training_gripper_command(value) for value in gripper[:, 0]]
    return targets


@dataclass(frozen=True)
class ShadowTargetCheck:
    translation_step_m: float
    rotation_step_deg: float
    gripper_target: float


def check_shadow_target(
    current_pose_xyz_quaternion_xyzw: Any,
    target_xyz_quaternion_xyzw_gripper: Any,
) -> ShadowTargetCheck:
    """Measure a target without approving or executing it.

    Deliberately returns measurements rather than a safety boolean: workspace,
    speed, acceleration and allowed-step thresholds must come from the G2 SDK,
    robot specification and the physical cell validation, not from this dataset.
    """
    current = _finite_array(current_pose_xyz_quaternion_xyzw, (7,), "current_pose")
    target = _finite_array(target_xyz_quaternion_xyzw_gripper, (8,), "target")
    q0 = normalize_quaternion_xyzw(current[3:7])
    q1 = normalize_quaternion_xyzw(target[3:7])
    dot = float(np.clip(abs(np.dot(q0, q1)), -1.0, 1.0))
    return ShadowTargetCheck(
        translation_step_m=float(np.linalg.norm(target[:3] - current[:3])),
        rotation_step_deg=float(math.degrees(2.0 * math.acos(dot))),
        gripper_target=float(clip_training_gripper_command(target[7])),
    )
