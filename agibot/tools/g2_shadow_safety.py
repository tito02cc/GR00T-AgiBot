#!/usr/bin/env python3
"""Fail-closed, non-actuating safety checks for G2 GR00T shadow targets.

The limits consumed here are deliberately marked shadow-only.  This module has
no G2 SDK import and cannot command a robot.  It screens decoded physical
targets before a future live adapter is allowed to exist.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from agibot.tools.g2_gr00t_shadow_adapter import (
    GRIPPER_TRAINING_OPEN,
    normalize_quaternion_xyzw,
)


@dataclass(frozen=True)
class ShadowSafetyLimits:
    workspace_min_xyz_m: np.ndarray
    workspace_max_xyz_m: np.ndarray
    max_translation_step_m: float
    max_rotation_step_deg: float
    gripper_close_threshold: float
    closure_min_xyz_m: np.ndarray
    closure_max_xyz_m: np.ndarray
    approval_status: str
    live_execution_enabled: bool


@dataclass(frozen=True)
class ShadowChunkDecision:
    chunk_allowed: bool
    safe_targets: np.ndarray
    violations: tuple[dict[str, Any], ...]
    gripper_gate_events: tuple[dict[str, Any], ...]
    max_translation_step_m: float
    max_rotation_step_deg: float


def _vector3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite XYZ vector, got {array}")
    return array


def load_shadow_safety_limits(path: str | Path) -> ShadowSafetyLimits:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != "xichong_g2_shadow_safety_v1":
        raise ValueError("unsupported shadow safety schema")
    approval = str(raw.get("approval_status", ""))
    if "SHADOW_ONLY" not in approval:
        raise ValueError("safety config must be explicitly marked SHADOW_ONLY")
    if bool(raw.get("live_execution_enabled", True)):
        raise ValueError("shadow safety config must set live_execution_enabled=false")
    workspace = raw["workspace"]
    motion = raw["motion"]
    gripper = raw["gripper_gate"]
    limits = ShadowSafetyLimits(
        workspace_min_xyz_m=_vector3(workspace["min_xyz_m"], "workspace.min_xyz_m"),
        workspace_max_xyz_m=_vector3(workspace["max_xyz_m"], "workspace.max_xyz_m"),
        max_translation_step_m=float(motion["max_translation_step_m"]),
        max_rotation_step_deg=float(motion["max_rotation_step_deg"]),
        gripper_close_threshold=float(gripper["close_threshold_training_units"]),
        closure_min_xyz_m=_vector3(gripper["closure_min_xyz_m"], "closure_min_xyz_m"),
        closure_max_xyz_m=_vector3(gripper["closure_max_xyz_m"], "closure_max_xyz_m"),
        approval_status=approval,
        live_execution_enabled=False,
    )
    if np.any(limits.workspace_min_xyz_m >= limits.workspace_max_xyz_m):
        raise ValueError("workspace min must be less than max")
    if np.any(limits.closure_min_xyz_m >= limits.closure_max_xyz_m):
        raise ValueError("closure min must be less than max")
    if limits.max_translation_step_m <= 0 or limits.max_rotation_step_deg <= 0:
        raise ValueError("motion limits must be positive")
    return limits


def _rotation_step_deg(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    q0 = normalize_quaternion_xyzw(quat_a).astype(np.float64)
    q1 = normalize_quaternion_xyzw(quat_b).astype(np.float64)
    dot = float(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def evaluate_shadow_chunk(
    current_pose_xyz_quaternion_xyzw: Any,
    current_gripper_training: float,
    decoded_targets: Any,
    limits: ShadowSafetyLimits,
    *,
    execution_horizon: int,
) -> ShadowChunkDecision:
    """Screen a target chunk without approving or executing robot motion.

    Workspace or per-step motion violations reject the entire chunk.  A close
    request outside the demonstrated closure envelope is replaced with the
    previous gripper command and recorded as a gate event.
    """
    current = np.asarray(current_pose_xyz_quaternion_xyzw, dtype=np.float64)
    targets = np.asarray(decoded_targets, dtype=np.float64)
    if current.shape != (7,) or not np.isfinite(current).all():
        raise ValueError("current pose must be finite XYZ+XYZW")
    if targets.ndim != 2 or targets.shape[1] != 8 or not np.isfinite(targets).all():
        raise ValueError("decoded_targets must be finite (T,8)")
    if execution_horizon <= 0 or execution_horizon > len(targets):
        raise ValueError("execution_horizon must be within decoded target length")
    gripper = float(current_gripper_training)
    if not np.isfinite(gripper):
        raise ValueError("current gripper is non-finite")

    screened = targets[:execution_horizon].copy()
    violations: list[dict[str, Any]] = []
    gate_events: list[dict[str, Any]] = []
    previous_pose = current.copy()
    previous_gripper = gripper
    max_translation = 0.0
    max_rotation = 0.0

    for index, target in enumerate(screened):
        xyz = target[:3]
        translation = float(np.linalg.norm(xyz - previous_pose[:3]))
        rotation = _rotation_step_deg(previous_pose[3:7], target[3:7])
        max_translation = max(max_translation, translation)
        max_rotation = max(max_rotation, rotation)

        outside = np.flatnonzero(
            (xyz < limits.workspace_min_xyz_m) | (xyz > limits.workspace_max_xyz_m)
        )
        if len(outside):
            violations.append(
                {"step": index, "code": "workspace", "xyz_m": xyz.tolist()}
            )
        if translation > limits.max_translation_step_m:
            violations.append(
                {
                    "step": index,
                    "code": "translation_step",
                    "value_m": translation,
                    "limit_m": limits.max_translation_step_m,
                }
            )
        if rotation > limits.max_rotation_step_deg:
            violations.append(
                {
                    "step": index,
                    "code": "rotation_step",
                    "value_deg": rotation,
                    "limit_deg": limits.max_rotation_step_deg,
                }
            )

        requests_close = (
            target[7] >= limits.gripper_close_threshold
            and previous_gripper < limits.gripper_close_threshold
        )
        inside_closure = bool(
            np.all(xyz >= limits.closure_min_xyz_m)
            and np.all(xyz <= limits.closure_max_xyz_m)
        )
        if requests_close and not inside_closure:
            screened[index, 7] = previous_gripper
            gate_events.append(
                {
                    "step": index,
                    "code": "close_outside_grasp_envelope",
                    "xyz_m": xyz.tolist(),
                    "requested": float(target[7]),
                    "held": previous_gripper,
                }
            )
        previous_pose = np.concatenate((target[:3], normalize_quaternion_xyzw(target[3:7])))
        previous_gripper = float(screened[index, 7])

    if violations:
        screened[:, :7] = np.tile(current, (execution_horizon, 1))
        screened[:, 7] = GRIPPER_TRAINING_OPEN

    return ShadowChunkDecision(
        chunk_allowed=not violations,
        safe_targets=screened.astype(np.float32),
        violations=tuple(violations),
        gripper_gate_events=tuple(gate_events),
        max_translation_step_m=max_translation,
        max_rotation_step_deg=max_rotation,
    )
