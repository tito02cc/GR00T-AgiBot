#!/usr/bin/env python3
"""Pure, non-actuating gate for GR00T right-arm G2 target chunks.

This module constructs the GDK A2D payload shape but never imports GDK and
cannot publish robot commands.  Its bundled configuration is explicitly
shadow-only; passing its output to a robot controller is prohibited.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from agibot.tools.g2_gr00t_shadow_adapter import normalize_quaternion_xyzw


SCHEMA = "g2_groot_right_action_guard_v1"


@dataclass(frozen=True)
class ActionGuardLimits:
    approval_status: str
    live_execution_enabled: bool
    sample_period_s: float
    maximum_horizon: int
    maximum_command_age_s: float
    workspace_min_xyz_m: np.ndarray
    workspace_max_xyz_m: np.ndarray
    maximum_translation_step_m: float
    maximum_rotation_step_deg: float
    maximum_translation_acceleration_m_s2: float
    maximum_rotation_acceleration_deg_s2: float
    gripper_minimum: float
    gripper_maximum: float
    gripper_maximum_step: float
    gripper_closure_threshold: float
    gripper_closure_enabled: bool


@dataclass(frozen=True)
class ActionGuardDecision:
    validation_pass: bool
    guarded_targets: np.ndarray
    violations: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _xyz(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite XYZ vector")
    return result


def load_shadow_action_guard(path: str | Path) -> ActionGuardLimits:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError("unsupported action guard schema")
    approval = str(raw.get("approval_status", ""))
    if "SHADOW_ONLY" not in approval:
        raise ValueError("this loader accepts only explicitly shadow-only configs")
    if bool(raw.get("live_execution_enabled", True)):
        raise ValueError("shadow action guard must set live_execution_enabled=false")
    workspace = raw["workspace"]
    motion = raw["motion"]
    gripper = raw["gripper"]
    limits = ActionGuardLimits(
        approval_status=approval,
        live_execution_enabled=False,
        sample_period_s=_positive(raw["sample_period_s"], "sample_period_s"),
        maximum_horizon=int(raw["maximum_horizon"]),
        maximum_command_age_s=_positive(
            raw["maximum_command_age_s"], "maximum_command_age_s"
        ),
        workspace_min_xyz_m=_xyz(workspace["min_xyz_m"], "workspace min"),
        workspace_max_xyz_m=_xyz(workspace["max_xyz_m"], "workspace max"),
        maximum_translation_step_m=_positive(
            motion["maximum_translation_step_m"], "translation step"
        ),
        maximum_rotation_step_deg=_positive(
            motion["maximum_rotation_step_deg"], "rotation step"
        ),
        maximum_translation_acceleration_m_s2=_positive(
            motion["maximum_translation_acceleration_m_s2"],
            "translation acceleration",
        ),
        maximum_rotation_acceleration_deg_s2=_positive(
            motion["maximum_rotation_acceleration_deg_s2"],
            "rotation acceleration",
        ),
        gripper_minimum=float(gripper["minimum_training_position"]),
        gripper_maximum=float(gripper["maximum_training_position"]),
        gripper_maximum_step=_positive(gripper["maximum_step"], "gripper step"),
        gripper_closure_threshold=float(gripper["closure_threshold"]),
        gripper_closure_enabled=bool(gripper["closure_enabled"]),
    )
    if not 1 <= limits.maximum_horizon <= 16:
        raise ValueError("maximum_horizon must be in 1..16")
    if np.any(limits.workspace_min_xyz_m >= limits.workspace_max_xyz_m):
        raise ValueError("workspace minimum must be below maximum")
    if not limits.gripper_minimum < limits.gripper_maximum:
        raise ValueError("invalid gripper range")
    if not (
        limits.gripper_minimum
        < limits.gripper_closure_threshold
        < limits.gripper_maximum
    ):
        raise ValueError("gripper closure threshold must be inside the range")
    return limits


def _rotation_step_deg(first: np.ndarray, second: np.ndarray) -> float:
    q0 = normalize_quaternion_xyzw(first).astype(np.float64)
    q1 = normalize_quaternion_xyzw(second).astype(np.float64)
    dot = float(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def evaluate_action_chunk(
    current_pose_xyz_quaternion_xyzw: Any,
    current_gripper_training: float,
    decoded_targets: Any,
    limits: ActionGuardLimits,
    *,
    execution_horizon: int,
    command_age_s: float,
) -> ActionGuardDecision:
    """Validate every physical target and reject the whole chunk on any fault."""
    current = np.asarray(current_pose_xyz_quaternion_xyzw, dtype=np.float64)
    targets = np.asarray(decoded_targets, dtype=np.float64)
    if current.shape != (7,) or not np.isfinite(current).all():
        raise ValueError("current pose must be finite XYZ+XYZW")
    current[3:7] = normalize_quaternion_xyzw(current[3:7])
    if targets.ndim != 2 or targets.shape[1] != 8:
        raise ValueError("decoded targets must have shape (T,8)")
    if not np.isfinite(targets).all():
        raise ValueError("decoded targets contain NaN/Inf")
    if not 1 <= execution_horizon <= min(len(targets), limits.maximum_horizon):
        raise ValueError("execution horizon exceeds available or configured targets")
    age = float(command_age_s)
    if not math.isfinite(age) or age < 0.0:
        raise ValueError("command age must be finite and non-negative")
    current_gripper = float(current_gripper_training)
    if not math.isfinite(current_gripper):
        raise ValueError("current gripper is non-finite")

    guarded = targets[:execution_horizon].copy()
    violations: list[dict[str, Any]] = []
    translation_steps: list[float] = []
    rotation_steps: list[float] = []
    translation_accelerations: list[float] = []
    rotation_accelerations: list[float] = []
    previous_pose = current
    previous_gripper = current_gripper
    previous_translation_velocity = 0.0
    previous_rotation_velocity = 0.0

    if age > limits.maximum_command_age_s:
        violations.append(
            {
                "code": "stale_command",
                "value_s": age,
                "limit_s": limits.maximum_command_age_s,
            }
        )

    for index, target in enumerate(guarded):
        target[3:7] = normalize_quaternion_xyzw(target[3:7])
        xyz = target[:3]
        outside = np.flatnonzero(
            (xyz < limits.workspace_min_xyz_m)
            | (xyz > limits.workspace_max_xyz_m)
        )
        if len(outside):
            violations.append(
                {"step": index, "code": "workspace", "xyz_m": xyz.tolist()}
            )

        translation = float(np.linalg.norm(xyz - previous_pose[:3]))
        rotation = _rotation_step_deg(previous_pose[3:7], target[3:7])
        translation_steps.append(translation)
        rotation_steps.append(rotation)
        if translation > limits.maximum_translation_step_m:
            violations.append(
                {
                    "step": index,
                    "code": "translation_step",
                    "value_m": translation,
                    "limit_m": limits.maximum_translation_step_m,
                }
            )
        if rotation > limits.maximum_rotation_step_deg:
            violations.append(
                {
                    "step": index,
                    "code": "rotation_step",
                    "value_deg": rotation,
                    "limit_deg": limits.maximum_rotation_step_deg,
                }
            )

        translation_velocity = translation / limits.sample_period_s
        rotation_velocity = rotation / limits.sample_period_s
        translation_acceleration = (
            abs(translation_velocity - previous_translation_velocity)
            / limits.sample_period_s
        )
        rotation_acceleration = (
            abs(rotation_velocity - previous_rotation_velocity)
            / limits.sample_period_s
        )
        translation_accelerations.append(translation_acceleration)
        rotation_accelerations.append(rotation_acceleration)
        if translation_acceleration > limits.maximum_translation_acceleration_m_s2:
            violations.append(
                {
                    "step": index,
                    "code": "translation_acceleration",
                    "value_m_s2": translation_acceleration,
                    "limit_m_s2": limits.maximum_translation_acceleration_m_s2,
                }
            )
        if rotation_acceleration > limits.maximum_rotation_acceleration_deg_s2:
            violations.append(
                {
                    "step": index,
                    "code": "rotation_acceleration",
                    "value_deg_s2": rotation_acceleration,
                    "limit_deg_s2": limits.maximum_rotation_acceleration_deg_s2,
                }
            )

        gripper = float(target[7])
        if not (
            limits.gripper_minimum - 1e-6
            <= gripper
            <= limits.gripper_maximum + 1e-6
        ):
            violations.append(
                {"step": index, "code": "gripper_range", "value": gripper}
            )
        target[7] = np.clip(
            gripper, limits.gripper_minimum, limits.gripper_maximum
        )
        gripper = float(target[7])
        if abs(gripper - previous_gripper) > limits.gripper_maximum_step:
            violations.append(
                {
                    "step": index,
                    "code": "gripper_step",
                    "value": abs(gripper - previous_gripper),
                    "limit": limits.gripper_maximum_step,
                }
            )
        requests_closure = (
            previous_gripper < limits.gripper_closure_threshold
            and gripper >= limits.gripper_closure_threshold
        )
        if not limits.gripper_closure_enabled and requests_closure:
            violations.append(
                {"step": index, "code": "gripper_closure_not_approved"}
            )

        previous_pose = target[:7].copy()
        previous_gripper = gripper
        previous_translation_velocity = translation_velocity
        previous_rotation_velocity = rotation_velocity

    metrics = {
        "execution_horizon": execution_horizon,
        "command_age_s": age,
        "maximum_translation_step_m": max(translation_steps),
        "maximum_rotation_step_deg": max(rotation_steps),
        "maximum_translation_acceleration_m_s2": max(translation_accelerations),
        "maximum_rotation_acceleration_deg_s2": max(rotation_accelerations),
    }
    if violations:
        guarded[:, :7] = np.tile(current, (execution_horizon, 1))
        guarded[:, 7] = current_gripper
    return ActionGuardDecision(
        validation_pass=not violations,
        guarded_targets=guarded.astype(np.float32),
        violations=tuple(violations),
        metrics=metrics,
    )


def build_gdk_a2d_actions(decision: ActionGuardDecision) -> list[dict[str, Any]]:
    """Build right-only GDK payloads after shadow validation.

    The result contains no left arm, head, waist, or chassis fields.  This
    function is serialization only and does not authorize controller use.
    """
    if not decision.validation_pass:
        raise ValueError("refusing to build controller payload for rejected chunk")
    actions = []
    for target in decision.guarded_targets:
        actions.append(
            {
                "right_arm": {
                    "control_type": "ABS_POSE",
                    "action_data": [float(value) for value in target[:7]],
                },
                "right_effector": {
                    "control_type": "ABS_JOINT",
                    "action_data": [float(target[7])],
                },
            }
        )
    return actions
