#!/usr/bin/env python3
"""Fault-injection tests for the non-actuating Xichong shadow safety layer."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agibot.tools.g2_shadow_safety import (  # noqa: E402
    evaluate_shadow_chunk,
    load_shadow_safety_limits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "agibot/configs/xichong_shadow_safety.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "agibot/reports/xichong_shadow_safety_fault_injection.json",
    )
    return parser.parse_args()


def target(xyz: np.ndarray, quat: np.ndarray, gripper: float, count: int = 8) -> np.ndarray:
    row = np.concatenate((xyz, quat, [gripper]))
    return np.tile(row, (count, 1)).astype(np.float32)


def main() -> None:
    args = parse_args()
    limits = load_shadow_safety_limits(args.config)
    workspace_center = (limits.workspace_min_xyz_m + limits.workspace_max_xyz_m) / 2.0
    closure_center = (limits.closure_min_xyz_m + limits.closure_max_xyz_m) / 2.0
    identity = np.asarray([0.0, 0.0, 0.0, 1.0])
    current = np.concatenate((workspace_center, identity))
    cases: dict[str, bool] = {}

    allowed = evaluate_shadow_chunk(
        current, -0.785, target(workspace_center, identity, -0.785), limits, execution_horizon=8
    )
    cases["nominal_allowed"] = allowed.chunk_allowed and not allowed.violations

    outside_xyz = workspace_center.copy()
    outside_xyz[0] = limits.workspace_max_xyz_m[0] + 0.05
    outside = evaluate_shadow_chunk(
        current, -0.785, target(outside_xyz, identity, -0.785), limits, execution_horizon=8
    )
    cases["workspace_rejected"] = (
        not outside.chunk_allowed
        and any(row["code"] == "workspace" for row in outside.violations)
    )

    jump_xyz = workspace_center.copy()
    jump_xyz[0] += limits.max_translation_step_m * 1.1
    jump = evaluate_shadow_chunk(
        current, -0.785, target(jump_xyz, identity, -0.785), limits, execution_horizon=8
    )
    cases["translation_rejected"] = (
        not jump.chunk_allowed
        and any(row["code"] == "translation_step" for row in jump.violations)
    )

    angle = np.deg2rad(limits.max_rotation_step_deg * 1.2)
    rotated_quat = Rotation.from_rotvec([0.0, 0.0, angle]).as_quat()
    rotation = evaluate_shadow_chunk(
        current,
        -0.785,
        target(workspace_center, rotated_quat, -0.785),
        limits,
        execution_horizon=8,
    )
    cases["rotation_rejected"] = (
        not rotation.chunk_allowed
        and any(row["code"] == "rotation_step" for row in rotation.violations)
    )

    premature = evaluate_shadow_chunk(
        current,
        -0.785,
        target(workspace_center, identity, -0.1),
        limits,
        execution_horizon=8,
    )
    cases["premature_close_gated"] = (
        premature.chunk_allowed
        and len(premature.gripper_gate_events) == 8
        and np.allclose(premature.safe_targets[:, 7], -0.785)
    )

    closure_current = np.concatenate((closure_center, identity))
    valid_close = evaluate_shadow_chunk(
        closure_current,
        -0.785,
        target(closure_center, identity, -0.1),
        limits,
        execution_horizon=8,
    )
    cases["close_inside_envelope_allowed"] = (
        valid_close.chunk_allowed and not valid_close.gripper_gate_events
    )

    already_closed = evaluate_shadow_chunk(
        current,
        -0.1,
        target(workspace_center, identity, -0.1),
        limits,
        execution_horizon=8,
    )
    cases["closed_lift_not_regated"] = (
        already_closed.chunk_allowed and not already_closed.gripper_gate_events
    )

    status = "PASS" if all(cases.values()) else "FAIL"
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "scope": "non-actuating shadow safety fault injection",
        "config": str(args.config.resolve()),
        "approval_status": limits.approval_status,
        "live_execution_enabled": limits.live_execution_enabled,
        "cases": cases,
        "motor_commands_sent": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
