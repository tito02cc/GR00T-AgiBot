#!/usr/bin/env python3
"""Validate the right-only G2 action gate using live-shadow model outputs."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agibot.tools.g2_groot_action_guard import (  # noqa: E402
    build_gdk_a2d_actions,
    evaluate_action_chunk,
    load_shadow_action_guard,
)


CONFIG = REPO_ROOT / "agibot/configs/g2_groot_right_action_guard_shadow.json"
INPUT = REPO_ROOT / "agibot/reports/g2_groot_live_shadow_inference_h8.json"
OUTPUT = REPO_ROOT / "agibot/reports/g2_groot_action_guard_shadow_validation.json"


def violation_codes(decision: Any) -> list[str]:
    return sorted({str(item["code"]) for item in decision.violations})


def main() -> None:
    limits = load_shadow_action_guard(CONFIG)
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = []
    for call in source["calls"]:
        current = call["current_pose_xyz_quaternion_xyzw"]
        gripper = call["current_gripper_training"]
        targets = call["decoded_first_h8_targets"]
        h1 = evaluate_action_chunk(
            current, gripper, targets, limits, execution_horizon=1, command_age_s=0.0
        )
        h8 = evaluate_action_chunk(
            current, gripper, targets, limits, execution_horizon=8, command_age_s=0.0
        )
        payload = build_gdk_a2d_actions(h1) if h1.validation_pass else []
        prohibited = {"left_arm", "left_effector", "head", "waist", "chassis"}
        payload_right_only = all(not prohibited.intersection(item) for item in payload)
        rows.append(
            {
                "call": call["call"],
                "h1_pass": h1.validation_pass,
                "h1_metrics": h1.metrics,
                "h1_violation_codes": violation_codes(h1),
                "h8_pass": h8.validation_pass,
                "h8_metrics": h8.metrics,
                "h8_violation_codes": violation_codes(h8),
                "h1_payload_right_only": payload_right_only,
                "h1_payload": payload,
            }
        )

    base = source["calls"][0]
    current = np.asarray(base["current_pose_xyz_quaternion_xyzw"], dtype=np.float64)
    gripper = float(base["current_gripper_training"])
    good = np.asarray(base["decoded_first_h8_targets"], dtype=np.float64)
    faults = {}

    cases = {
        "stale": (good, 2.0),
        "workspace": (good.copy(), 0.0),
        "translation_spike": (good.copy(), 0.0),
        "rotation_spike": (good.copy(), 0.0),
        "gripper_close": (good.copy(), 0.0),
        "gripper_range": (good.copy(), 0.0),
    }
    cases["workspace"][0][0, 0] = 10.0
    cases["translation_spike"][0][0, 0] = current[0] + 0.02
    cases["rotation_spike"][0][0, 3:7] = [0.0, 0.0, 0.0, 1.0]
    cases["gripper_close"][0][0, 7] = -0.4
    cases["gripper_range"][0][0, 7] = 1.0
    for name, (targets, age) in cases.items():
        decision = evaluate_action_chunk(
            current, gripper, targets, limits, execution_horizon=1, command_age_s=age
        )
        faults[name] = {
            "rejected": not decision.validation_pass,
            "violation_codes": violation_codes(decision),
            "fallback_holds_current_pose": bool(
                np.allclose(decision.guarded_targets[0, :7], current, atol=1e-6)
            ),
            "fallback_holds_current_gripper": bool(
                abs(float(decision.guarded_targets[0, 7]) - gripper) <= 1e-6
            ),
        }

    h1_passes = sum(row["h1_pass"] for row in rows)
    all_faults_rejected = all(item["rejected"] for item in faults.values())
    status = (
        "PASS_NON_ACTUATING_ACTION_GUARD"
        if h1_passes == len(rows)
        and all(row["h1_payload_right_only"] for row in rows)
        and all_faults_rejected
        else "FAIL"
    )
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "scope": "right-only G2 action serialization and provisional shadow gates",
        "config": str(CONFIG.relative_to(REPO_ROOT)),
        "approval_status": limits.approval_status,
        "live_execution_enabled": False,
        "source_report": str(INPUT.relative_to(REPO_ROOT)),
        "live_shadow_calls": len(rows),
        "h1_passes": h1_passes,
        "h8_passes": sum(row["h8_pass"] for row in rows),
        "rows": rows,
        "fault_injection": faults,
        "all_faults_rejected": all_faults_rejected,
        "controller_api_called": False,
        "motor_commands_sent": 0,
        "remaining_before_live": [
            "cell-owner approval of workspace, velocity, acceleration, and gripper limits",
            "object-relative gripper closure gate",
            "controller watchdog and physical E-stop witnessed test",
            "guarded model-independent sub-millimetre round-trip",
            "operator approval immediately before first one-step model command"
        ]
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
