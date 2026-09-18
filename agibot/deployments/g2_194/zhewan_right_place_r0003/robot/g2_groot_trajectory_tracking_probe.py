#!/usr/bin/env python3
"""Guarded G2 right-arm probe for GDK trajectory tracking.

The default mode is strictly read-only.  The only physical operation exposed
by this diagnostic is a zero-displacement hold: the current right EEF pose is
repeated as an ABS_POSE trajectory.  It never commands the left arm, grippers,
waist, head, or chassis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any

import agibot_gdk
import numpy as np


CONFIRMATION = "TEST_G2_GROOT_TRAJECTORY_ZERO_HOLD"
LEFT_FRAME = "arm_l_end_link"
RIGHT_FRAME = "arm_r_end_link"
REQUIRED_MODE = 5
STATIONARY_DELAY_S = 0.30
STATIONARY_MAX_TRANSLATION_M = 0.00020
STATIONARY_MAX_ROTATION_RAD = 0.002
LEFT_MAX_TRANSLATION_M = 0.00050
LEFT_MAX_ROTATION_RAD = 0.005
RIGHT_MAX_TRANSLATION_M = 0.0050
RIGHT_MAX_ROTATION_RAD = 0.020
MAX_DOWNWARD_M = 0.0040
MONITOR_PERIOD_S = 0.01


def emit(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def pose_values(tf: Any, frame: str) -> np.ndarray:
    value = tf.get_tf_from_base_link(frame)
    pose = np.asarray(
        [
            value.translation.x,
            value.translation.y,
            value.translation.z,
            value.rotation.x,
            value.rotation.y,
            value.rotation.z,
            value.rotation.w,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(pose[3:7]))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("invalid EEF quaternion")
    pose[3:7] /= norm
    return pose


def translation_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first[:3] - second[:3]))


def quaternion_angle(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(first[3:7], second[3:7])))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def snapshot(robot: Any, tf: Any) -> dict[str, Any]:
    motion = robot.get_motion_control_status()
    return {
        "mode": int(getattr(motion, "mode", -1)),
        "motion_error": int(getattr(motion, "error_code", -1)),
        "motion_error_msg": str(getattr(motion, "error_msg", "")),
        "whole": robot.get_whole_body_status(),
        "left": pose_values(tf, LEFT_FRAME),
        "right": pose_values(tf, RIGHT_FRAME),
    }


def require_safe_state(
    state: dict[str, Any], required_modes: int | tuple[int, ...] = REQUIRED_MODE
) -> None:
    accepted = (required_modes,) if isinstance(required_modes, int) else required_modes
    if state["mode"] not in accepted:
        raise RuntimeError(
            f"motion mode must be one of {accepted}; got {state['mode']}"
        )
    if state["motion_error"] != 0:
        raise RuntimeError(
            f"motion control error={state['motion_error']}: "
            f"{state.get('motion_error_msg', '')}"
        )
    whole = state["whole"]
    for key in (
        "left_arm_error",
        "right_arm_error",
        "left_end_error",
        "right_end_error",
    ):
        if int(whole.get(key, -1)) != 0:
            raise RuntimeError(f"{key}={whole.get(key)}")
    for key in ("left_arm_estop", "right_arm_estop"):
        if bool(whole.get(key, False)):
            raise RuntimeError(f"{key}=true")
    for key in ("left_arm_control", "right_arm_control"):
        if bool(whole.get(key, False)):
            raise RuntimeError(f"{key}=true before probe")


def competing_controllers() -> list[dict[str, Any]]:
    needles = (
        "g2_v10.py",
        "g2_v11_motionplan.py",
        "g2_groot_right_action_bridge.py",
        "g2_groot_motionplan_single_owner_probe.py",
        "g2_krightarm_motionplan_roundtrip_probe.py",
        "g2_groot_trajectory_tracking_probe.py",
        "g2_groot_trajectory_roundtrip_probe.py",
        "g2_groot_wbc_trajectory_zero_hold_probe.py",
        "g2_groot_wbc_compensated_roundtrip_probe.py",
        "g2_groot_recover_pre_wbc_pose.py",
        "g2_groot_persistent_right_arm_controller.py",
        "g2_groot_persistent_h1_action_bridge.py",
    )
    excluded_pids = set()
    pid = os.getpid()
    while pid > 1 and pid not in excluded_pids:
        excluded_pids.add(pid)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as stream:
                pid = int(stream.read().rsplit(")", 1)[1].split()[1])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            break
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) in excluded_pids:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as stream:
                command = stream.read().replace(b"\0", b" ").decode(
                    "utf-8", errors="replace"
                )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(needle in command for needle in needles):
            found.append({"pid": int(entry), "command": command.strip()})
    return found


def displacement_metrics(
    left_start: np.ndarray, right_start: np.ndarray, left: np.ndarray, right: np.ndarray
) -> dict[str, float]:
    return {
        "left_translation_m": translation_distance(left_start, left),
        "left_rotation_rad": quaternion_angle(left_start, left),
        "right_translation_m": translation_distance(right_start, right),
        "right_rotation_rad": quaternion_angle(right_start, right),
        "right_downward_m": max(0.0, float(right_start[2] - right[2])),
    }


def violations(values: dict[str, float]) -> list[str]:
    found = []
    if values["left_translation_m"] > LEFT_MAX_TRANSLATION_M:
        found.append("left arm translated")
    if values["left_rotation_rad"] > LEFT_MAX_ROTATION_RAD:
        found.append("left arm rotated")
    if values["right_translation_m"] > RIGHT_MAX_TRANSLATION_M:
        found.append("right arm exceeded zero-hold envelope")
    if values["right_rotation_rad"] > RIGHT_MAX_ROTATION_RAD:
        found.append("right arm exceeded zero-hold rotation envelope")
    if values["right_downward_m"] > MAX_DOWNWARD_M:
        found.append("right arm moved downward beyond envelope")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-zero-hold", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--trajectory-time-s", type=float, default=0.8)
    parser.add_argument("--post-monitor-s", type=float, default=1.0)
    args = parser.parse_args()
    if not 2 <= args.points <= 16:
        parser.error("--points must be in [2, 16]")
    if not 0.2 <= args.trajectory_time_s <= 2.0:
        parser.error("--trajectory-time-s must be in [0.2, 2.0]")
    if not 0.5 <= args.post_monitor_s <= 3.0:
        parser.error("--post-monitor-s must be in [0.5, 3.0]")
    if args.execute_zero_hold and args.confirm != CONFIRMATION:
        parser.error(
            f"--execute-zero-hold requires --confirm {CONFIRMATION}"
        )
    if not args.execute_zero_hold and args.confirm:
        parser.error("--confirm is only valid with --execute-zero-hold")

    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise RuntimeError("gdk_init failed")
    try:
        robot = agibot_gdk.Robot()
        tf = agibot_gdk.TF()
        time.sleep(2.0)
        before = snapshot(robot, tf)
        require_safe_state(before)
        time.sleep(STATIONARY_DELAY_S)
        stable = snapshot(robot, tf)
        require_safe_state(stable)
        stationary = displacement_metrics(
            before["left"], before["right"], stable["left"], stable["right"]
        )
        if max(
            stationary["left_translation_m"], stationary["right_translation_m"]
        ) > STATIONARY_MAX_TRANSLATION_M:
            raise RuntimeError(f"robot is not translationally stationary: {stationary}")
        if max(
            stationary["left_rotation_rad"], stationary["right_rotation_rad"]
        ) > STATIONARY_MAX_ROTATION_RAD:
            raise RuntimeError(f"robot is not rotationally stationary: {stationary}")
        competitors = competing_controllers()
        action = {
            "right_arm": {
                "control_type": "ABS_POSE",
                "action_data": stable["right"].tolist(),
            }
        }
        emit(
            "preflight",
            execute=args.execute_zero_hold,
            request_owner="agibot_gdk.Robot.trajectory_tracking_control",
            mode=stable["mode"],
            whole=stable["whole"],
            start_pose=stable["right"].round(9).tolist(),
            points=args.points,
            trajectory_time_s=args.trajectory_time_s,
            stationary={key: round(value, 9) for key, value in stationary.items()},
            competing_controllers=competitors,
            payload_channels=["right_arm"],
            payload_control_type="ABS_POSE",
            omitted_channels=[
                "left_arm",
                "head",
                "waist",
                "left_effector",
                "right_effector",
                "chassis",
            ],
        )
        if not args.execute_zero_hold:
            emit("shutdown", reason="read_only")
            return 0
        if competitors:
            raise RuntimeError(f"competing controllers found: {competitors}")
        for remaining in range(5, 0, -1):
            emit("physical_countdown", remaining_s=remaining)
            time.sleep(1.0)

        left_start = pose_values(tf, LEFT_FRAME)
        right_start = pose_values(tf, RIGHT_FRAME)
        zero_action = {
            "right_arm": {
                "control_type": "ABS_POSE",
                "action_data": right_start.tolist(),
            }
        }
        result = robot.trajectory_tracking_control(
            time.time_ns(),
            {},
            [zero_action for _ in range(args.points)],
            robot_link="base_link",
            trajectory_reference_time=args.trajectory_time_s,
        )
        emit("trajectory_published", gdk_result=int(result))
        if int(result) != 0:
            raise RuntimeError(f"GDK result={result}")

        deadline = time.monotonic() + args.trajectory_time_s + args.post_monitor_s
        maxima: dict[str, float] = {}
        samples = 0
        first_violation = None
        while time.monotonic() < deadline:
            left = pose_values(tf, LEFT_FRAME)
            right = pose_values(tf, RIGHT_FRAME)
            current = displacement_metrics(left_start, right_start, left, right)
            for key, value in current.items():
                maxima[key] = max(maxima.get(key, 0.0), value)
            current_violations = violations(current)
            if current_violations and first_violation is None:
                first_violation = {
                    "violations": current_violations,
                    "metrics": current,
                    "instruction": "use physical emergency stop if motion continues",
                }
                emit("guard_violation", **first_violation)
            samples += 1
            time.sleep(MONITOR_PERIOD_S)

        after = snapshot(robot, tf)
        final = displacement_metrics(
            left_start, right_start, after["left"], after["right"]
        )
        require_safe_state(after)
        passed = first_violation is None and not violations(final)
        emit(
            "result",
            passed=passed,
            samples=samples,
            start_pose=right_start.round(9).tolist(),
            final_pose=after["right"].round(9).tolist(),
            max=maxima,
            final=final,
            first_violation=first_violation,
            mode=after["mode"],
            whole=after["whole"],
        )
        return 0 if passed else 2
    finally:
        agibot_gdk.gdk_release()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        emit("fatal", error_type=type(error).__name__, message=str(error))
        raise
