#!/usr/bin/env python3
"""Guarded G2 right-arm probe using only GDK's MotionPlan request owner.

Default operation is read-only.  Execution requires an exact confirmation
token.  This probe never commands the left arm, either gripper, or the chassis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from typing import Any

import agibot_gdk
import numpy as np


CONFIRMATION = "TEST_G2_GROOT_SINGLE_OWNER_MOTIONPLAN"
LEFT_FRAME = "arm_l_end_link"
RIGHT_FRAME = "arm_r_end_link"
REQUIRED_MODE = 5
WORKSPACE_MIN = np.asarray([0.448, -0.333, 0.987], dtype=np.float64)
WORKSPACE_MAX = np.asarray([0.754, -0.197, 1.217], dtype=np.float64)
STATIONARY_DELAY_S = 0.30
STATIONARY_MAX_TRANSLATION_M = 0.00020
STATIONARY_MAX_ROTATION_RAD = 0.002
LEFT_MAX_TRANSLATION_M = 0.00050
LEFT_MAX_ROTATION_RAD = 0.005
RIGHT_ENVELOPE_MARGIN_M = 0.0010
RIGHT_MAX_ROTATION_FROM_START_RAD = 0.010
MAX_UNCOMMANDED_DOWNWARD_M = 0.00080
MAX_CROSS_AXIS_ERROR_M = 0.00060
DEFAULT_RESPONSE_TIMEOUT_S = 12.0
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
        "whole": robot.get_whole_body_status(),
        "left": pose_values(tf, LEFT_FRAME),
        "right": pose_values(tf, RIGHT_FRAME),
    }


def require_safe_state(state: dict[str, Any]) -> None:
    if state["mode"] != REQUIRED_MODE:
        raise RuntimeError(f"motion mode must be {REQUIRED_MODE}; got {state['mode']}")
    if state["motion_error"] != 0:
        raise RuntimeError(f"motion control error={state['motion_error']}")
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
        "g2_krightarm_motionplan_roundtrip_probe.py",
    )
    own_pid = os.getpid()
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == own_pid:
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


def metrics(
    left_start: np.ndarray,
    right_start: np.ndarray,
    target: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    planned_distance_m: float,
) -> dict[str, float]:
    delta = target[:3] - right_start[:3]
    axis = int(np.argmax(np.abs(delta))) if planned_distance_m > 0.0 else 0
    cross = [index for index in range(3) if index != axis]
    return {
        "left_translation_m": translation_distance(left_start, left),
        "left_rotation_rad": quaternion_angle(left_start, left),
        "right_translation_from_start_m": translation_distance(right_start, right),
        "right_rotation_from_start_rad": quaternion_angle(right_start, right),
        "target_position_error_m": translation_distance(target, right),
        "target_rotation_error_rad": quaternion_angle(target, right),
        "uncommanded_downward_m": max(
            0.0, min(float(right_start[2]), float(target[2])) - float(right[2])
        ),
        "cross_axis_error_m": float(np.linalg.norm(right[cross] - target[cross])),
    }


def enforce_guards(values: dict[str, float], planned_distance_m: float) -> None:
    violations = []
    if values["left_translation_m"] > LEFT_MAX_TRANSLATION_M:
        violations.append("left arm translated")
    if values["left_rotation_rad"] > LEFT_MAX_ROTATION_RAD:
        violations.append("left arm rotated")
    if values["right_translation_from_start_m"] > (
        planned_distance_m + RIGHT_ENVELOPE_MARGIN_M
    ):
        violations.append("right arm exceeded translation envelope")
    if values["right_rotation_from_start_rad"] > RIGHT_MAX_ROTATION_FROM_START_RAD:
        violations.append("right arm exceeded rotation envelope")
    if values["uncommanded_downward_m"] > MAX_UNCOMMANDED_DOWNWARD_M:
        violations.append("right arm moved downward outside target segment")
    if values["cross_axis_error_m"] > MAX_CROSS_AXIS_ERROR_M:
        violations.append("right arm cross-axis error")
    if violations:
        raise RuntimeError("; ".join(violations) + ": " + json.dumps(values))


def call_gdk_motion_plan(
    robot: Any,
    tf: Any,
    name: str,
    left_start: np.ndarray,
    right_start: np.ndarray,
    target: np.ndarray,
    speed_scale: float,
    response_timeout_s: float,
) -> dict[str, Any]:
    """Call GDK's synchronous request while the main thread monitors TF."""
    planned_distance_m = translation_distance(right_start, target)
    output: dict[str, Any] = {}
    completed = threading.Event()

    def worker() -> None:
        try:
            actions = [
                {
                    "right_arm": {
                        "control_type": "ABS_POSE",
                        "action_data": target.tolist(),
                    }
                }
            ]
            output["result"] = int(
                robot.motion_plan_request(
                    actions,
                    robot_link="base_link",
                    move_type="INTERP_CARTESIAN_SPACE",
                    trajectory_reference_time=speed_scale,
                    timeout=response_timeout_s,
                    enable_env_collision=True,
                    enable_self_collision=True,
                    enable_com_check=True,
                    log_level=2,
                )
            )
        except Exception as error:
            output["error"] = f"{type(error).__name__}: {error}"
        finally:
            completed.set()

    emit(
        "segment_start",
        segment=name,
        owner="agibot_gdk.Robot.motion_plan_request",
        target_pose=target.round(9).tolist(),
        planned_distance_m=planned_distance_m,
        speed_scale=speed_scale,
    )
    thread = threading.Thread(target=worker, name=f"motionplan-{name}", daemon=True)
    thread.start()
    deadline = time.monotonic() + response_timeout_s + 2.0
    maxima: dict[str, float] = {}
    samples = 0
    guard_error = None
    while not completed.wait(MONITOR_PERIOD_S):
        left = pose_values(tf, LEFT_FRAME)
        right = pose_values(tf, RIGHT_FRAME)
        values = metrics(
            left_start, right_start, target, left, right, planned_distance_m
        )
        for key, value in values.items():
            maxima[key] = max(maxima.get(key, 0.0), value)
        samples += 1
        try:
            enforce_guards(values, planned_distance_m)
        except Exception as error:
            guard_error = f"{type(error).__name__}: {error}"
            emit(
                "guard_violation",
                segment=name,
                message=guard_error,
                instruction="use physical emergency stop if motion continues",
            )
            break
        if time.monotonic() >= deadline:
            guard_error = "monitor deadline exceeded"
            break
    thread.join(timeout=1.0)
    if thread.is_alive():
        raise TimeoutError(f"{name}: GDK MotionPlan call did not return")
    if guard_error is not None:
        raise RuntimeError(f"{name}: {guard_error}")
    if "error" in output:
        raise RuntimeError(f"{name}: {output['error']}")
    if output.get("result") != 0:
        raise RuntimeError(f"{name}: GDK result={output.get('result')}")

    time.sleep(0.20)
    left = pose_values(tf, LEFT_FRAME)
    right = pose_values(tf, RIGHT_FRAME)
    final = metrics(left_start, right_start, target, left, right, planned_distance_m)
    enforce_guards(final, planned_distance_m)
    result = {
        "gdk_result": output["result"],
        "samples": samples,
        "max": maxima,
        "final": final,
        "final_pose": right.tolist(),
    }
    emit("segment_complete", segment=name, **result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--distance-mm", type=float, default=0.0)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--speed-scale", type=float, default=0.2)
    parser.add_argument("--response-timeout-s", type=float, default=12.0)
    args = parser.parse_args()
    if not 0.0 <= args.distance_mm <= 1.0:
        parser.error("--distance-mm must be in [0, 1.0]")
    if not 0.05 <= args.speed_scale <= 0.3:
        parser.error("--speed-scale must be in [0.05, 0.3]")
    if not 5.0 <= args.response_timeout_s <= 20.0:
        parser.error("--response-timeout-s must be in [5, 20]")
    if args.execute and args.confirm != CONFIRMATION:
        parser.error(f"--execute requires --confirm {CONFIRMATION}")
    if not args.execute and args.confirm:
        parser.error("--confirm is only valid with --execute")

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
        stationary = {
            "left_translation_m": translation_distance(before["left"], stable["left"]),
            "left_rotation_rad": quaternion_angle(before["left"], stable["left"]),
            "right_translation_m": translation_distance(before["right"], stable["right"]),
            "right_rotation_rad": quaternion_angle(before["right"], stable["right"]),
        }
        if max(
            stationary["left_translation_m"], stationary["right_translation_m"]
        ) > STATIONARY_MAX_TRANSLATION_M:
            raise RuntimeError(f"robot is not translationally stationary: {stationary}")
        if max(
            stationary["left_rotation_rad"], stationary["right_rotation_rad"]
        ) > STATIONARY_MAX_ROTATION_RAD:
            raise RuntimeError(f"robot is not rotationally stationary: {stationary}")
        competitors = competing_controllers()
        axis_index = {"x": 0, "y": 1, "z": 2}[args.axis]
        target = stable["right"].copy()
        target[axis_index] += args.distance_mm / 1000.0
        if np.any(target[:3] < WORKSPACE_MIN) or np.any(target[:3] > WORKSPACE_MAX):
            raise RuntimeError("target outside provisional workspace")
        emit(
            "preflight",
            execute=args.execute,
            request_owner="agibot_gdk.Robot.motion_plan_request",
            mode=stable["mode"],
            whole=stable["whole"],
            start_pose=stable["right"].round(9).tolist(),
            target_pose=target.round(9).tolist(),
            distance_mm=args.distance_mm,
            axis=args.axis,
            stationary={key: round(value, 9) for key, value in stationary.items()},
            competing_controllers=competitors,
        )
        if not args.execute:
            emit("shutdown", reason="read_only")
            return 0
        if competitors:
            raise RuntimeError(f"competing controllers found: {competitors}")
        for remaining in range(5, 0, -1):
            emit("physical_countdown", remaining_s=remaining)
            time.sleep(1.0)

        left_start = pose_values(tf, LEFT_FRAME)
        right_start = pose_values(tf, RIGHT_FRAME)
        command_target = right_start.copy()
        command_target[axis_index] += args.distance_mm / 1000.0
        segments = [
            call_gdk_motion_plan(
                robot,
                tf,
                "zero_hold" if args.distance_mm == 0.0 else "outbound",
                left_start,
                right_start,
                command_target,
                args.speed_scale,
                args.response_timeout_s,
            )
        ]
        if args.distance_mm > 0.0:
            segments.append(
                call_gdk_motion_plan(
                    robot,
                    tf,
                    "return",
                    left_start,
                    right_start,
                    right_start,
                    args.speed_scale,
                    args.response_timeout_s,
                )
            )
        final_pose = pose_values(tf, RIGHT_FRAME)
        emit(
            "result",
            passed=True,
            final_pose=final_pose.round(9).tolist(),
            residual_from_start_translation_m=translation_distance(
                right_start, final_pose
            ),
            residual_from_start_rotation_rad=quaternion_angle(
                right_start, final_pose
            ),
            segments=segments,
        )
        return 0
    finally:
        agibot_gdk.gdk_release()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        emit("fatal", error_type=type(error).__name__, message=str(error))
        raise
