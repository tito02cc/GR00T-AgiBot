#!/usr/bin/env python3
"""Guarded sub-millimetre G2 right-arm trajectory round-trip probe.

Default operation is read-only.  Physical execution requires an exact token.
Only the right arm is present in the GDK payload; no gripper, left-arm, head,
waist, or chassis command is constructed.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import agibot_gdk
import numpy as np

from g2_groot_trajectory_tracking_probe import (
    LEFT_FRAME,
    RIGHT_FRAME,
    competing_controllers,
    displacement_metrics,
    emit,
    pose_values,
    quaternion_angle,
    require_safe_state,
    snapshot,
    translation_distance,
)


CONFIRMATION = "TEST_G2_GROOT_TRAJECTORY_0P5MM_ROUNDTRIP"
WORKSPACE_MIN = np.asarray([0.448, -0.333, 0.987], dtype=np.float64)
WORKSPACE_MAX = np.asarray([0.754, -0.197, 1.217], dtype=np.float64)
STATIONARY_DELAY_S = 0.30
STATIONARY_MAX_TRANSLATION_M = 0.00020
STATIONARY_MAX_ROTATION_RAD = 0.002
LEFT_MAX_TRANSLATION_M = 0.00050
LEFT_MAX_ROTATION_RAD = 0.005
RIGHT_ENVELOPE_MARGIN_M = 0.0010
RIGHT_MAX_ROTATION_RAD = 0.010
MAX_UNCOMMANDED_DOWNWARD_M = 0.00080
MAX_CROSS_AXIS_ERROR_M = 0.00060
MONITOR_PERIOD_S = 0.01


def segment_metrics(
    left_start: np.ndarray,
    right_start: np.ndarray,
    target: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    axis_index: int,
) -> dict[str, float]:
    cross = [index for index in range(3) if index != axis_index]
    return {
        **displacement_metrics(left_start, right_start, left, right),
        "target_position_error_m": translation_distance(target, right),
        "target_rotation_error_rad": quaternion_angle(target, right),
        "cross_axis_error_m": float(np.linalg.norm(right[cross] - target[cross])),
    }


def segment_violations(
    values: dict[str, float], planned_distance_m: float
) -> list[str]:
    found = []
    if values["left_translation_m"] > LEFT_MAX_TRANSLATION_M:
        found.append("left arm translated")
    if values["left_rotation_rad"] > LEFT_MAX_ROTATION_RAD:
        found.append("left arm rotated")
    if values["right_translation_m"] > planned_distance_m + RIGHT_ENVELOPE_MARGIN_M:
        found.append("right arm exceeded translation envelope")
    if values["right_rotation_rad"] > RIGHT_MAX_ROTATION_RAD:
        found.append("right arm exceeded rotation envelope")
    if values["right_downward_m"] > MAX_UNCOMMANDED_DOWNWARD_M:
        found.append("right arm moved downward outside target segment")
    if values["cross_axis_error_m"] > MAX_CROSS_AXIS_ERROR_M:
        found.append("right arm cross-axis error")
    return found


def run_segment(
    robot: Any,
    tf: Any,
    name: str,
    left_start: np.ndarray,
    right_start: np.ndarray,
    segment_start: np.ndarray,
    target: np.ndarray,
    axis_index: int,
    points: int,
    trajectory_time_s: float,
    post_monitor_s: float,
) -> dict[str, Any]:
    actions = []
    for index in range(points):
        alpha = float(index + 1) / float(points)
        pose = segment_start.copy()
        pose[:3] = segment_start[:3] * (1.0 - alpha) + target[:3] * alpha
        # This diagnostic does not command an orientation change.
        pose[3:7] = right_start[3:7]
        actions.append(
            {
                "right_arm": {
                    "control_type": "ABS_POSE",
                    "action_data": pose.tolist(),
                }
            }
        )

    emit(
        "segment_start",
        segment=name,
        target_pose=target.round(9).tolist(),
        points=points,
        trajectory_time_s=trajectory_time_s,
    )
    result = robot.trajectory_tracking_control(
        time.time_ns(),
        {},
        actions,
        robot_link="base_link",
        trajectory_reference_time=trajectory_time_s,
    )
    emit("trajectory_published", segment=name, gdk_result=int(result))
    if int(result) != 0:
        raise RuntimeError(f"{name}: GDK result={result}")

    planned_distance_m = translation_distance(right_start, target)
    deadline = time.monotonic() + trajectory_time_s + post_monitor_s
    maxima: dict[str, float] = {}
    samples = 0
    first_violation = None
    while time.monotonic() < deadline:
        left = pose_values(tf, LEFT_FRAME)
        right = pose_values(tf, RIGHT_FRAME)
        current = segment_metrics(
            left_start, right_start, target, left, right, axis_index
        )
        for key, value in current.items():
            maxima[key] = max(maxima.get(key, 0.0), value)
        current_violations = segment_violations(current, planned_distance_m)
        if current_violations and first_violation is None:
            first_violation = {
                "violations": current_violations,
                "metrics": current,
                "instruction": "use physical emergency stop if motion continues",
            }
            emit("guard_violation", segment=name, **first_violation)
        samples += 1
        time.sleep(MONITOR_PERIOD_S)

    left_final = pose_values(tf, LEFT_FRAME)
    right_final = pose_values(tf, RIGHT_FRAME)
    final = segment_metrics(
        left_start, right_start, target, left_final, right_final, axis_index
    )
    final_violations = segment_violations(final, planned_distance_m)
    outcome = {
        "segment": name,
        "passed": first_violation is None and not final_violations,
        "samples": samples,
        "max": maxima,
        "final": final,
        "final_pose": right_final.tolist(),
        "first_violation": first_violation,
        "final_violations": final_violations,
    }
    emit("segment_complete", **outcome)
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--distance-mm", type=float, default=0.5)
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--trajectory-time-s", type=float, default=0.8)
    parser.add_argument("--post-monitor-s", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.1 <= args.distance_mm <= 0.5:
        parser.error("--distance-mm must be in [0.1, 0.5]")
    if not 2 <= args.points <= 16:
        parser.error("--points must be in [2, 16]")
    if not 0.4 <= args.trajectory_time_s <= 2.0:
        parser.error("--trajectory-time-s must be in [0.4, 2.0]")
    if not 0.3 <= args.post_monitor_s <= 2.0:
        parser.error("--post-monitor-s must be in [0.3, 2.0]")
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
        axis_index = {"x": 0, "y": 1}[args.axis]
        target = stable["right"].copy()
        target[axis_index] += args.distance_mm / 1000.0
        if np.any(target[:3] < WORKSPACE_MIN) or np.any(target[:3] > WORKSPACE_MAX):
            raise RuntimeError("target outside provisional workspace")
        emit(
            "preflight",
            execute=args.execute,
            request_owner="agibot_gdk.Robot.trajectory_tracking_control",
            mode=stable["mode"],
            whole=stable["whole"],
            start_pose=stable["right"].round(9).tolist(),
            target_pose=target.round(9).tolist(),
            distance_mm=args.distance_mm,
            axis=args.axis,
            points=args.points,
            trajectory_time_s=args.trajectory_time_s,
            stationary={key: round(value, 9) for key, value in stationary.items()},
            competing_controllers=competitors,
            payload_channels=["right_arm"],
            payload_control_type="ABS_POSE",
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
        outbound_target = right_start.copy()
        outbound_target[axis_index] += args.distance_mm / 1000.0
        outbound = run_segment(
            robot,
            tf,
            "outbound",
            left_start,
            right_start,
            right_start,
            outbound_target,
            axis_index,
            args.points,
            args.trajectory_time_s,
            args.post_monitor_s,
        )
        if not outbound["passed"]:
            raise RuntimeError("outbound guard failed; return segment withheld")
        return_segment = run_segment(
            robot,
            tf,
            "return",
            left_start,
            right_start,
            np.asarray(outbound["final_pose"], dtype=np.float64),
            right_start,
            axis_index,
            args.points,
            args.trajectory_time_s,
            args.post_monitor_s,
        )
        after = snapshot(robot, tf)
        require_safe_state(after)
        final_pose = after["right"]
        passed = outbound["passed"] and return_segment["passed"]
        emit(
            "result",
            passed=passed,
            outbound=outbound,
            return_segment=return_segment,
            final_pose=final_pose.round(9).tolist(),
            residual_from_start_translation_m=translation_distance(
                right_start, final_pose
            ),
            residual_from_start_rotation_rad=quaternion_angle(
                right_start, final_pose
            ),
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
