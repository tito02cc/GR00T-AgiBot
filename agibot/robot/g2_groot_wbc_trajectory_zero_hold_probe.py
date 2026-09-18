#!/usr/bin/env python3
"""Guarded zero-hold probe using the vendor /wbc/model_predict path.

This follows the G2 firmware's bundled ``model_predict_demo.py`` message path.
Default operation is read-only.  Physical execution publishes only right-arm
group 3 ABS_POSE points at the current EEF pose.
"""

from __future__ import annotations

import argparse
import time

import agibot_gdk

from genie_msgs_pb.msg.RetargetInfo_pb2 import RetargetInfo
from genie_msgs_pb.msg.RetargetInfoArray_pb2 import RetargetInfoArray
from genie_msgs_pb.msg.TrajectoryTrackingControl_pb2 import (
    TrajectoryTrackingControl,
)

from g2_groot_trajectory_tracking_probe import (
    LEFT_FRAME,
    RIGHT_FRAME,
    competing_controllers,
    displacement_metrics,
    emit,
    pose_values,
    require_safe_state,
    snapshot,
    violations,
)


CONFIRMATION = "TEST_G2_GROOT_WBC_TRAJECTORY_ZERO_HOLD"
TOPIC = "/wbc/model_predict"
RIGHT_ARM_GROUP_ID = 3
ABS_POSE_CONTROL_TYPE = 0
INPUT_TYPE_GDK = 54


def add_pose(info: RetargetInfo, pose) -> None:
    target = info.target_frame_poses.add()
    target.position.x = float(pose[0])
    target.position.y = float(pose[1])
    target.position.z = float(pose[2])
    target.orientation.x = float(pose[3])
    target.orientation.y = float(pose[4])
    target.orientation.z = float(pose[5])
    target.orientation.w = float(pose[6])


def build_message(pose, points: int, trajectory_time_s: float):
    message = TrajectoryTrackingControl()
    message.trajectory_reference_time = float(trajectory_time_s)
    for _ in range(points):
        group_array = RetargetInfoArray()
        info = group_array.retarget_infos.add()
        info.group_id = RIGHT_ARM_GROUP_ID
        info.input_type = INPUT_TYPE_GDK
        info.control_type = ABS_POSE_CONTROL_TYPE
        info.frame_id = "base_link"
        info.target_frame_names.append(RIGHT_FRAME)
        add_pose(info, pose)
        message.retarget_groups_trajectory.append(group_array)
    return message


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
        dds = agibot_gdk.dds
        node = dds.Node("g2_groot_wbc_trajectory_zero_hold")
        publisher = node.create_publisher(
            TOPIC, TrajectoryTrackingControl, dds.GDKQoS()
        )
        time.sleep(2.0)

        before = snapshot(robot, tf)
        require_safe_state(before)
        time.sleep(0.30)
        stable = snapshot(robot, tf)
        require_safe_state(stable)
        stationary = displacement_metrics(
            before["left"], before["right"], stable["left"], stable["right"]
        )
        if max(
            stationary["left_translation_m"], stationary["right_translation_m"]
        ) > 0.00020:
            raise RuntimeError(f"robot is not translationally stationary: {stationary}")
        if max(
            stationary["left_rotation_rad"], stationary["right_rotation_rad"]
        ) > 0.002:
            raise RuntimeError(f"robot is not rotationally stationary: {stationary}")
        competitors = competing_controllers()
        message = build_message(
            stable["right"], args.points, args.trajectory_time_s
        )
        emit(
            "preflight",
            execute=args.execute_zero_hold,
            transport="vendor_direct_wbc_model_predict",
            topic=TOPIC,
            mode=stable["mode"],
            whole=stable["whole"],
            start_pose=stable["right"].round(9).tolist(),
            points=args.points,
            trajectory_time_s=args.trajectory_time_s,
            stationary={key: round(value, 9) for key, value in stationary.items()},
            competing_controllers=competitors,
            group_id=RIGHT_ARM_GROUP_ID,
            input_type=INPUT_TYPE_GDK,
            control_type=ABS_POSE_CONTROL_TYPE,
            frame_id="base_link",
            target_frame_name=RIGHT_FRAME,
            protobuf_bytes=len(message.SerializeToString()),
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
        message = build_message(right_start, args.points, args.trajectory_time_s)
        publisher.publish(message)
        emit("trajectory_published", topic=TOPIC)

        deadline = time.monotonic() + args.trajectory_time_s + args.post_monitor_s
        maxima = {}
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
            time.sleep(0.01)

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
