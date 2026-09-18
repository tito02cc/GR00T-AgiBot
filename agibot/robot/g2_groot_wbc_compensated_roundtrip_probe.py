#!/usr/bin/env python3
"""Validate compensated /wbc right-arm zero hold and 0.5 mm round trip."""

import argparse
import time

import agibot_gdk
import numpy as np
from genie_msgs_pb.msg.RetargetInfoArray_pb2 import RetargetInfoArray
from genie_msgs_pb.msg.TrajectoryTrackingControl_pb2 import TrajectoryTrackingControl

from g2_groot_recover_pre_wbc_pose import normalize, qmul
from g2_groot_trajectory_roundtrip_probe import (
    segment_metrics,
    segment_violations,
)
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
from g2_groot_wbc_trajectory_zero_hold_probe import add_pose


CONFIRMATION = "TEST_G2_GROOT_WBC_COMPENSATED_0P5MM"
TOPIC = "/wbc/model_predict"
TCOMP = np.asarray([-0.0009362963, 0.0004766516, 0.0040684294])
QCOMP = normalize([0.0005282369, 0.0066007553, -0.0008167803, 0.9999777417])


def compensated(desired):
    command = desired.copy()
    command[:3] += TCOMP
    command[3:7] = qmul(QCOMP, normalize(desired[3:7]))
    return command


def build_message(desired_poses, trajectory_time_s):
    message = TrajectoryTrackingControl()
    message.trajectory_reference_time = float(trajectory_time_s)
    for desired in desired_poses:
        array = RetargetInfoArray()
        info = array.retarget_infos.add()
        info.group_id = 3
        info.input_type = 54
        info.control_type = 0
        info.frame_id = "base_link"
        info.target_frame_names.append(RIGHT_FRAME)
        add_pose(info, compensated(desired))
        message.retarget_groups_trajectory.append(array)
    return message


def publish_segment(
    publisher,
    tf,
    name,
    left_start,
    right_start,
    segment_start,
    target,
    points,
    trajectory_time_s,
    post_monitor_s,
):
    desired_poses = []
    for index in range(points):
        alpha = float(index + 1) / points
        desired = segment_start.copy()
        desired[:3] = segment_start[:3] * (1 - alpha) + target[:3] * alpha
        desired[3:7] = right_start[3:7]
        desired_poses.append(desired)
    publisher.publish(build_message(desired_poses, trajectory_time_s))
    emit("trajectory_published", segment=name, topic=TOPIC)
    deadline = time.monotonic() + trajectory_time_s + post_monitor_s
    maxima = {}
    first_violation = None
    samples = 0
    while time.monotonic() < deadline:
        left = pose_values(tf, LEFT_FRAME)
        right = pose_values(tf, RIGHT_FRAME)
        values = segment_metrics(left_start, right_start, target, left, right, 0)
        for key, value in values.items():
            maxima[key] = max(maxima.get(key, 0.0), value)
        problems = segment_violations(
            values, translation_distance(right_start, target)
        )
        if problems and first_violation is None:
            first_violation = {"violations": problems, "metrics": values}
            emit("guard_violation", segment=name, **first_violation)
        samples += 1
        time.sleep(0.01)
    left = pose_values(tf, LEFT_FRAME)
    right = pose_values(tf, RIGHT_FRAME)
    final = segment_metrics(left_start, right_start, target, left, right, 0)
    problems = segment_violations(final, translation_distance(right_start, target))
    outcome = {
        "segment": name,
        "passed": first_violation is None and not problems,
        "samples": samples,
        "max": maxima,
        "final": final,
        "final_pose": right.tolist(),
        "first_violation": first_violation,
        "final_violations": problems,
    }
    emit("segment_complete", **outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if args.execute and args.confirm != CONFIRMATION:
        parser.error(f"--execute requires --confirm {CONFIRMATION}")
    if not args.execute and args.confirm:
        parser.error("--confirm is only valid with --execute")
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise RuntimeError("gdk_init failed")
    try:
        robot, tf = agibot_gdk.Robot(), agibot_gdk.TF()
        dds = agibot_gdk.dds
        node = dds.Node("g2_groot_wbc_compensated_roundtrip")
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
        competitors = competing_controllers()
        emit(
            "preflight",
            execute=args.execute,
            start_pose=stable["right"].round(9).tolist(),
            mode=stable["mode"],
            whole=stable["whole"],
            stationary=stationary,
            competing_controllers=competitors,
            translation_compensation_m=TCOMP.tolist(),
            rotation_compensation_xyzw=QCOMP.tolist(),
            test_sequence=["zero_hold", "outbound_x_0p5mm", "return"],
        )
        if max(stationary["left_translation_m"], stationary["right_translation_m"]) > 0.0002:
            raise RuntimeError("robot is not stationary")
        if not args.execute:
            return 0
        if competitors:
            raise RuntimeError(f"competing controllers found: {competitors}")
        for remaining in range(5, 0, -1):
            emit("physical_countdown", remaining_s=remaining)
            time.sleep(1.0)
        left_start = pose_values(tf, LEFT_FRAME)
        right_start = pose_values(tf, RIGHT_FRAME)
        zero = publish_segment(
            publisher, tf, "compensated_zero_hold", left_start, right_start,
            right_start, right_start, 8, 0.8, 0.5,
        )
        if (not zero["passed"] or
                zero["final"]["target_position_error_m"] > 0.0002 or
                zero["final"]["target_rotation_error_rad"] > 0.002):
            raise RuntimeError("compensated zero-hold gate failed; motion withheld")
        target = right_start.copy()
        target[0] += 0.0005
        outbound = publish_segment(
            publisher, tf, "outbound", left_start, right_start,
            np.asarray(zero["final_pose"]), target, 8, 0.8, 0.5,
        )
        if not outbound["passed"]:
            raise RuntimeError("outbound guard failed; return withheld")
        returned = publish_segment(
            publisher, tf, "return", left_start, right_start,
            np.asarray(outbound["final_pose"]), right_start, 8, 0.8, 0.5,
        )
        after = snapshot(robot, tf)
        require_safe_state(after)
        emit(
            "result",
            passed=zero["passed"] and outbound["passed"] and returned["passed"],
            zero_hold=zero,
            outbound=outbound,
            return_segment=returned,
            residual_from_start_translation_m=translation_distance(right_start, after["right"]),
            residual_from_start_rotation_rad=quaternion_angle(right_start, after["right"]),
            final_pose=after["right"].round(9).tolist(),
            mode=after["mode"],
            whole=after["whole"],
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
