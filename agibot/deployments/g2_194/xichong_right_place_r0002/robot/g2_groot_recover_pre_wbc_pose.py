#!/usr/bin/env python3
"""One-purpose guarded recovery to the pre-WBC-test right EEF pose."""

import json
import math
import os
import time

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


CONFIRMATION = "RECOVER_G2_GROOT_PRE_WBC_POSE"
TARGET = np.asarray(
    [
        0.508551137,
        -0.259144854,
        1.048253966,
        0.661883078,
        -0.020726068,
        0.749220127,
        0.012264684,
    ],
    dtype=np.float64,
)


def normalize(q):
    return np.asarray(q, dtype=np.float64) / np.linalg.norm(q)


def clamp(v, limit):
    n = float(np.linalg.norm(v))
    return v if n <= limit or n == 0.0 else v * (limit / n)


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return normalize(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def qinv(q):
    q = normalize(q)
    return np.asarray([-q[0], -q[1], -q[2], q[3]])


def q_to_rv(q):
    q = normalize(q)
    if q[3] < 0:
        q = -q
    s = float(np.linalg.norm(q[:3]))
    if s < 1e-12:
        return np.zeros(3)
    return q[:3] * (2.0 * math.atan2(s, float(q[3])) / s)


def rv_to_q(v):
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0])
    axis = v / angle
    return np.asarray(
        [
            axis[0] * math.sin(angle / 2),
            axis[1] * math.sin(angle / 2),
            axis[2] * math.sin(angle / 2),
            math.cos(angle / 2),
        ]
    )


def send(robot, pose):
    command = agibot_gdk.EndEffectorPose()
    command.life_time = 0.04
    command.group = agibot_gdk.EndEffectorControlGroup.kRightArm
    out = command.right_end_effector_pose
    out.position.x, out.position.y, out.position.z = map(float, pose[:3])
    (
        out.orientation.x,
        out.orientation.y,
        out.orientation.z,
        out.orientation.w,
    ) = map(float, pose[3:7])
    result = robot.end_effector_pose_control(command)
    if int(result) != 0:
        raise RuntimeError(f"end_effector_pose_control result={result}")


def main():
    import argparse

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
        time.sleep(2.0)
        state = snapshot(robot, tf)
        require_safe_state(state)
        current = state["right"]
        distance = translation_distance(current, TARGET)
        rotation = quaternion_angle(current, TARGET)
        competitors = competing_controllers()
        emit(
            "preflight",
            execute=args.execute,
            current_pose=current.round(9).tolist(),
            recovery_target=TARGET.round(9).tolist(),
            target_distance_m=distance,
            target_rotation_rad=rotation,
            competing_controllers=competitors,
        )
        if distance > 0.005 or rotation > 0.02:
            raise RuntimeError("current pose outside recovery envelope")
        if not args.execute:
            return 0
        if competitors:
            raise RuntimeError(f"competing controllers found: {competitors}")
        for remaining in range(5, 0, -1):
            emit("physical_countdown", remaining_s=remaining)
            time.sleep(1.0)
        left_start = pose_values(tf, LEFT_FRAME)
        right_start = pose_values(tf, RIGHT_FRAME)
        # Seed from the successful first recovery so repeated recovery does not
        # reproduce the controller's roughly 4 mm cold-start pose bias.
        tcomp = np.asarray([-0.0009362963, 0.0004766516, 0.0040684294])
        rcomp = normalize(
            [0.0005282369, 0.0066007553, -0.0008167803, 0.9999777417]
        )
        maxima = {"left_translation_m": 0.0, "right_from_start_m": 0.0}
        deadline = time.monotonic()
        for index in range(200):
            live = pose_values(tf, RIGHT_FRAME)
            left = pose_values(tf, LEFT_FRAME)
            tcomp = clamp(tcomp + clamp(0.12 * (TARGET[:3] - live[:3]), 0.00015), 0.0065)
            qerr = qmul(normalize(TARGET[3:7]), qinv(live[3:7]))
            rstep = clamp(0.10 * q_to_rv(qerr), 0.0005)
            rcomp = qmul(rv_to_q(rstep), rcomp)
            if quaternion_angle(np.asarray([0, 0, 0, 0, 0, 0, 1.0]), np.r_[np.zeros(3), rcomp]) > 0.05:
                raise RuntimeError("rotation compensation exceeded limit")
            command = TARGET.copy()
            command[:3] += tcomp
            command[3:7] = qmul(rcomp, normalize(TARGET[3:7]))
            send(robot, command)
            lm = translation_distance(left_start, left)
            rm = translation_distance(right_start, live)
            maxima["left_translation_m"] = max(maxima["left_translation_m"], lm)
            maxima["right_from_start_m"] = max(maxima["right_from_start_m"], rm)
            if lm > 0.0005 or rm > 0.006:
                raise RuntimeError(f"recovery guard violation left={lm} right={rm}")
            deadline += 0.02
            wait = deadline - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        time.sleep(0.2)
        after = snapshot(robot, tf)
        require_safe_state(after)
        final_error = translation_distance(after["right"], TARGET)
        final_rotation = quaternion_angle(after["right"], TARGET)
        emit(
            "result",
            passed=final_error <= 0.0002 and final_rotation <= 0.002,
            final_pose=after["right"].round(9).tolist(),
            target_position_error_m=final_error,
            target_rotation_error_rad=final_rotation,
            translation_compensation_m=tcomp.tolist(),
            rotation_compensation_xyzw=rcomp.tolist(),
            max=maxima,
            mode=after["mode"],
            whole=after["whole"],
        )
        return 0 if final_error <= 0.0002 and final_rotation <= 0.002 else 2
    finally:
        agibot_gdk.gdk_release()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        emit("fatal", error_type=type(error).__name__, message=str(error))
        raise
