#!/usr/bin/env bash
# Read-only right-arm state probe.
#
# Reads GDK motion status, whole-body status and the right EEF pose without
# creating any publisher or issuing any motion command.  Also reports whether
# GDK exposes right-arm joint angles, which the observation bridge does not
# currently capture and which the collision diagnosis needs.

# ``env.sh`` dereferences $1, so no nounset here.
source /home/agi/app/env.sh
cd /home/agi/vla_ct/bridges/10.20.15.194

python3 - <<'PY'
import json
import sys

sys.path.insert(0, "/home/agi/vla_ct/bridges/10.20.15.194")

import agibot_gdk
from g2_groot_trajectory_tracking_probe import (
    LEFT_FRAME,
    RIGHT_FRAME,
    competing_controllers,
    snapshot,
)

WORKSPACE_MIN = (0.4172, -0.2392, 0.9750)
WORKSPACE_MAX = (0.8403, -0.1031, 1.2674)
TRAINING_START = (0.5013904571533203, -0.1765020340681076, 1.0539140701293945)

if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
    raise SystemExit("gdk_init failed")
try:
    import time

    robot, tf = agibot_gdk.Robot(), agibot_gdk.TF()
    time.sleep(2.0)
    state = snapshot(robot, tf)

    print("motion_mode      :", state["mode"])
    print("motion_error     :", state["motion_error"])
    print("motion_error_msg :", repr(state["motion_error_msg"]))
    print("whole_body       :", json.dumps(state["whole"], default=str))
    print("right_eef_frame  :", RIGHT_FRAME)
    print("right_eef_pose   :", [round(v, 6) for v in state["right"].tolist()])
    print("left_eef_pose    :", [round(v, 6) for v in state["left"].tolist()])

    xyz = state["right"][:3]
    inside = all(
        WORKSPACE_MIN[i] <= xyz[i] <= WORKSPACE_MAX[i] for i in range(3)
    )
    print("workspace_min    :", WORKSPACE_MIN)
    print("workspace_max    :", WORKSPACE_MAX)
    print("inside_workspace :", inside)
    print(
        "offset_from_training_start_m :",
        round(sum((xyz[i] - TRAINING_START[i]) ** 2 for i in range(3)) ** 0.5, 6),
    )
    print("competing_controllers :", json.dumps(competing_controllers()))

    print()
    print("=== joint-angle availability (collision-diagnosis instrumentation) ===")
    for name in sorted(dir(robot)):
        if "joint" in name.lower() or "arm_state" in name.lower():
            print("  robot." + name)
    for getter in ("get_joint_state", "get_arm_state", "get_joint_states"):
        if hasattr(robot, getter):
            try:
                value = getattr(robot, getter)()
                text = json.dumps(value, default=str)
                print(f"  {getter}() -> {text[:600]}")
            except Exception as error:
                print(f"  {getter}() raised {type(error).__name__}: {error}")
finally:
    agibot_gdk.gdk_release()
PY
