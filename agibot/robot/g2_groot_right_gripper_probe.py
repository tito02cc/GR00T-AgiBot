#!/usr/bin/env python3
"""Fail-closed G2 right omnipicker preflight and tiny round-trip probe.

Default operation is read-only.  Physical execution is deliberately fixed to
one small closing command followed by a return to fully open; arbitrary target
positions, the left tool, arms, head, waist, and chassis are not exposed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import agibot_gdk

from g2_groot_trajectory_tracking_probe import competing_controllers


CONFIRMATION = "EXECUTE_G2_RIGHT_GRIPPER_TINY_ROUNDTRIP"
OPEN_COMMAND = -0.785
TEST_COMMAND = -0.760
RAW_CLOSED = 120.0
MAX_INITIAL_RAW = 2.0
MAX_RAW_POSITION = 122.0
MAX_LEFT_RAW_CHANGE = 0.5
TARGET_RAW_TOLERANCE = 0.5
SETTLE_TIMEOUT_S = 3.0
POLL_PERIOD_S = 0.05


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def raw_from_command(command: float) -> float:
    return RAW_CLOSED * (command - OPEN_COMMAND) / (0.0 - OPEN_COMMAND)


def training_from_raw(raw_position: float) -> float:
    fraction = min(1.0, max(0.0, raw_position / RAW_CLOSED))
    return OPEN_COMMAND * (1.0 - fraction)


def snapshot(robot: Any) -> dict[str, Any]:
    whole = robot.get_whole_body_status()
    ends = robot.get_end_state()
    right = ends["right_end_state"]
    left = ends["left_end_state"]
    if whole["right_end_model"] != "omnipicker":
        raise RuntimeError(f"unexpected right end model: {whole['right_end_model']}")
    if right["names"] != ["right_gripper_joint1"] or len(right["end_states"]) != 1:
        raise RuntimeError("unexpected right omnipicker state schema")
    if left["names"] != ["left_gripper_joint1"] or len(left["end_states"]) != 1:
        raise RuntimeError("unexpected left omnipicker state schema")
    state = {
        "whole": whole,
        "right": {**right, "end_states": [dict(right["end_states"][0])]},
        "left": {**left, "end_states": [dict(left["end_states"][0])]},
    }
    require_safe(state)
    return state


def require_safe(state: dict[str, Any]) -> None:
    whole = state["whole"]
    error_fields = (
        "right_arm_error",
        "left_arm_error",
        "right_end_error",
        "left_end_error",
        "waist_error",
        "lift_error",
        "neck_error",
        "chassis_error",
    )
    errors = {name: int(whole[name]) for name in error_fields if int(whole[name]) != 0}
    if errors:
        raise RuntimeError(f"whole-body errors: {errors}")
    for side in ("right", "left"):
        motor = state[side]["end_states"][0]
        if not motor["enable"] or int(motor["err_code"]) != 0:
            raise RuntimeError(f"unsafe {side} tool motor state: {motor}")
        if int(motor["status"]) not in (0, 1):
            raise RuntimeError(f"unknown {side} tool motor status: {motor}")
        position = float(motor["position"])
        if not math.isfinite(position) or not -0.5 <= position <= MAX_RAW_POSITION:
            raise RuntimeError(f"invalid {side} tool feedback position: {position}")


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    right = state["right"]["end_states"][0]
    left = state["left"]["end_states"][0]
    return {
        "whole": state["whole"],
        "right_tool": {
            "model": "omnipicker",
            "name": state["right"]["names"][0],
            "controlled": bool(state["right"]["controlled"]),
            "raw_position": float(right["position"]),
            "training_position": training_from_raw(float(right["position"])),
            "velocity": float(right["velocity"]),
            "effort": float(right["effort"]),
            "err_code": int(right["err_code"]),
        },
        "left_tool_monitor": {
            "raw_position": float(left["position"]),
            "velocity": float(left["velocity"]),
            "effort": float(left["effort"]),
            "err_code": int(left["err_code"]),
        },
    }


def command_right_tool(robot: Any, position: float) -> None:
    if position not in (OPEN_COMMAND, TEST_COMMAND):
        raise ValueError("probe only permits its fixed open and tiny-close commands")
    joint = agibot_gdk.JointState()
    joint.position = position
    joints = agibot_gdk.JointStates()
    joints.group = "right_tool"
    joints.target_type = "omnipicker"
    joints.states = [joint]
    joints.nums = 1
    result = robot.move_ee_pos(joints)
    if result != 0:
        raise RuntimeError(f"move_ee_pos returned {result}")


def wait_for_target(robot: Any, target_raw: float, left_start_raw: float) -> dict[str, Any]:
    started = time.monotonic()
    samples = []
    while True:
        state = snapshot(robot)
        right = float(state["right"]["end_states"][0]["position"])
        left = float(state["left"]["end_states"][0]["position"])
        sample = {
            "elapsed_s": time.monotonic() - started,
            "right_raw_position": right,
            "left_raw_position": left,
            "right_velocity": float(state["right"]["end_states"][0]["velocity"]),
            "right_effort": float(state["right"]["end_states"][0]["effort"]),
            "right_status": int(state["right"]["end_states"][0]["status"]),
            "right_err_code": int(state["right"]["end_states"][0]["err_code"]),
        }
        samples.append(sample)
        if abs(left - left_start_raw) > MAX_LEFT_RAW_CHANGE:
            raise RuntimeError("left tool moved during right-only probe")
        error = abs(right - target_raw)
        if (
            error <= TARGET_RAW_TOLERANCE
            and abs(sample["right_velocity"]) <= 0.5
            and sample["right_status"] == 0
        ):
            return {
                "target_raw_position": target_raw,
                "settled": True,
                "settle_s": sample["elapsed_s"],
                "final_raw_error": error,
                "samples": samples,
            }
        if sample["elapsed_s"] >= SETTLE_TIMEOUT_S:
            return {
                "target_raw_position": target_raw,
                "settled": False,
                "settle_s": sample["elapsed_s"],
                "final_raw_error": error,
                "samples": samples,
            }
        time.sleep(POLL_PERIOD_S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if args.execute and args.confirm != CONFIRMATION:
        parser.error(f"--execute requires --confirm {CONFIRMATION}")
    if not args.execute and args.confirm:
        parser.error("--confirm is only valid with --execute")
    return args


def main() -> int:
    args = parse_args()
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise RuntimeError("gdk_init failed")
    robot = agibot_gdk.Robot()
    try:
        time.sleep(2.0)
        before = snapshot(robot)
        competitors = competing_controllers()
        summary = summarize(before)
        summary.update(
            {
                "execute": args.execute,
                "competing_controllers": competitors,
                "command_range": [OPEN_COMMAND, 0.0],
                "fixed_test_command": TEST_COMMAND,
                "fixed_test_raw_target": raw_from_command(TEST_COMMAND),
            }
        )
        emit("right_gripper_preflight", **summary)
        if not args.execute:
            emit("read_only_complete", motor_commands_sent=0)
            return 0
        if competitors:
            raise RuntimeError(f"competing controllers found: {competitors}")
        initial_raw = float(before["right"]["end_states"][0]["position"])
        left_start_raw = float(before["left"]["end_states"][0]["position"])
        if initial_raw > MAX_INITIAL_RAW:
            raise RuntimeError(
                f"right tool is not initially open: raw position {initial_raw}"
            )
        for remaining in range(5, 0, -1):
            emit("physical_countdown", remaining_s=remaining)
            time.sleep(1.0)

        close_result = None
        return_result = None
        try:
            command_right_tool(robot, TEST_COMMAND)
            close_result = wait_for_target(
                robot, raw_from_command(TEST_COMMAND), left_start_raw
            )
        finally:
            command_right_tool(robot, OPEN_COMMAND)
            return_result = wait_for_target(robot, 0.0, left_start_raw)

        final = snapshot(robot)
        passed = bool(
            close_result
            and close_result["settled"]
            and return_result
            and return_result["settled"]
        )
        emit(
            "physical_result",
            status="PASS_TINY_RIGHT_GRIPPER_ROUNDTRIP" if passed else "FAIL",
            close=close_result,
            return_open=return_result,
            final=summarize(final),
            motor_commands_sent=2,
        )
        return 0 if passed else 1
    finally:
        agibot_gdk.gdk_release()


if __name__ == "__main__":
    raise SystemExit(main())
