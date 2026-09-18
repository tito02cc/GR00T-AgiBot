#!/usr/bin/env python3
"""Fixed, fail-open right-omnipicker empty-close baseline probe.

This probe is deliberately not a generic gripper CLI.  Physical execution uses
fixed small steps ending at the training median closed command, holds briefly,
then returns fully open.  It never exposes arm or any other robot group.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import agibot_gdk

from g2_groot_right_gripper_probe import (
    MAX_INITIAL_RAW,
    MAX_LEFT_RAW_CHANGE,
    OPEN_COMMAND,
    POLL_PERIOD_S,
    RAW_CLOSED,
    competing_controllers,
    raw_from_command,
    snapshot,
    summarize,
)


CONFIRMATION = "EXECUTE_G2_RIGHT_GRIPPER_EMPTY_CLOSE_BASELINE"
FIXED_TARGETS = (
    -0.720,
    -0.655,
    -0.590,
    -0.525,
    -0.460,
    -0.395,
    -0.330,
    -0.265,
    -0.200,
    -0.135,
    -0.070,
    -0.008070454,
)
MAX_ABS_EFFORT = 25.0
TARGET_RAW_TOLERANCE = 1.5
STEP_TIMEOUT_S = 2.0
HOLD_S = 1.0
OPEN_TIMEOUT_S = 3.0


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def command_right_tool(robot: Any, position: float) -> None:
    if position not in (*FIXED_TARGETS, OPEN_COMMAND):
        raise ValueError("probe only permits its fixed target sequence and open")
    joint = agibot_gdk.JointState()
    joint.position = float(position)
    joints = agibot_gdk.JointStates()
    joints.group = "right_tool"
    joints.target_type = "omnipicker"
    joints.states = [joint]
    joints.nums = 1
    result = robot.move_ee_pos(joints)
    if result != 0:
        raise RuntimeError(f"move_ee_pos returned {result}")


def sample(state: dict[str, Any], started: float) -> dict[str, Any]:
    right = state["right"]["end_states"][0]
    left = state["left"]["end_states"][0]
    return {
        "elapsed_s": time.monotonic() - started,
        "right_raw_position": float(right["position"]),
        "right_velocity": float(right["velocity"]),
        "right_effort": float(right["effort"]),
        "right_current": float(right.get("current", 0.0)),
        "right_voltage": float(right.get("voltage", 0.0)),
        "right_temperature": float(right.get("temperature", 0.0)),
        "right_status": int(right["status"]),
        "right_err_code": int(right["err_code"]),
        "left_raw_position": float(left["position"]),
    }


def require_probe_limits(
    row: dict[str, Any], left_start_raw: float, *, enforce_effort: bool
) -> None:
    if not math.isfinite(row["right_effort"]):
        raise RuntimeError("non-finite right gripper effort")
    if enforce_effort and abs(row["right_effort"]) > MAX_ABS_EFFORT:
        raise RuntimeError(
            f"right gripper effort {row['right_effort']:.3f} exceeds "
            f"{MAX_ABS_EFFORT:.3f}"
        )
    if abs(row["left_raw_position"] - left_start_raw) > MAX_LEFT_RAW_CHANGE:
        raise RuntimeError("left tool moved during right-only probe")


def wait_fixed_target(
    robot: Any,
    command: float,
    left_start_raw: float,
    timeout_s: float,
    *,
    enforce_effort: bool = True,
) -> dict[str, Any]:
    target_raw = raw_from_command(command)
    started = time.monotonic()
    samples = []
    saw_transition = False
    while True:
        state = snapshot(robot)
        row = sample(state, started)
        samples.append(row)
        try:
            require_probe_limits(
                row, left_start_raw, enforce_effort=enforce_effort
            )
        except Exception as error:
            return {
                "command": command,
                "target_raw_position": target_raw,
                "settled": False,
                "saw_transition": saw_transition,
                "settle_s": row["elapsed_s"],
                "final_raw_error": abs(row["right_raw_position"] - target_raw),
                "maximum_abs_effort": max(
                    abs(x["right_effort"]) for x in samples
                ),
                "limit_violation": f"{type(error).__name__}: {error}",
                "samples": samples,
            }
        saw_transition = saw_transition or row["right_status"] == 1
        error = abs(row["right_raw_position"] - target_raw)
        if row["right_status"] == 0 and error <= TARGET_RAW_TOLERANCE:
            return {
                "command": command,
                "target_raw_position": target_raw,
                "settled": True,
                "saw_transition": saw_transition,
                "settle_s": row["elapsed_s"],
                "final_raw_error": error,
                "maximum_abs_effort": max(abs(x["right_effort"]) for x in samples),
                "samples": samples,
            }
        if row["elapsed_s"] >= timeout_s:
            return {
                "command": command,
                "target_raw_position": target_raw,
                "settled": False,
                "saw_transition": saw_transition,
                "settle_s": row["elapsed_s"],
                "final_raw_error": error,
                "maximum_abs_effort": max(abs(x["right_effort"]) for x in samples),
                "samples": samples,
            }
        time.sleep(POLL_PERIOD_S)


def hold_samples(robot: Any, left_start_raw: float) -> list[dict[str, Any]]:
    started = time.monotonic()
    rows = []
    while time.monotonic() - started < HOLD_S:
        state = snapshot(robot)
        row = sample(state, started)
        require_probe_limits(row, left_start_raw, enforce_effort=True)
        if row["right_status"] != 0:
            raise RuntimeError("right gripper left settled state during hold")
        rows.append(row)
        time.sleep(POLL_PERIOD_S)
    return rows


def end_is_healthy(robot: Any) -> bool:
    try:
        state = snapshot(robot)
    except Exception:
        return False
    motor = state["right"]["end_states"][0]
    return bool(
        motor["enable"]
        and int(motor["err_code"]) == 0
        and int(state["whole"]["right_end_error"]) == 0
    )


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
    motor_commands_sent = 0
    try:
        time.sleep(2.0)
        before = snapshot(robot)
        competitors = competing_controllers()
        emit(
            "empty_close_preflight",
            execute=args.execute,
            fixed_targets=list(FIXED_TARGETS),
            maximum_abs_effort=MAX_ABS_EFFORT,
            before=summarize(before),
            competing_controllers=competitors,
        )
        if not args.execute:
            emit("read_only_complete", motor_commands_sent=0)
            return 0
        if competitors:
            raise RuntimeError(f"competing controllers found: {competitors}")
        initial_raw = float(before["right"]["end_states"][0]["position"])
        left_start_raw = float(before["left"]["end_states"][0]["position"])
        if initial_raw > MAX_INITIAL_RAW:
            raise RuntimeError(f"right gripper is not open: raw={initial_raw}")
        for remaining in range(5, 0, -1):
            emit("physical_countdown", remaining_s=remaining)
            time.sleep(1.0)

        results = []
        hold = []
        failure: str | None = None
        try:
            for target in FIXED_TARGETS:
                command_right_tool(robot, target)
                motor_commands_sent += 1
                result = wait_fixed_target(
                    robot, target, left_start_raw, STEP_TIMEOUT_S
                )
                results.append(result)
                emit("empty_close_step", **result)
                if not result["settled"]:
                    raise RuntimeError(
                        f"target {target} did not settle: "
                        f"{result.get('limit_violation', 'timeout')}"
                    )
            hold = hold_samples(robot, left_start_raw)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        finally:
            if end_is_healthy(robot):
                command_right_tool(robot, OPEN_COMMAND)
                motor_commands_sent += 1
                open_result = wait_fixed_target(
                    robot,
                    OPEN_COMMAND,
                    left_start_raw,
                    OPEN_TIMEOUT_S,
                    enforce_effort=False,
                )
            else:
                open_result = {
                    "settled": False,
                    "recovery_command_sent": False,
                    "reason": "end_not_healthy",
                }

        final = snapshot(robot)
        passed = bool(
            failure is None
            and len(results) == len(FIXED_TARGETS)
            and all(x["settled"] for x in results)
            and hold
            and open_result.get("settled")
        )
        emit(
            "empty_close_result",
            status="PASS_EMPTY_CLOSE_BASELINE" if passed else "FAIL",
            failure=failure,
            close_steps=results,
            closed_hold=hold,
            return_open=open_result,
            final=summarize(final),
            motor_commands_sent=motor_commands_sent,
        )
        return 0 if passed else 1
    finally:
        agibot_gdk.gdk_release()


if __name__ == "__main__":
    raise SystemExit(main())
