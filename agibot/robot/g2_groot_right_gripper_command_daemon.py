#!/usr/bin/env python3
"""Loopback-only GDK owner for right-omnipicker position commands."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import socket
import time

import agibot_gdk


CONFIRMATION = "ENABLE_G2_GROOT_RIGHT_GRIPPER_COMMAND_DAEMON"
OPEN_COMMAND = -0.785
CLOSED_COMMAND = 0.0
# The policy path clips the gripper action to [-0.785, 0] in float64 and then
# stores it in a float32 action array.  -0.785 has no exact float32
# representation, so a saturated open command arrives here as
# -0.7850000262260437, which is 2.6e-8 past the actuator bound.  Absorb that
# representation error instead of rejecting the model's own open command; a
# value further out than this tolerance is still a real range violation.
COMMAND_BOUND_TOLERANCE = 1e-4
MAX_LINE_BYTES = 4096
CLIENT_TIMEOUT_S = 2.0


def require_loopback(host: str) -> None:
    addresses = {item[4][0] for item in socket.getaddrinfo(host, None)}
    if not addresses or any(
        not ipaddress.ip_address(address).is_loopback for address in addresses
    ):
        raise ValueError("gripper daemon must bind to a loopback address")


def clamp_command(target: float) -> float:
    """Map a policy gripper target onto the actuator's representable range."""
    value = float(target)
    if not math.isfinite(value):
        raise ValueError("non-finite omnipicker target")
    if (
        value < OPEN_COMMAND - COMMAND_BOUND_TOLERANCE
        or value > CLOSED_COMMAND + COMMAND_BOUND_TOLERANCE
    ):
        raise ValueError(
            f"target {value!r} outside [{OPEN_COMMAND}, {CLOSED_COMMAND}]"
        )
    return min(CLOSED_COMMAND, max(OPEN_COMMAND, value))


def send_gripper(robot: object, target: float) -> dict:
    commanded = clamp_command(target)
    joint = agibot_gdk.JointState()
    joint.position = commanded
    joints = agibot_gdk.JointStates()
    joints.group = "right_tool"
    joints.target_type = "omnipicker"
    joints.states = [joint]
    joints.nums = 1
    result_code = int(robot.move_ee_pos(joints))
    if result_code != 0:
        raise RuntimeError(f"move_ee_pos returned {result_code}")
    return {"result": result_code, "commanded": commanded}


def read_gripper(robot: object) -> dict:
    whole = robot.get_whole_body_status()
    right = robot.get_end_state()["right_end_state"]
    motor = right["end_states"][0]
    return {
        "position": float(motor["position"]),
        "status": int(motor["status"]),
        "err_code": int(motor["err_code"]),
        "whole_end_error": int(whole["right_end_error"]),
        "effort": float(motor["effort"]),
        "names": list(right["names"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--session-limit-s", type=float, default=1800.0)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    require_loopback(args.bind_host)
    if args.confirm != CONFIRMATION:
        parser.error(f"physical control requires --confirm {CONFIRMATION}")
    if not 30 <= args.session_limit_s <= 3600:
        parser.error("--session-limit-s must be in [30, 3600]")
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise RuntimeError("gdk_init failed")
    try:
        robot = agibot_gdk.Robot()
        time.sleep(2.0)
        deadline = time.monotonic() + args.session_limit_s
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.bind_host, args.port))
            listener.listen(4)
            listener.settimeout(0.5)
            print(json.dumps({"event": "ready", "port": args.port}), flush=True)
            stopping = False
            while time.monotonic() < deadline and not stopping:
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    connection.close()
                    break
                connection.settimeout(min(CLIENT_TIMEOUT_S, remaining_s))
                with connection, connection.makefile("rb") as stream:
                    try:
                        line = stream.readline(MAX_LINE_BYTES + 1)
                    except (ConnectionError, socket.timeout):
                        continue
                    if not line:
                        continue
                    try:
                        if not line or len(line) > MAX_LINE_BYTES:
                            raise ValueError("invalid request")
                        request = json.loads(line)
                        operation = request.get("op")
                        if operation == "command":
                            target = float(request["target"])
                            outcome = send_gripper(robot, target)
                            response = {"ok": True, "target": target, **outcome}
                        elif operation == "ping":
                            response = {"ok": True, "mode": "right_gripper_only"}
                        elif operation == "status":
                            response = {"ok": True, **read_gripper(robot)}
                        elif operation == "shutdown":
                            response = {"ok": True, "shutdown": True}
                            stopping = True
                        else:
                            raise ValueError("unsupported operation")
                    except Exception as error:
                        response = {
                            "ok": False,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                    try:
                        connection.sendall((json.dumps(response) + "\n").encode())
                    except (ConnectionError, socket.timeout):
                        continue
        return 0
    finally:
        agibot_gdk.gdk_release()


if __name__ == "__main__":
    raise SystemExit(main())
