#!/usr/bin/env python3
"""Small JSON-line client for the loopback-only G2 H1 action bridge."""

from __future__ import annotations

import argparse
import json
import socket
import time
import uuid


CONFIRMATION = "SEND_G2_GROOT_RIGHT_ARM_H1"
GRIPPER_CONFIRMATION = "SET_G2_GROOT_RIGHT_GRIPPER"
BRIDGE_RESPONSE_TIMEOUT_S = 60.0
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class BridgeSession:
    def __init__(self, host: str, port: int):
        self.connection = socket.create_connection((host, port), timeout=BRIDGE_RESPONSE_TIMEOUT_S)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection.settimeout(BRIDGE_RESPONSE_TIMEOUT_S)
        self.stream = self.connection.makefile("rb")

    def request(self, payload: dict) -> dict:
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        self.connection.sendall(encoded)
        line = self.stream.readline(MAX_RESPONSE_BYTES + 1)
        if not line or len(line) > MAX_RESPONSE_BYTES:
            raise RuntimeError("invalid or missing bridge response")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("bridge response is not an object")
        return response

    def close(self):
        self.stream.close()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def request(host: str, port: int, payload: dict) -> dict:
    with BridgeSession(host, port) as session:
        return session.request(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op", choices=("info", "status", "execute_h1", "set_gripper", "shutdown"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--command-id", default="")
    parser.add_argument("--target-pose", type=float, nargs=7)
    parser.add_argument("--target-gripper", type=float)
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    payload: dict = {"op": args.op}
    if args.op == "set_gripper":
        if args.confirm != GRIPPER_CONFIRMATION:
            parser.error(f"set_gripper requires --confirm {GRIPPER_CONFIRMATION}")
        if args.target_gripper is None:
            parser.error("set_gripper requires --target-gripper")
        if args.target_pose is not None or args.command_id:
            parser.error("set_gripper does not accept an arm pose or command id")
        payload["target_gripper"] = args.target_gripper
    elif args.op == "execute_h1":
        if args.target_gripper is not None:
            parser.error("--target-gripper is only valid for set_gripper")
        if args.confirm != CONFIRMATION:
            parser.error(f"execute_h1 requires --confirm {CONFIRMATION}")
        if args.target_pose is None:
            parser.error("execute_h1 requires --target-pose X Y Z QX QY QZ QW")
        payload.update(
            {
                "command_id": args.command_id or str(uuid.uuid4()),
                "timestamp_ns": time.time_ns(),
                "target_pose": args.target_pose,
            }
        )
    elif (
        args.target_pose is not None
        or args.target_gripper is not None
        or args.confirm
        or args.command_id
    ):
        parser.error("target/confirmation/command-id are only valid for execute_h1")

    response = request(args.host, args.port, payload)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
