#!/usr/bin/env python3
"""Robot-local slow-path proxy with an autonomous return watchdog.

The proxy owns the sole loopback connection to the existing guarded H1 bridge.
It accepts an already safety-checked path from the workstation, validates every
physical substep again, executes and settles it locally, and retains an exact
reverse path.  Loss of workstation keepalives triggers a robot-local return to
the session origin without relying on Wi-Fi.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import json
import math
import queue
import socket
import threading
import time
from typing import Any
import uuid

import numpy as np

from g2_groot_h1_bridge_client import BridgeSession


SCHEMA = "g2_groot_robot_local_slow_path_proxy_v1"
CONFIRMATION = "ENABLE_G2_GROOT_ROBOT_LOCAL_SLOW_PATH_PROXY"
CONTINUOUS_CONFIRMATION = "ENABLE_G2_GROOT_ROBOT_LOCAL_CONTINUOUS_PATH_PROXY"
UNDERLYING_SCHEMA = "g2_groot_persistent_h1_action_bridge_v3"
OPEN_GRIPPER = -0.785
MAX_PATH_POINTS = 96
MAX_SUBSTEP_M = 0.00101
MAX_SUBSTEP_RAD = math.radians(0.251)
CONTINUOUS_MAX_SUBSTEP_M = 0.00181
CONTINUOUS_MAX_SUBSTEP_RAD = math.radians(0.501)
SETTLE_POSITION_M = 0.00015
SETTLE_ROTATION_RAD = 0.0015
SETTLE_TIMEOUT_S = 3.0
MAX_COMMAND_AGE_S = 0.25
WORKSPACE_MIN = np.asarray([0.503, -0.311, 1.005], dtype=np.float64)
WORKSPACE_MAX = np.asarray([0.746, -0.227, 1.081], dtype=np.float64)


def normalize(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid quaternion")
    return quaternion / norm


def rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(first[3:7], second[3:7])))
    return float(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def require_loopback(host: str) -> None:
    addresses = {item[4][0] for item in socket.getaddrinfo(host, None)}
    if not addresses or any(
        not ipaddress.ip_address(address).is_loopback for address in addresses
    ):
        raise ValueError("proxy must bind to loopback")


def recv_line(stream: Any) -> dict | None:
    line = stream.readline(65538)
    if not line:
        return None
    if len(line) > 65536 or not line.endswith(b"\n"):
        raise ValueError("invalid request line")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("request must be an object")
    return value


def send_response(connection: socket.socket, response: dict) -> None:
    connection.sendall(
        (json.dumps(response, separators=(",", ":")) + "\n").encode()
    )


def results_by_id(status: dict) -> dict[str, dict]:
    return {
        str(item["command_id"]): item
        for item in status.get("recent_results", [])
    }


@dataclass
class PathCommand:
    command_id: str
    kind: str
    targets: list[np.ndarray]


class LocalPathWorker:
    def __init__(self, bridge_port: int, watchdog_s: float, continuous: bool = False):
        self.bridge = BridgeSession("127.0.0.1", bridge_port)
        info = self.bridge.request({"op": "info"})
        status = self.bridge.request({"op": "status"})
        if info.get("schema") != UNDERLYING_SCHEMA:
            raise RuntimeError("unexpected underlying bridge schema")
        protected_close = info.get(
            "right_gripper_protected_closure_enabled",
            info.get("right_gripper_full_closure_enabled"),
        )
        if protected_close is not False:
            raise RuntimeError("underlying bridge unexpectedly permits closure")
        if info.get("right_gripper_physical_zero_closure_enabled", False) is not False:
            raise RuntimeError("underlying bridge unexpectedly permits physical zero closure")
        if not status.get("ready") or status.get("fatal_error"):
            raise RuntimeError("underlying bridge is not ready")
        self.origin = np.asarray(status["desired_pose"], dtype=np.float64)
        self.last_target = self.origin.copy()
        self.watchdog_s = watchdog_s
        self.continuous = bool(continuous)
        self.maximum_substep_m = (
            CONTINUOUS_MAX_SUBSTEP_M if self.continuous else MAX_SUBSTEP_M
        )
        self.maximum_substep_rad = (
            CONTINUOUS_MAX_SUBSTEP_RAD if self.continuous else MAX_SUBSTEP_RAD
        )
        self.commands: queue.Queue[PathCommand] = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.busy = False
        self.fatal_error: str | None = None
        self.accepted: list[np.ndarray] = []
        self.last_activity = time.monotonic()
        self.current_command_id: str | None = None
        self.current_kind: str | None = None
        self.path_progress = 0
        self.path_total = 0
        self.last_result: dict | None = None
        self.underlying_status = status
        self.last_underlying_keepalive = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def touch(self) -> None:
        with self.lock:
            self.last_activity = time.monotonic()

    def _validate_path(self, raw: Any) -> list[np.ndarray]:
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_PATH_POINTS:
            raise ValueError(f"path must contain 1..{MAX_PATH_POINTS} points")
        with self.lock:
            previous = self.last_target.copy()
        result = []
        for index, value in enumerate(raw):
            target = np.asarray(value, dtype=np.float64)
            if target.shape != (7,) or not np.isfinite(target).all():
                raise ValueError(f"path point {index} must be finite XYZ+XYZW")
            target[3:7] = normalize(target[3:7])
            if np.any(target[:3] < WORKSPACE_MIN) or np.any(
                target[:3] > WORKSPACE_MAX
            ):
                raise ValueError(f"path point {index} outside pre-close workspace")
            distance = float(np.linalg.norm(target[:3] - previous[:3]))
            angle = rotation_angle(target, previous)
            if distance > self.maximum_substep_m:
                raise ValueError(f"path point {index} translation step too large")
            if angle > self.maximum_substep_rad:
                raise ValueError(f"path point {index} rotation step too large")
            result.append(target.copy())
            previous = target
        return result

    def submit_forward(self, request: dict) -> str:
        timestamp_ns = int(request.get("timestamp_ns", 0))
        age_s = max(0.0, (time.time_ns() - timestamp_ns) / 1e9)
        if timestamp_ns <= 0 or age_s > MAX_COMMAND_AGE_S:
            raise ValueError(f"command age {age_s:.3f}s exceeds limit")
        if abs(float(request.get("target_gripper", 0.0)) - OPEN_GRIPPER) > 1e-6:
            raise ValueError("proxy only permits the fully open gripper")
        targets = self._validate_path(request.get("path"))
        return self._submit(str(request.get("command_id", "")), "forward", targets)

    def submit_return(self, command_id: str) -> str:
        with self.lock:
            targets = [item.copy() for item in reversed(self.accepted[:-1])]
            targets.append(self.origin.copy())
        return self._submit(command_id, "return", targets)

    def submit_shutdown_hold(self, command_id: str) -> str:
        return self._submit(command_id, "shutdown_hold", [])

    def _submit(self, command_id: str, kind: str, targets: list[np.ndarray]) -> str:
        if not command_id or len(command_id) > 128:
            raise ValueError("command_id is required and must be <=128 chars")
        with self.lock:
            if self.fatal_error:
                raise RuntimeError(self.fatal_error)
            if self.busy or not self.commands.empty():
                raise RuntimeError("proxy is busy")
            self.busy = True
            self.current_command_id = command_id
            self.current_kind = kind
            self.path_progress = 0
            self.path_total = len(targets)
            self.last_activity = time.monotonic()
        self.commands.put_nowait(PathCommand(command_id, kind, targets))
        return command_id

    def status(self) -> dict:
        with self.lock:
            remaining = max(0.0, self.watchdog_s - (time.monotonic() - self.last_activity))
            return {
                "ok": True,
                "schema": SCHEMA,
                "ready": self.fatal_error is None and not self.stop_event.is_set(),
                "fatal_error": self.fatal_error,
                "busy": self.busy,
                "current_command_id": self.current_command_id,
                "current_kind": self.current_kind,
                "path_progress": self.path_progress,
                "path_total": self.path_total,
                "accepted_waypoints": len(self.accepted),
                "watchdog_seconds": self.watchdog_s,
                "watchdog_remaining_s": remaining,
                "last_result": self.last_result,
                "underlying_status": self.underlying_status,
            }

    def _wait_underlying(self, command_id: str, require_settled: bool = True) -> dict:
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while time.monotonic() < deadline:
            status = self.bridge.request({"op": "status"})
            self.underlying_status = status
            if status.get("fatal_error"):
                raise RuntimeError(str(status["fatal_error"]))
            result = results_by_id(status).get(command_id)
            if (
                result
                and status.get("queue_depth") == 0
                and (
                    not require_settled
                    or (
                        status.get("live_target_position_error_m")
                        <= SETTLE_POSITION_M
                        and status.get("live_target_rotation_error_rad")
                        <= SETTLE_ROTATION_RAD
                    )
                )
            ):
                if (
                    not require_settled
                    and (
                        result.get("position_error_at_100ms_m", float("inf")) > 0.0015
                        or result.get("rotation_error_at_100ms_rad", float("inf"))
                        > 0.02
                    )
                ):
                    raise RuntimeError("continuous waypoint tracking gate exceeded")
                return result
            time.sleep(0.05)
        raise TimeoutError(f"local waypoint did not settle: {command_id}")

    def _execute_targets(self, command: PathCommand) -> None:
        for index, target in enumerate(command.targets):
            child_id = f"proxy-{command.kind}-{index}-{uuid.uuid4()}"
            acknowledgement = self.bridge.request(
                {
                    "op": "execute_h1_gripper",
                    "command_id": child_id,
                    "timestamp_ns": time.time_ns(),
                    "target_pose": target.tolist(),
                    "target_gripper": OPEN_GRIPPER,
                }
            )
            if not acknowledgement.get("ok"):
                raise RuntimeError(f"underlying command rejected: {acknowledgement}")
            require_settled = not self.continuous or index == len(command.targets) - 1
            self._wait_underlying(child_id, require_settled=require_settled)
            with self.lock:
                self.last_target = target.copy()
                if command.kind == "forward":
                    self.accepted.append(target.copy())
                self.path_progress = index + 1
        if command.kind == "return":
            with self.lock:
                self.accepted.clear()

    def _finish(self, command: PathCommand, status: str) -> None:
        with self.lock:
            self.last_result = {
                "command_id": command.command_id,
                "kind": command.kind,
                "status": status,
                "path_points": len(command.targets),
                "completed_at": time.time(),
            }
            self.busy = False
            self.current_command_id = None
            self.current_kind = None

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    command = self.commands.get(timeout=0.05)
                except queue.Empty:
                    if time.monotonic() - self.last_underlying_keepalive >= 1.0:
                        self.underlying_status = self.bridge.request({"op": "status"})
                        self.last_underlying_keepalive = time.monotonic()
                    with self.lock:
                        watchdog_due = bool(self.accepted) and (
                            time.monotonic() - self.last_activity >= self.watchdog_s
                        )
                        busy = self.busy
                    if watchdog_due and not busy:
                        with self.lock:
                            targets = [
                                item.copy() for item in reversed(self.accepted[:-1])
                            ]
                            targets.append(self.origin.copy())
                            self.busy = True
                            self.current_command_id = "autonomous-watchdog-return"
                            self.current_kind = "watchdog_return"
                            self.path_progress = 0
                            self.path_total = len(targets)
                        command = PathCommand(
                            "autonomous-watchdog-return", "return", targets
                        )
                    else:
                        continue
                if command.kind == "shutdown_hold":
                    self.bridge.request({"op": "shutdown"})
                    self._finish(command, "PASS_SHUTDOWN_HOLD")
                    self.stop_event.set()
                    continue
                self._execute_targets(command)
                result_status = (
                    "PASS_AUTONOMOUS_RETURN"
                    if command.command_id == "autonomous-watchdog-return"
                    else "PASS"
                )
                self._finish(command, result_status)
                if command.command_id == "autonomous-watchdog-return":
                    self.bridge.request({"op": "shutdown"})
                    self.stop_event.set()
                if command.command_id != "autonomous-watchdog-return":
                    self.commands.task_done()
        except Exception as error:
            with self.lock:
                self.fatal_error = f"{type(error).__name__}: {error}"
                self.busy = False
            try:
                self.bridge.request({"op": "shutdown"})
            except Exception:
                pass
            self.stop_event.set()

    def close(self) -> None:
        if not self.stop_event.is_set():
            try:
                self.bridge.request({"op": "shutdown"})
            except Exception:
                pass
            self.stop_event.set()
        self.thread.join(timeout=4.0)
        self.bridge.close()


def serve(host: str, port: int, worker: LocalPathWorker) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(2)
        listener.settimeout(0.5)
        print(json.dumps({"event": "proxy_ready", "schema": SCHEMA, "bind": f"{host}:{port}"}), flush=True)
        while not worker.stop_event.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            with connection, connection.makefile("rb") as stream:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.settimeout(10.0)
                while not worker.stop_event.is_set():
                    try:
                        request = recv_line(stream)
                        if request is None:
                            break
                        operation = str(request.get("op", ""))
                        worker.touch()
                        if operation == "info":
                            response = {
                                "ok": True,
                                "schema": SCHEMA,
                                "operations": ["info", "status", "execute_path", "return_to_origin", "shutdown_hold"],
                                "bind_loopback_only": True,
                                "right_arm_enabled": True,
                                "right_gripper_forced_open": True,
                                "left_arm_enabled": False,
                                "head_waist_chassis_enabled": False,
                                "maximum_path_points": MAX_PATH_POINTS,
                                "continuous_10hz": worker.continuous,
                                "maximum_substep_m": worker.maximum_substep_m,
                                "maximum_substep_rad": worker.maximum_substep_rad,
                                "watchdog_s": worker.watchdog_s,
                            }
                        elif operation == "status":
                            response = worker.status()
                        elif operation == "execute_path":
                            command_id = worker.submit_forward(request)
                            response = {"ok": True, "accepted": True, "command_id": command_id}
                        elif operation == "return_to_origin":
                            command_id = worker.submit_return(str(request.get("command_id", "")))
                            response = {"ok": True, "accepted": True, "command_id": command_id}
                        elif operation == "shutdown_hold":
                            command_id = worker.submit_shutdown_hold(str(request.get("command_id", "")))
                            response = {"ok": True, "accepted": True, "command_id": command_id}
                        else:
                            raise ValueError("unsupported operation")
                    except Exception as error:
                        response = {"ok": False, "error_type": type(error).__name__, "message": str(error)}
                    send_response(connection, response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--bridge-port", type=int, default=9200)
    parser.add_argument("--watchdog-s", type=float, default=8.0)
    args = parser.parse_args()
    require_loopback(args.bind_host)
    required_confirmation = CONTINUOUS_CONFIRMATION if args.continuous else CONFIRMATION
    if not args.enable or args.confirm != required_confirmation:
        parser.error(
            f"proxy requires --enable --confirm {required_confirmation}"
        )
    if not 5.0 <= args.watchdog_s <= 20.0:
        parser.error("--watchdog-s must be in [5, 20]")
    worker = LocalPathWorker(
        args.bridge_port, args.watchdog_s, continuous=args.continuous
    )
    try:
        worker.start()
        serve(args.bind_host, args.port, worker)
        return 0 if worker.fatal_error is None else 2
    finally:
        worker.close()
        print(json.dumps({"event": "proxy_final", **worker.status()}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
