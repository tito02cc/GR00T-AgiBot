#!/usr/bin/env python3
"""Loopback-only persistent 50 Hz H1 action bridge for the G2 right arm.

Physical startup requires an exact CLI confirmation.  Once calibrated, one
worker exclusively owns all GDK control calls and continuously refreshes the
right-arm hold at 50 Hz.  The socket protocol accepts sequential 10 Hz
absolute EEF waypoints over one persistent connection.  A separately confirmed
full-close mode carries the model's absolute right-gripper action in the native
[-0.785, 0.0] range.  Left arm, head, waist, and chassis remain absent.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import gc
import ipaddress
import json
import queue
import socket
import sys
import threading
import time
from typing import Any

import agibot_gdk
from g2_groot_gripper_runtime import (
    PROTECTED_MAXIMUM_COMMAND,
    PROVISIONAL_MAXIMUM_COMMAND,
    GripperRuntime,
)
from g2_groot_gripper_state_machine import (
    FEEDBACK_ENCODINGS,
    FEEDBACK_RAW_0_120,
    OPEN_COMMAND,
    GripperObservation,
    feedback_to_command,
)
from g2_groot_persistent_right_arm_controller import (
    APPROACH_SESSION_MAX_ROTATION_RAD,
    APPROACH_SESSION_MAX_TRANSLATION_M,
    CONTROL_HZ,
    MAX_ALLOWED_SESSION_ROTATION_RAD,
    MAX_ALLOWED_SESSION_TRANSLATION_M,
    MAX_TARGET_ROTATION_RAD,
    MAX_TARGET_STEP_M,
    MODEL_PERIOD_S,
    RIGHT_SESSION_MAX_ROTATION_RAD,
    RIGHT_SESSION_MAX_TRANSLATION_M,
    WORKSPACE_MAX,
    WORKSPACE_MIN,
    PersistentRightArmController,
)
from g2_groot_recover_pre_wbc_pose import normalize
from g2_groot_trajectory_tracking_probe import (
    competing_controllers,
    emit,
    pose_values,
    quaternion_angle,
    require_safe_state,
    snapshot,
    translation_distance,
)
import numpy as np


ARM_ONLY_SCHEMA = "g2_groot_persistent_h1_action_bridge_v2"
ARM_GRIPPER_SCHEMA = "g2_groot_persistent_h1_action_bridge_v3"
CONFIRMATION = "ENABLE_G2_GROOT_PERSISTENT_RIGHT_ARM_H1"
GRIPPER_CONFIRMATION = "ENABLE_G2_GROOT_RIGHT_ARM_H1_WITH_TINY_GRIPPER"
APPROACH_CONFIRMATION = "ENABLE_G2_GROOT_RIGHT_ARM_OPEN_GRIPPER_APPROACH_SESSION"
PROTECTED_CLOSE_CONFIRMATION = (
    "ENABLE_G2_GROOT_RIGHT_ARM_SINGLE_PROTECTED_CLOSE_SESSION"
)
RIGHT_FRAME = "arm_r_end_link"
MAX_LINE_BYTES = 16384
MAX_COMMAND_AGE_S = 5.0
DEFAULT_SESSION_LIMIT_S = 300.0
STATUS_LOCK_TIMEOUT_S = 0.25
# Arm EEF commands use a 40 ms lifetime. Leave a wider quiet interval before
# move_ee_pos so this GDK release relinquishes the shared motion owner.
GRIPPER_PRE_COMMAND_QUIET_S = 0.12


def require_loopback(host: str):
    addresses = {item[4][0] for item in socket.getaddrinfo(host, None)}
    if not addresses or any(
        not ipaddress.ip_address(address).is_loopback for address in addresses
    ):
        raise ValueError("action bridge must bind to a loopback address")


def recv_line(stream: Any) -> dict[str, Any] | None:
    line = stream.readline(MAX_LINE_BYTES + 2)
    if not line:
        return None
    if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
        raise ValueError("request exceeds maximum size or lacks newline")
    payload = json.loads(line.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    return payload


def send_response(connection: socket.socket, payload: dict[str, Any]):
    connection.sendall(
        (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    )


def read_right_gripper_observation(
    robot: Any, joint_name: str
) -> GripperObservation:
    whole = robot.get_whole_body_status()
    ends = robot.get_end_state()
    right = ends["right_end_state"]
    if whole["right_end_model"] != "omnipicker":
        raise RuntimeError(f"unexpected right end model: {whole['right_end_model']}")
    if right["names"] != [joint_name] or len(right["end_states"]) != 1:
        raise RuntimeError(
            "unexpected right omnipicker feedback schema: "
            f"names={right['names']!r}, expected={[joint_name]!r}"
        )
    motor = right["end_states"][0]
    if not motor["enable"]:
        raise RuntimeError("right omnipicker motor is disabled")
    return GripperObservation(
        monotonic_s=time.monotonic(),
        raw_position=float(motor["position"]),
        motor_status=int(motor["status"]),
        motor_err_code=int(motor["err_code"]),
        whole_end_error=int(whole["right_end_error"]),
        effort=float(motor["effort"]),
    )


def send_right_gripper_command(
    robot: Any, target: float, maximum_allowed_command: float
) -> None:
    if not OPEN_COMMAND <= target <= maximum_allowed_command:
        raise ValueError("right gripper target exceeds enabled command range")
    # GDK 3.3.8 exposes two different tool-control paths. move_ee_pos is a
    # transaction that conflicts with an active end_effector_pose_control
    # publisher. The documented servo API is the compatible path when arm and
    # tool commands are interleaved by a VLA policy.
    result = robot.move_end_effector_joint(
        [float(target)], [0.0], "right_tool"
    )
    if result != 0:
        raise RuntimeError(f"move_end_effector_joint returned {result}")


def send_right_gripper_command_external(
    port: int, target: float, maximum_allowed_command: float
) -> None:
    if not OPEN_COMMAND <= target <= maximum_allowed_command:
        raise ValueError("right gripper target exceeds enabled command range")
    with socket.create_connection(("127.0.0.1", port), timeout=2.0) as connection:
        connection.settimeout(2.0)
        connection.sendall(
            (json.dumps({"op": "command", "target": float(target)}) + "\n").encode()
        )
        with connection.makefile("rb") as stream:
            line = stream.readline(MAX_LINE_BYTES + 1)
    if not line or len(line) > MAX_LINE_BYTES:
        raise RuntimeError("invalid gripper daemon response")
    response = json.loads(line)
    if not response.get("ok"):
        raise RuntimeError(f"gripper daemon rejected command: {response}")


def ping_gripper_command_external(port: int) -> None:
    with socket.create_connection(("127.0.0.1", port), timeout=2.0) as connection:
        connection.settimeout(2.0)
        connection.sendall(b'{"op":"ping"}\n')
        with connection.makefile("rb") as stream:
            line = stream.readline(MAX_LINE_BYTES + 1)
    if not line or len(line) > MAX_LINE_BYTES:
        raise RuntimeError("invalid gripper daemon ping response")
    response = json.loads(line)
    if response.get("ok") is not True:
        raise RuntimeError(f"gripper daemon ping failed: {response}")


@dataclass
class H1Command:
    command_id: str
    target: np.ndarray
    submitted_monotonic: float
    gripper_target: float | None = None
    completed: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: str | None = None


class ControlWorker:
    def __init__(
        self,
        controller: PersistentRightArmController,
        tf: Any,
        robot: Any,
        gripper_enabled: bool = False,
        protected_close_enabled: bool = False,
        gripper_joint_name: str = "right_gripper_joint1",
        gripper_feedback_encoding: str = FEEDBACK_RAW_0_120,
        shutdown_gripper_action: str = "hold",
        gripper_command_port: int = 0,
    ):
        self.controller = controller
        self.tf = tf
        self.robot = robot
        self.commands: queue.Queue[H1Command] = queue.Queue(maxsize=32)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.fatal_error: str | None = None
        self.last_command_id: str | None = None
        self.last_result: dict[str, Any] | None = None
        self.recent_results: deque[dict[str, Any]] = deque(maxlen=32)
        self.seen_command_ids: set[str] = set()
        self.last_live_pose = pose_values(tf, RIGHT_FRAME)
        self.desired_pose = controller.desired.copy()
        self.last_accepted_pose = controller.desired.copy()
        self.translation_compensation = controller.translation_compensation.copy()
        self.rotation_compensation = controller.rotation_compensation.copy()
        self.tick_count = controller.tick_count
        self.started_monotonic = time.monotonic()
        if shutdown_gripper_action not in ("open", "hold"):
            raise ValueError("shutdown_gripper_action must be 'open' or 'hold'")
        self.shutdown_gripper_action = shutdown_gripper_action
        self.gripper_command_port = int(gripper_command_port)
        self.shutdown_requested = threading.Event()
        self.shutdown_deadline_s: float | None = None
        maximum_gripper_command = (
            PROTECTED_MAXIMUM_COMMAND
            if protected_close_enabled
            else PROVISIONAL_MAXIMUM_COMMAND
        )
        if self.gripper_command_port:
            def gripper_sender(target):
                return self._send_gripper_with_gdk_handoff(target, maximum_gripper_command)
        else:
            def gripper_sender(target):
                return send_right_gripper_command(self.robot, target, maximum_gripper_command)
        self.gripper = (
            GripperRuntime(
                lambda: read_right_gripper_observation(
                    self.robot, gripper_joint_name
                ),
                gripper_sender,
                closure_enabled=protected_close_enabled,
                maximum_allowed_command=maximum_gripper_command,
                feedback_encoding=gripper_feedback_encoding,
            )
            if gripper_enabled
            else None
        )
        if self.gripper is not None:
            initial_gripper = read_right_gripper_observation(
                self.robot, gripper_joint_name
            )
            initial_command = feedback_to_command(
                initial_gripper.raw_position, gripper_feedback_encoding
            )
            self.gripper.machine.desired = initial_command
            self.gripper.machine.last_completed = initial_command
            self.gripper.last_observation = initial_gripper
        self.thread = threading.Thread(
            target=self._run, name="g2-groot-right-arm-50hz", daemon=True
        )

    def start(self):
        self.thread.start()

    def submit(self, command: H1Command):
        with self.lock:
            if self.fatal_error:
                raise RuntimeError(f"control worker failed: {self.fatal_error}")
            if command.command_id in self.seen_command_ids:
                raise ValueError(f"duplicate command_id: {command.command_id}")
        self.commands.put_nowait(command)
        with self.lock:
            self.seen_command_ids.add(command.command_id)
            self.last_accepted_pose = command.target.copy()

    def status(self):
        if not self.lock.acquire(timeout=STATUS_LOCK_TIMEOUT_S):
            raise TimeoutError(
                f"control status lock unavailable for {STATUS_LOCK_TIMEOUT_S:.2f}s"
            )
        try:
            live = self.last_live_pose.copy()
            desired = self.desired_pose.copy()
            snapshot = {
                "fatal_error": self.fatal_error,
                "last_accepted_pose": self.last_accepted_pose.copy(),
                "translation_compensation": self.translation_compensation.copy(),
                "rotation_compensation": self.rotation_compensation.copy(),
                "tick_count": self.tick_count,
                "last_command_id": self.last_command_id,
                "last_result": self.last_result,
                "recent_results": list(self.recent_results),
            }
        finally:
            self.lock.release()
        gripper_status = self.gripper.status() if self.gripper else None
        return {
            "schema": ARM_GRIPPER_SCHEMA if self.gripper else ARM_ONLY_SCHEMA,
            "ready": self.controller is not None
            and self.controller.calibrated
            and snapshot["fatal_error"] is None,
            "fatal_error": snapshot["fatal_error"],
            "control_hz": CONTROL_HZ,
            "model_waypoint_hz": 1.0 / MODEL_PERIOD_S,
            "interpolation_ticks": int(round(MODEL_PERIOD_S * CONTROL_HZ)),
            "desired_pose": desired.tolist(),
            "last_accepted_pose": snapshot["last_accepted_pose"].tolist(),
            "live_pose": live.tolist(),
            "live_target_position_error_m": translation_distance(desired, live),
            "live_target_rotation_error_rad": quaternion_angle(desired, live),
            "translation_compensation_m": snapshot[
                "translation_compensation"
            ].tolist(),
            "rotation_compensation_xyzw": snapshot[
                "rotation_compensation"
            ].tolist(),
            "tick_count": snapshot["tick_count"],
            "queue_depth": self.commands.qsize(),
            "last_command_id": snapshot["last_command_id"],
            "last_result": snapshot["last_result"],
            "recent_results": snapshot["recent_results"],
            "session_age_s": time.monotonic() - self.started_monotonic,
            "right_gripper_enabled": self.gripper is not None,
            "right_gripper": gripper_status,
        }

    def _run(self):
        try:
            while not self.stop_event.is_set():
                if self.shutdown_requested.is_set():
                    self.controller.tick(self.controller.desired)
                    if self.gripper and self.controller.tick_count % 5 == 0:
                        self._poll_gripper(block_until_idle=True)
                        if self.gripper.safe_open_complete:
                            self.stop_event.set()
                            continue
                        assert self.shutdown_deadline_s is not None
                        if time.monotonic() >= self.shutdown_deadline_s:
                            raise RuntimeError("right gripper safe-open shutdown timed out")
                    continue
                try:
                    command = self.commands.get_nowait()
                except queue.Empty:
                    self.controller.tick(self.controller.desired)
                    if self.gripper and self.controller.tick_count % 5 == 0:
                        self._poll_gripper(block_until_idle=True)
                    live = pose_values(self.tf, RIGHT_FRAME)
                    with self.lock:
                        self._refresh_cache(live)
                    continue
                try:
                    started = time.monotonic()
                    if self.gripper and command.gripper_target is not None:
                        self._apply_gripper_target(command.gripper_target)
                    command_result = self.controller.move_to(
                        command.target, MODEL_PERIOD_S
                    )
                    if self.gripper:
                        self._poll_gripper()
                    live = pose_values(self.tf, RIGHT_FRAME)
                    gripper_status = self.gripper.status() if self.gripper else None
                    with self.lock:
                        self._refresh_cache(live)
                        command.result = {
                            "command_id": command.command_id,
                            "accepted": True,
                            "queue_delay_s": started - command.submitted_monotonic,
                            "duration_s": time.monotonic() - started,
                            "ticks": command_result["ticks"],
                            "target_pose": command.target.tolist(),
                            "live_pose_at_100ms": live.tolist(),
                            "position_error_at_100ms_m": translation_distance(
                                command.target, live
                            ),
                            "rotation_error_at_100ms_rad": quaternion_angle(
                                command.target, live
                            ),
                            "gripper_target": command.gripper_target,
                            "gripper_status": gripper_status,
                        }
                        self.last_command_id = command.command_id
                        self.last_result = command.result
                        self.recent_results.append(command.result)
                except Exception as error:
                    command.error = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    command.completed.set()
                    self.commands.task_done()
        except Exception as error:
            with self.lock:
                self.fatal_error = f"{type(error).__name__}: {error}"
            emit("control_worker_fatal", message=self.fatal_error)

    def _poll_gripper(self, *, block_until_idle: bool = False):
        assert self.gripper is not None
        decision = self.gripper.poll()
        if block_until_idle and self.gripper.machine.active_target is not None:
            deadline = time.monotonic() + 2.0
            while self.gripper.machine.active_target is not None:
                if time.monotonic() >= deadline:
                    break
                # move_end_effector_joint is a servo command, not a latched
                # trajectory transaction. Refresh the active target at 50 Hz
                # until position feedback confirms completion/contact.
                active_target = self.gripper.machine.active_target
                if active_target is not None and not self.gripper_command_port:
                    self.gripper.send_command(active_target)
                time.sleep(1.0 / CONTROL_HZ)
                decision = self.gripper.poll()
            # Arm refresh intentionally paused while move_ee_pos owns the
            # omnipicker transaction. Resume its 50 Hz schedule from now.
            self.controller.next_tick = time.monotonic()
        if self.gripper.machine.fault:
            raise RuntimeError(
                f"right gripper fault: {self.gripper.machine.fault}"
            )
        return decision

    def _send_gripper_with_gdk_handoff(
        self, target: float, maximum_allowed_command: float
    ) -> None:
        """Temporarily relinquish GDK's Cartesian owner for a tool command."""
        old = self.controller
        desired = old.desired.copy()
        left_session_start = old.left_session_start.copy()
        right_session_start = old.right_session_start.copy()
        tick_count = old.tick_count
        maxima = dict(old.maxima)
        right_session_max_translation_m = old.right_session_max_translation_m
        right_session_max_rotation_rad = old.right_session_max_rotation_rad
        translation_compensation = old.translation_compensation.copy()
        rotation_compensation = old.rotation_compensation.copy()
        workspace_min = old.workspace_min.copy()
        workspace_max = old.workspace_max.copy()
        required_motion_mode = old.required_motion_mode
        emit("gripper_gdk_handoff_begin", target=target)
        # The GDK publishers remain alive while Robot/TF/controller Python
        # objects are referenced. Drop every owner before gdk_release; merely
        # pausing or releasing the global runtime is insufficient on 3.3.8.
        self.controller = None
        self.robot = None
        self.tf = None
        del old
        gc.collect()
        released = agibot_gdk.gdk_release()
        if released != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("gdk_release failed before gripper handoff")
        command_error: Exception | None = None
        try:
            send_right_gripper_command_external(
                self.gripper_command_port, target, maximum_allowed_command
            )
        except Exception as error:
            command_error = error
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("gdk_init failed after gripper handoff")
        self.robot = agibot_gdk.Robot()
        self.tf = agibot_gdk.TF()
        time.sleep(0.5)
        require_safe_state(
            snapshot(self.robot, self.tf), required_motion_mode
        )
        rebuilt = PersistentRightArmController(
            self.robot,
            self.tf,
            right_session_max_translation_m=right_session_max_translation_m,
            right_session_max_rotation_rad=right_session_max_rotation_rad,
            initial_translation_compensation=translation_compensation,
            initial_rotation_compensation=rotation_compensation,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            required_motion_mode=required_motion_mode,
        )
        rebuilt.left_session_start = left_session_start
        rebuilt.right_session_start = right_session_start
        rebuilt.desired = desired
        rebuilt.calibrated = True
        rebuilt.tick_count = tick_count
        rebuilt.maxima = maxima
        rebuilt.next_tick = time.monotonic()
        self.controller = rebuilt
        emit("gripper_gdk_handoff_complete", target=target)
        if command_error is not None:
            raise command_error

    def _apply_gripper_target(self, target: float) -> None:
        """Serialize a tool command after the arm pose stream becomes idle."""
        assert self.gripper is not None
        self.gripper.update_desired(target)
        observation = self.gripper.read_observation()
        feedback = feedback_to_command(
            observation.raw_position, self.gripper.machine.feedback_encoding
        )
        needs_command = bool(
            self.gripper.machine.active_target is not None
            or abs(self.gripper.machine.desired - feedback)
            > self.gripper.machine.command_deadband
        )
        self.gripper.last_observation = observation
        if not needs_command:
            return

        # No controller.tick/move_to call is allowed during this interval.
        # The subsequent blocking poll sends move_ee_pos and confirms motion
        # from physical position feedback before the arm stream resumes.
        time.sleep(GRIPPER_PRE_COMMAND_QUIET_S)
        self._poll_gripper(block_until_idle=True)

    def _refresh_cache(self, live: np.ndarray):
        self.last_live_pose = live.copy()
        self.desired_pose = self.controller.desired.copy()
        self.translation_compensation = (
            self.controller.translation_compensation.copy()
        )
        self.rotation_compensation = self.controller.rotation_compensation.copy()
        self.tick_count = self.controller.tick_count

    def initiate_shutdown(self):
        if not self.thread.is_alive():
            self.stop_event.set()
            return
        if self.gripper and self.shutdown_gripper_action == "open":
            if not self.shutdown_requested.is_set():
                self.gripper.request_safe_open("bridge_shutdown")
                self.shutdown_deadline_s = time.monotonic() + 2.5
                self.shutdown_requested.set()
        else:
            self.stop_event.set()

    def stop(self):
        self.initiate_shutdown()
        # Joining an unstarted thread would hide the original startup failure.
        if self.thread.ident is not None:
            self.thread.join(timeout=3.5)
        if self.thread.is_alive():
            raise RuntimeError("control worker did not stop")


def parse_target(request: dict[str, Any], worker: ControlWorker):
    raw = request.get("target_pose")
    target = np.asarray(raw, dtype=np.float64)
    if target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError("target_pose must be finite XYZ+XYZW")
    target[3:7] = normalize(target[3:7])
    if np.any(target[:3] < worker.controller.workspace_min) or np.any(
        target[:3] > worker.controller.workspace_max
    ):
        raise ValueError("target_pose outside workspace")
    with worker.lock:
        previous_target = worker.last_accepted_pose.copy()
    distance = translation_distance(previous_target, target)
    rotation = quaternion_angle(previous_target, target)
    if distance > MAX_TARGET_STEP_M:
        raise ValueError(
            f"translation step {distance:.6f}m exceeds {MAX_TARGET_STEP_M:.6f}m"
        )
    if rotation > MAX_TARGET_ROTATION_RAD:
        raise ValueError(
            f"rotation step {rotation:.6f}rad exceeds {MAX_TARGET_ROTATION_RAD:.6f}rad"
        )
    timestamp_ns = int(request.get("timestamp_ns", 0))
    if timestamp_ns <= 0:
        raise ValueError("timestamp_ns is required")
    age_s = max(0.0, (time.time_ns() - timestamp_ns) / 1e9)
    if age_s > MAX_COMMAND_AGE_S:
        raise ValueError(
            f"command age {age_s:.3f}s exceeds {MAX_COMMAND_AGE_S:.3f}s"
        )
    return target, age_s


def parse_gripper_target(request: dict[str, Any], worker: ControlWorker) -> float:
    if worker.gripper is None:
        raise ValueError("right gripper is not enabled")
    raw = request.get("target_gripper")
    if isinstance(raw, bool):
        raise ValueError("target_gripper must be a finite scalar")
    try:
        target = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("target_gripper must be a finite scalar") from error
    if not np.isfinite(target):
        raise ValueError("target_gripper must be a finite scalar")
    maximum = worker.gripper.maximum_allowed_command
    tolerance = 1e-5
    if not OPEN_COMMAND - tolerance <= target <= maximum + tolerance:
        raise ValueError(
            f"target_gripper {target} outside enabled range "
            f"[{OPEN_COMMAND}, {maximum}]"
        )
    return max(OPEN_COMMAND, min(maximum, target))


def serve(
    host: str,
    port: int,
    worker: ControlWorker,
    calibration: dict[str, Any],
    session_limit_s: float,
):
    deadline = time.monotonic() + session_limit_s
    schema = ARM_GRIPPER_SCHEMA if worker.gripper else ARM_ONLY_SCHEMA
    operations = ["info", "status", "execute_h1"]
    if worker.gripper:
        operations.append("execute_h1_gripper")
    operations.append("shutdown")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(4)
        listener.settimeout(0.5)
        emit(
            "service_ready",
            bind_host=host,
            port=port,
            schema=schema,
            operations=operations,
            calibration=calibration,
        )
        while time.monotonic() < deadline and not worker.stop_event.is_set():
            try:
                connection, address = listener.accept()
            except socket.timeout:
                if worker.fatal_error:
                    raise RuntimeError(worker.fatal_error)
                continue
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with connection.makefile("rb") as stream:
                    while (
                        time.monotonic() < deadline
                        and not worker.stop_event.is_set()
                    ):
                        operation = ""
                        try:
                            try:
                                remaining_s = deadline - time.monotonic()
                                if remaining_s <= 0.0:
                                    break
                                connection.settimeout(min(60.0, remaining_s))
                                request = recv_line(stream)
                            except (ConnectionError, socket.timeout):
                                # An idle client owns no pending operation. Close
                                # this connection cleanly instead of emitting an
                                # unsolicited error response that desynchronizes
                                # the JSON-line protocol.
                                break
                            if request is None:
                                break
                            operation = str(request.get("op", ""))
                            if operation == "info":
                                response = {
                                    "ok": True,
                                    "server_time_ns": time.time_ns(),
                                    "schema": schema,
                                    "mode": (
                                        "persistent_right_arm_h1_protected_close"
                                        if (
                                            worker.gripper
                                            and worker.gripper.machine.closure_enabled
                                        )
                                        else "persistent_right_arm_h1_tiny_gripper"
                                        if worker.gripper
                                        else "persistent_right_arm_h1_only"
                                    ),
                                    "operations": operations,
                                    "bind_loopback_only": True,
                                    "right_arm_enabled": True,
                                    "right_gripper_enabled": worker.gripper is not None,
                                    "right_gripper_protected_closure_enabled": bool(
                                        worker.gripper
                                        and worker.gripper.machine.closure_enabled
                                    ),
                                    "right_gripper_physical_zero_closure_enabled": bool(
                                        worker.gripper
                                        and worker.gripper.maximum_allowed_command >= 0.0
                                    ),
                                    "right_gripper_command_range": (
                                        [
                                            OPEN_COMMAND,
                                            worker.gripper.maximum_allowed_command,
                                        ]
                                        if worker.gripper
                                        else None
                                    ),
                                    "left_arm_enabled": False,
                                    "head_waist_chassis_enabled": False,
                                    "maximum_horizon": 16,
                                    "model_waypoint_hz": 10.0,
                                    "control_hz": 50.0,
                                    "interpolation_ticks": 5,
                                    "maximum_translation_step_m": MAX_TARGET_STEP_M,
                                    "maximum_rotation_step_rad": MAX_TARGET_ROTATION_RAD,
                                    "maximum_command_age_s": MAX_COMMAND_AGE_S,
                                    "right_session_max_translation_m": (
                                        worker.controller.right_session_max_translation_m
                                    ),
                                    "right_session_max_rotation_rad": (
                                        worker.controller.right_session_max_rotation_rad
                                    ),
                                    "shutdown_gripper_action": (
                                        worker.shutdown_gripper_action
                                    ),
                                    "gripper_command_mode": (
                                        "external_daemon"
                                        if worker.gripper_command_port
                                        else "in_process_servo"
                                    ),
                                    "acknowledgement": "immediate_queue_acceptance",
                                    "execution_results": "status.recent_results",
                                }
                            elif operation == "status":
                                status_started = time.monotonic()
                                response = {"ok": True, **worker.status()}
                                status_elapsed_s = time.monotonic() - status_started
                                if status_elapsed_s >= 0.05:
                                    emit(
                                        "slow_status_response",
                                        elapsed_s=status_elapsed_s,
                                    )
                            elif operation in ("execute_h1", "execute_h1_gripper"):
                                command_id = str(request.get("command_id", "")).strip()
                                if not command_id or len(command_id) > 128:
                                    raise ValueError(
                                        "command_id is required and must be <=128 chars"
                                    )
                                target, age_s = parse_target(request, worker)
                                gripper_target = (
                                    parse_gripper_target(request, worker)
                                    if operation == "execute_h1_gripper"
                                    else None
                                )
                                command = H1Command(
                                    command_id=command_id,
                                    target=target,
                                    submitted_monotonic=time.monotonic(),
                                    gripper_target=gripper_target,
                                )
                                worker.submit(command)
                                response = {
                                    "ok": True,
                                    "command_age_s": age_s,
                                    "command_id": command.command_id,
                                    "accepted": True,
                                    "queued": True,
                                    "gripper_target": command.gripper_target,
                                }
                            elif operation == "shutdown":
                                response = {"ok": True, "shutdown": True}
                                worker.initiate_shutdown()
                            else:
                                raise ValueError("unsupported operation")
                        except Exception as error:
                            response = {
                                "ok": False,
                                "error_type": type(error).__name__,
                                "message": str(error),
                            }
                        try:
                            send_response(connection, response)
                        except (ConnectionError, socket.timeout):
                            break
                        if operation == "shutdown":
                            break
    emit("service_stopped", reason="shutdown_or_session_limit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-control", action="store_true")
    parser.add_argument("--enable-gripper", action="store_true")
    parser.add_argument("--enable-approach-session", action="store_true")
    parser.add_argument("--enable-protected-close", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--session-limit-s", type=float, default=DEFAULT_SESSION_LIMIT_S)
    parser.add_argument("--startup-countdown-s", type=int, default=5)
    parser.add_argument("--calibration-duration-s", type=float, default=2.0)
    parser.add_argument(
        "--right-session-max-translation-m",
        type=float,
        default=None,
        help="session displacement envelope; defaults depend on approach mode",
    )
    parser.add_argument(
        "--right-session-max-rotation-deg",
        type=float,
        default=None,
        help="session rotation envelope in degrees; defaults depend on approach mode",
    )
    parser.add_argument("--required-motion-mode", type=int, default=5)
    parser.add_argument("--gripper-joint-name", default="right_gripper_joint1")
    parser.add_argument(
        "--gripper-feedback-encoding",
        choices=FEEDBACK_ENCODINGS,
        default=FEEDBACK_RAW_0_120,
    )
    parser.add_argument(
        "--shutdown-gripper-action",
        choices=("open", "hold"),
        default="hold",
        help="whether bridge shutdown opens the gripper or leaves its last command",
    )
    parser.add_argument(
        "--gripper-command-port",
        type=int,
        default=0,
        help="loopback port of a separate GDK gripper owner; 0 uses this process",
    )
    parser.add_argument(
        "--workspace-min",
        type=float,
        nargs=3,
        default=WORKSPACE_MIN.tolist(),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--workspace-max",
        type=float,
        nargs=3,
        default=WORKSPACE_MAX.tolist(),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--initial-translation-compensation",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--initial-rotation-compensation",
        type=float,
        nargs=4,
        metavar=("X", "Y", "Z", "W"),
    )
    args = parser.parse_args()
    require_loopback(args.bind_host)
    if not 1 <= args.port <= 65535:
        parser.error("invalid port")
    if not 30.0 <= args.session_limit_s <= 1800.0:
        parser.error("--session-limit-s must be in [30, 1800]")
    if not 0 <= args.startup_countdown_s <= 5:
        parser.error("--startup-countdown-s must be in [0, 5]")
    if not 0.2 <= args.calibration_duration_s <= 2.0:
        parser.error("--calibration-duration-s must be in [0.2, 2.0]")
    if any(a >= b for a, b in zip(args.workspace_min, args.workspace_max)):
        parser.error("--workspace-min must be below --workspace-max")
    default_session_translation_m = (
        APPROACH_SESSION_MAX_TRANSLATION_M
        if args.enable_approach_session
        else RIGHT_SESSION_MAX_TRANSLATION_M
    )
    default_session_rotation_rad = (
        APPROACH_SESSION_MAX_ROTATION_RAD
        if args.enable_approach_session
        else RIGHT_SESSION_MAX_ROTATION_RAD
    )
    session_translation_m = (
        default_session_translation_m
        if args.right_session_max_translation_m is None
        else args.right_session_max_translation_m
    )
    session_rotation_rad = (
        default_session_rotation_rad
        if args.right_session_max_rotation_deg is None
        else np.deg2rad(args.right_session_max_rotation_deg)
    )
    if not 0 < session_translation_m <= MAX_ALLOWED_SESSION_TRANSLATION_M:
        parser.error(
            "--right-session-max-translation-m must be in "
            f"(0, {MAX_ALLOWED_SESSION_TRANSLATION_M}]"
        )
    if not 0 < session_rotation_rad <= MAX_ALLOWED_SESSION_ROTATION_RAD:
        parser.error("--right-session-max-rotation-deg must be in (0, 180]")
    if args.enable_gripper and not args.enable_control:
        parser.error("--enable-gripper requires --enable-control")
    if args.gripper_command_port and not args.enable_gripper:
        parser.error("--gripper-command-port requires --enable-gripper")
    if not 0 <= args.gripper_command_port <= 65535:
        parser.error("invalid --gripper-command-port")
    if args.enable_approach_session and not args.enable_gripper:
        parser.error("--enable-approach-session requires --enable-gripper")
    if args.enable_protected_close and not args.enable_approach_session:
        parser.error("--enable-protected-close requires --enable-approach-session")
    if args.gripper_command_port:
        ping_gripper_command_external(args.gripper_command_port)
    custom_seed = (
        args.initial_translation_compensation is not None
        or args.initial_rotation_compensation is not None
    )
    if custom_seed and (
        args.initial_translation_compensation is None
        or args.initial_rotation_compensation is None
    ):
        parser.error("both initial compensation arguments are required together")
    if custom_seed and not args.enable_control:
        parser.error("custom initial compensation requires --enable-control")
    required_confirmation = (
        PROTECTED_CLOSE_CONFIRMATION
        if args.enable_protected_close
        else APPROACH_CONFIRMATION
        if args.enable_approach_session
        else GRIPPER_CONFIRMATION
        if args.enable_gripper
        else CONFIRMATION
    )
    if args.enable_control and args.confirm != required_confirmation:
        parser.error(
            f"selected control mode requires --confirm {required_confirmation}"
        )
    if not args.enable_control and args.confirm:
        parser.error("--confirm is only valid with --enable-control")
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise RuntimeError("gdk_init failed")
    worker = None
    try:
        robot, tf = agibot_gdk.Robot(), agibot_gdk.TF()
        time.sleep(2.0)
        state = snapshot(robot, tf)
        require_safe_state(state, args.required_motion_mode)
        gripper_preflight = None
        if args.enable_gripper:
            gripper_observation = read_right_gripper_observation(
                robot, args.gripper_joint_name
            )
            if gripper_observation.motor_status not in (0, 2, 3):
                raise RuntimeError("right gripper is not settled at startup")
            feedback_to_command(
                gripper_observation.raw_position,
                args.gripper_feedback_encoding,
            )
            gripper_preflight = {
                "raw_position": gripper_observation.raw_position,
                "motor_status": gripper_observation.motor_status,
                "motor_err_code": gripper_observation.motor_err_code,
                "whole_end_error": gripper_observation.whole_end_error,
                "joint_name": args.gripper_joint_name,
                "feedback_encoding": args.gripper_feedback_encoding,
            }
        competitors = competing_controllers()
        if competitors:
            raise RuntimeError(f"competing controllers found: {competitors}")
        emit(
            "physical_preflight",
            execute=args.enable_control,
            mode=state["mode"],
            whole=state["whole"],
            start_pose=state["right"].round(9).tolist(),
            right_gripper_enabled=args.enable_gripper,
            right_gripper_protected_closure_enabled=args.enable_protected_close,
            right_gripper_physical_zero_closure_enabled=False,
            right_gripper_command_range=(
                [
                    OPEN_COMMAND,
                    (
                        PROTECTED_MAXIMUM_COMMAND
                        if args.enable_protected_close
                        else PROVISIONAL_MAXIMUM_COMMAND
                    ),
                ]
                if args.enable_gripper
                else None
            ),
            right_gripper_preflight=gripper_preflight,
            competing_controllers=competitors,
            session_limit_s=args.session_limit_s,
            approach_session_enabled=args.enable_approach_session,
            right_session_max_translation_m=session_translation_m,
            right_session_max_rotation_rad=session_rotation_rad,
            shutdown_gripper_action=args.shutdown_gripper_action,
            gripper_command_port=args.gripper_command_port,
            initial_translation_compensation=args.initial_translation_compensation,
            initial_rotation_compensation=args.initial_rotation_compensation,
            workspace_min=args.workspace_min,
            workspace_max=args.workspace_max,
            required_motion_mode=args.required_motion_mode,
        )
        if not args.enable_control:
            return 0
        for remaining in range(args.startup_countdown_s, 0, -1):
            emit("physical_countdown", remaining_s=remaining)
            time.sleep(1.0)
        controller = PersistentRightArmController(
            robot,
            tf,
            right_session_max_translation_m=session_translation_m,
            right_session_max_rotation_rad=session_rotation_rad,
            initial_translation_compensation=args.initial_translation_compensation,
            initial_rotation_compensation=args.initial_rotation_compensation,
            workspace_min=args.workspace_min,
            workspace_max=args.workspace_max,
            required_motion_mode=args.required_motion_mode,
        )
        calibration = controller.calibrate(args.calibration_duration_s)
        worker = ControlWorker(
            controller,
            tf,
            robot,
            gripper_enabled=args.enable_gripper,
            protected_close_enabled=args.enable_protected_close,
            gripper_joint_name=args.gripper_joint_name,
            gripper_feedback_encoding=args.gripper_feedback_encoding,
            shutdown_gripper_action=args.shutdown_gripper_action,
            gripper_command_port=args.gripper_command_port,
        )
        worker.start()
        serve(args.bind_host, args.port, worker, calibration, args.session_limit_s)
        worker.stop()
        final_worker_status = worker.status()
        worker = None
        final = snapshot(robot, tf)
        require_safe_state(final, args.required_motion_mode)
        emit(
            "final_state",
            mode=final["mode"],
            whole=final["whole"],
            final_pose=final["right"].tolist(),
            right_gripper=final_worker_status.get("right_gripper"),
        )
        return 0
    finally:
        # Cleanup must not turn a controller/GDK failure into a thread error.
        original_error = sys.exc_info()[1]
        try:
            if worker is not None:
                try:
                    worker.stop()
                except Exception as cleanup_error:
                    if original_error is None:
                        raise
                    emit("worker_cleanup_failed", message=str(cleanup_error))
        finally:
            agibot_gdk.gdk_release()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        emit("fatal", error_type=type(error).__name__, message=str(error))
        raise
