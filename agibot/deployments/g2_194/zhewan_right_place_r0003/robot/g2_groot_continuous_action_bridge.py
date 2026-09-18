#!/usr/bin/env python3
"""Candidate single-owner, combined GDK arm/tool service; not hardware-qualified.

The listener is read-only in standby. Physical publishing starts only after
both --enable-control and the runner's explicit activate request. No child
process, tool-motion API, or controller restart is used during a policy run.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
import math
import queue
import socket
import threading
import time
from typing import Any

from g2_groot_contact_guard import ContactGuard
from g2_groot_gripper_state_machine import FEEDBACK_NATIVE_RADIANS, feedback_to_command
from g2_groot_native_collision_latch import NativeCollisionLatch
from g2_groot_persistent_h1_action_bridge import (
    ARM_GRIPPER_SCHEMA,
    RIGHT_FRAME,
    H1Command,
    parse_target,
    pose_values,
    quaternion_angle,
    read_right_gripper_observation,
    recv_line,
    require_loopback,
    send_response,
)
import numpy as np


ACTIVATION_PROTOCOL = "explicit_activate_v1"
ACTIVATION_CONFIRMATION = "ACTIVATE_G2_GROOT_PLACE_ARM"
OPEN_THRESHOLD = -0.72
MAX_FEEDBACK_AGE_S = 0.5
COMPACT_RESULT_FIELDS = (
    "command_id",
    "accepted",
    "ok",
    "error",
    "ticks",
    "execution_started_monotonic_ns",
    "execution_finished_monotonic_ns",
    "gripper_release_event",
    "live_pose_at_100ms",
    "gripper_target",
)


class StopPublishing(RuntimeError):
    """The worker was stopped before its next physical publication."""


def gripper_value(raw: Any) -> float:
    if isinstance(raw, bool):
        raise ValueError("target_gripper must be a finite scalar in [-0.785, 0]")
    value = float(raw)
    if not math.isfinite(value) or not -0.7851 <= value <= 0.0001:
        raise ValueError("target_gripper must be a finite scalar in [-0.785, 0]")
    return min(0.0, max(-0.785, value))


class ContinuousActionService:
    def __init__(
        self,
        robot: Any,
        tf: Any,
        *,
        initial_gripper_command: float,
        workspace_min: Any,
        workspace_max: Any,
        enable_control: bool = False,
        required_motion_mode: int = 1,
        calibration_duration_s: float = 2.0,
        gripper_joint_name: str = "right_gripper_joint1",
        controller_factory: Any = None,
        sender_factory: Any = None,
        contact_guard: ContactGuard | None = None,
        native_collision_latch: NativeCollisionLatch | None = None,
        freeze_compensation_after_calibration: bool = False,
    ):
        self.robot, self.tf = robot, tf
        self.contact_guard = contact_guard
        self.native_collision_latch = native_collision_latch
        self.freeze_compensation_after_calibration = freeze_compensation_after_calibration
        self.enable_control = enable_control
        self.initial_gripper_command = gripper_value(initial_gripper_command)
        self.requested_gripper = self.initial_gripper_command
        self.workspace_min = np.asarray(workspace_min, dtype=np.float64)
        self.workspace_max = np.asarray(workspace_max, dtype=np.float64)
        if (
            self.workspace_min.shape != (3,)
            or self.workspace_max.shape != (3,)
            or not np.isfinite([self.workspace_min, self.workspace_max]).all()
            or np.any(self.workspace_min >= self.workspace_max)
        ):
            raise ValueError("workspace bounds must be finite ordered XYZ vectors")
        if not 0.2 <= calibration_duration_s <= 2.0:
            raise ValueError("calibration duration must be in [0.2, 2.0] seconds")
        self.required_motion_mode = required_motion_mode
        self.calibration_duration_s = calibration_duration_s
        self.gripper_joint_name = gripper_joint_name
        self.controller_factory, self.sender_factory = controller_factory, sender_factory
        self.lock = threading.RLock()
        self.stop_event, self.activation_done = threading.Event(), threading.Event()
        self.activation_state = "standby"
        self.owner = None
        self.fatal_error = None
        self.controller = self.sender = self.thread = None
        self.commands: queue.Queue[H1Command] = queue.Queue(maxsize=32)
        self.seen_command_ids: set[str] = set()
        self.recent_results: deque[dict] = deque(maxlen=32)
        self.active_command_id = self.last_command_id = self.gripper_command_id = None
        self.active_command_started_monotonic_ns = None
        self.last_result = self.release_event = self.last_observation = None
        self.last_live_pose = np.asarray(pose_values(tf, RIGHT_FRAME), dtype=np.float64)
        self.desired_pose = self.last_accepted_pose = self.last_live_pose.copy()
        self.last_pose_read_s = time.monotonic()
        self.calibration = None
        self._poll_feedback()

    def _poll_feedback(self):
        observation = read_right_gripper_observation(self.robot, self.gripper_joint_name)
        if (
            not all(
                math.isfinite(x)
                for x in (observation.raw_position, observation.effort, observation.monotonic_s)
            )
            or observation.motor_err_code
            or observation.whole_end_error
        ):
            raise RuntimeError("invalid or faulted physical right gripper feedback")
        feedback_to_command(observation.raw_position, FEEDBACK_NATIVE_RADIANS)
        live = np.asarray(pose_values(self.tf, RIGHT_FRAME), dtype=np.float64)
        if live.shape != (7,) or not np.isfinite(live).all():
            raise RuntimeError("invalid physical right EEF feedback")
        with self.lock:
            previous = self.last_observation
            self.last_observation = observation
            self.last_live_pose = live.copy()
            self.last_pose_read_s = time.monotonic()
            if (
                self.release_event is None
                and previous is not None
                and previous.raw_position > OPEN_THRESHOLD
                and observation.raw_position <= OPEN_THRESHOLD
                and self.requested_gripper <= OPEN_THRESHOLD
                and self.gripper_command_id
            ):
                self.release_event = {
                    "command_id": self.gripper_command_id,
                    "feedback_position": observation.raw_position,
                    "pose": live.tolist(),
                    "monotonic_s": observation.monotonic_s,
                    "time_basis": "local_feedback_read_not_sensor_timestamp",
                }
            if self.controller is not None:
                self.desired_pose = self.controller.desired.copy()

    def before_publish(self):
        if self.stop_event.is_set():
            raise StopPublishing("publishing stopped")
        self._poll_feedback()
        if self.native_collision_latch is not None:
            self.native_collision_latch.check_robot(self.robot)
        if self.contact_guard is not None:
            self.contact_guard.check_feedback(
                self.robot.get_motion_control_status(), self.last_live_pose
            )
        if self.stop_event.is_set():
            raise StopPublishing("publishing stopped")

    def activation_fields(self):
        with self.lock:
            return {
                "activation_protocol": ACTIVATION_PROTOCOL,
                "activation_state": self.activation_state,
                "arm_owner_active": bool(self.thread and self.thread.is_alive()),
            }

    def info(self):
        return {
            "ok": True,
            "server_time_ns": time.time_ns(),
            "schema": ARM_GRIPPER_SCHEMA,
            **self.activation_fields(),
            "mode": "continuous_right_arm_and_gripper_candidate",
            "operations": [
                "info", "status", "activate", "execute_h1_gripper", "execute_h16_gripper", "shutdown"
            ],
            "chunk_submission": "atomic_h16_native_v1",
            "right_arm_enabled": True,
            "right_gripper_enabled": True,
            "right_gripper_protected_closure_enabled": True,
            "right_gripper_physical_zero_closure_enabled": True,
            "right_gripper_command_range": [-0.785, 0.0],
            "left_arm_enabled": False,
            "head_waist_chassis_enabled": False,
            "maximum_horizon": 16,
            "model_waypoint_hz": 10.0,
            "control_hz": 50.0,
            "interpolation_ticks": 5,
            "shutdown_gripper_action": "hold",
            "gripper_command_mode": "native_trajectory_tracking",
            "gripper_policy_mode": "per_row_absolute",
            "gripper_partial_targets": "executed_each_waypoint",
            "acknowledgement": "immediate_queue_acceptance",
            "execution_results": "status.recent_results",
            "hardware_validation": "pending",
            "enable_control": self.enable_control,
            "native_collision_latch": None if self.native_collision_latch is None else self.native_collision_latch.snapshot(),
            "compensation_mode": "calibration_only" if self.freeze_compensation_after_calibration else "online_adaptive",
            "contact_guard": None if self.contact_guard is None else self.contact_guard.snapshot(),
        }

    def status(self, *, compact: bool = False, after_command_id: str | None = None):
        with self.lock:
            now = time.monotonic()
            observation = self.last_observation
            pose_age = max(0.0, now - self.last_pose_read_s)
            feedback_age = max(0.0, now - observation.monotonic_s)
            stale = pose_age > MAX_FEEDBACK_AGE_S or feedback_age > MAX_FEEDBACK_AGE_S
            actual = float(observation.raw_position)
            result = {
                "ok": True,
                "schema": ARM_GRIPPER_SCHEMA,
                **self.activation_fields(),
                "ready": self.activation_state == "active" and not self.fatal_error and not stale,
                "fatal_error": self.fatal_error,
                "native_collision_latch": None if self.native_collision_latch is None else self.native_collision_latch.snapshot(),
                "contact_guard": None if self.contact_guard is None else self.contact_guard.snapshot(),
                "live_pose": self.last_live_pose.tolist(),
                "desired_pose": self.desired_pose.tolist(),
                "last_accepted_pose": self.last_accepted_pose.tolist(),
                "live_target_position_error_m": float(
                    np.linalg.norm(self.desired_pose[:3] - self.last_live_pose[:3])
                ),
                "live_target_rotation_error_rad": float(
                    quaternion_angle(self.desired_pose, self.last_live_pose)
                ),
                "live_pose_age_s": pose_age,
                "live_pose_stale": pose_age > MAX_FEEDBACK_AGE_S,
                "gripper_feedback_age_s": feedback_age,
                "gripper_feedback_stale": feedback_age > MAX_FEEDBACK_AGE_S,
                "feedback_age_basis": "local_read_cache_not_sensor_timestamp",
                "queue_depth": self.commands.qsize() + int(self.active_command_id is not None),
                "active_command_id": self.active_command_id,
                "active_command_started_monotonic_ns": self.active_command_started_monotonic_ns,
                "last_command_id": self.last_command_id,
                "last_result": self.last_result,
                "recent_results": list(self.recent_results),
                "calibration": self.calibration,
                "gripper_release_event": self.release_event,
                "right_gripper": {
                    "feedback_encoding": FEEDBACK_NATIVE_RADIANS,
                    "last_observation": asdict(observation),
                    "desired": self.requested_gripper,
                    "last_completed": actual,
                    "last_completed_semantics": "latest_physical_feedback",
                    "target_reached": abs(self.requested_gripper - actual) <= 0.01,
                    "fault": self.fatal_error,
                    "recovery_requested": False,
                },
            }
            if compact:
                # Keep the default diagnostic schema untouched. Incremental
                # polling omits only redundant/heavy diagnostic payloads.
                result.pop("calibration")
                result.pop("last_result")
                recent = result["recent_results"]
                if after_command_id is not None:
                    for index, receipt in enumerate(recent):
                        if receipt.get("command_id") == after_command_id:
                            recent = recent[index + 1 :]
                            break
                result["recent_results"] = [
                    {key: receipt[key] for key in COMPACT_RESULT_FIELDS if key in receipt}
                    for receipt in recent
                ]
        return result

    def activate(self, payload: dict, owner: object):
        if payload.get("confirm") != ACTIVATION_CONFIRMATION:
            raise ValueError("explicit activation confirmation is required")
        with self.lock:
            if not self.enable_control:
                raise RuntimeError("read-only service: --enable-control was not supplied")
            if self.activation_state == "active" and self.owner is owner:
                return {"ok": True, **self.activation_fields()}
            if self.activation_state != "standby":
                raise RuntimeError(f"cannot activate from {self.activation_state}")
            if self.native_collision_latch is not None:
                self.native_collision_latch.arm(self.robot)
            if self.contact_guard is not None:
                self.contact_guard.ensure_ready()
                self._poll_feedback()
                self.contact_guard.check_feedback(
                    self.robot.get_motion_control_status(), self.last_live_pose
                )
            self.owner = owner
            self.activation_state = "activating"
            self.thread = threading.Thread(target=self._run, name="g2-continuous-50hz", daemon=True)
            self.thread.start()
        if not self.activation_done.wait(50.0):
            self.fail("activation timed out")
            self.stop()
        with self.lock:
            if self.activation_state != "active":
                raise RuntimeError(self.fatal_error or "activation stopped")
            return {"ok": True, **self.activation_fields()}

    def execute(self, payload: dict, owner: object):
        with self.lock:
            if self.activation_state != "active" or self.owner is not owner:
                raise RuntimeError("active owner connection is required")
            try:
                command_id = payload.get("command_id")
                if (
                    not isinstance(command_id, str)
                    or not command_id.strip()
                    or len(command_id) > 128
                ):
                    raise ValueError("command_id must be a nonempty string of at most 128 chars")
                if command_id in self.seen_command_ids:
                    raise ValueError("duplicate command_id")
                target, age_s = parse_target(payload, self)
                if self.contact_guard is not None:
                    self.contact_guard.check_pose(target, source="model_target")
                target_gripper = gripper_value(payload.get("target_gripper"))
                command = H1Command(command_id, target, time.monotonic(), target_gripper)
                self.commands.put_nowait(command)
                self.seen_command_ids.add(command_id)
                self.last_accepted_pose = target.copy()
            except Exception as error:
                self.fail(f"command rejected: {type(error).__name__}: {error}")
                raise
        return {
            "ok": True,
            "accepted": True,
            "queued": True,
            "command_id": command_id,
            "command_age_s": age_s,
            "gripper_target": target_gripper,
        }

    def execute_chunk(self, payload: dict, owner: object):
        """Validate all H16 rows before enqueueing any; the local worker owns pacing.

        This only changes transport, not targets or controller interpolation.
        Never append another chunk while one is active or retry an ambiguous ACK.
        """
        with self.lock:
            if self.activation_state != "active" or self.owner is not owner:
                raise RuntimeError("active owner connection is required")
            try:
                if self.active_command_id is not None or not self.commands.empty():
                    raise RuntimeError("H16 submission requires an idle command queue")
                raw_rows = payload.get("commands")
                if not isinstance(raw_rows, list) or len(raw_rows) != 16:
                    raise ValueError("expected exactly 16 commands")
                commands, command_ids = [], set()
                previous = self.last_accepted_pose.copy()
                for row in raw_rows:
                    command_id = row.get("command_id")
                    if (
                        not isinstance(command_id, str)
                        or not command_id.strip()
                        or len(command_id) > 128
                    ):
                        raise ValueError("command_id must be a nonempty string of at most 128 chars")
                    if command_id in self.seen_command_ids or command_id in command_ids:
                        raise ValueError("duplicate command_id")
                    target, _ = parse_target(row, self, previous_target=previous)
                    if self.contact_guard is not None:
                        self.contact_guard.check_pose(target, source="model_target")
                    gripper = gripper_value(row.get("target_gripper"))
                    commands.append(H1Command(command_id, target, time.monotonic(), gripper))
                    command_ids.add(command_id)
                    previous = target
                # The consumer takes this same lock when dequeuing. No row can
                # start before validation and the complete enqueue both finish.
                for command in commands:
                    self.commands.put_nowait(command)
                self.seen_command_ids.update(command_ids)
                self.last_accepted_pose = previous.copy()
            except Exception as error:
                self.fail(f"chunk rejected: {type(error).__name__}: {error}")
                raise
        return {
            "ok": True,
            "accepted": True,
            "queued": True,
            "command_ids": [command.command_id for command in commands],
            "chunk_submission": "atomic_h16_native_v1",
        }

    def fail(self, message: str):
        with self.lock:
            self.fatal_error = self.fatal_error or message
            self.activation_state = "fault"
            self.stop_event.set()

    def _record(self, command: H1Command, result: dict):
        with self.lock:
            started_ns = (
                self.active_command_started_monotonic_ns
                if self.active_command_id == command.command_id
                else None
            )
            result.setdefault("execution_started_monotonic_ns", started_ns)
            result.setdefault(
                "execution_finished_monotonic_ns",
                time.monotonic_ns() if started_ns is not None else None,
            )
            command.result = result
            self.last_command_id, self.last_result = command.command_id, result
            self.recent_results.append(result)
            self.active_command_id = None
            self.active_command_started_monotonic_ns = None
            command.completed.set()

    def _run(self):
        command = None
        try:
            if self.sender_factory is None:
                from g2_groot_continuous_sender import NativeTrajectorySender

                self.sender_factory = NativeTrajectorySender
            if self.controller_factory is None:
                from g2_groot_continuous_controller import ContinuousRightArmController

                self.controller_factory = ContinuousRightArmController
            self.sender = self.sender_factory(
                self.initial_gripper_command,
                before_publish=self.before_publish,
                **({"pose_guard": self.contact_guard.check_pose} if self.contact_guard else {}),
            )
            self.sender.initialize_references(self.robot, self.tf)
            self.controller = self.controller_factory(
                self.robot,
                self.tf,
                sender=self.sender,
                workspace_min=self.workspace_min,
                workspace_max=self.workspace_max,
                required_motion_mode=self.required_motion_mode,
                **({"freeze_compensation_after_calibration": True} if self.freeze_compensation_after_calibration else {}),
            )
            self.calibration = self.controller.calibrate(self.calibration_duration_s)
            self._poll_feedback()
            with self.lock:
                if self.stop_event.is_set():
                    raise StopPublishing("activation stopped")
                self.desired_pose = self.last_accepted_pose = self.controller.desired.copy()
                self.activation_state = "active"
            self.activation_done.set()
            while not self.stop_event.is_set():
                with self.lock:
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        command = None
                    if command is not None:
                        self.active_command_id = command.command_id
                        self.active_command_started_monotonic_ns = time.monotonic_ns()
                        self.gripper_command_id = command.command_id
                        self.requested_gripper = command.gripper_target
                if command is None:
                    self.controller.tick(self.controller.desired)
                    self._poll_feedback()
                    continue
                started = time.monotonic()
                self.sender.set_target(command.gripper_target, ticks=5)
                outcome = self.controller.move_to(command.target, 0.1)
                execution_finished_ns = time.monotonic_ns()
                if outcome["ticks"] != 5:
                    raise RuntimeError("a policy row must publish exactly five control ticks")
                self._poll_feedback()
                native_snapshot = self.sender.snapshot()
                with self.lock:
                    result = {
                        "command_id": command.command_id,
                        "accepted": True,
                        "ok": True,
                        "queue_delay_s": started - command.submitted_monotonic,
                        "duration_s": time.monotonic() - started,
                        "execution_started_monotonic_ns": self.active_command_started_monotonic_ns,
                        "execution_finished_monotonic_ns": execution_finished_ns,
                        "ticks": outcome["ticks"],
                        "target_pose": command.target.tolist(),
                        "gripper_target": command.gripper_target,
                        "live_pose_at_100ms": self.last_live_pose.tolist(),
                        "gripper_release_event": self.release_event,
                        "gripper_execution": "native_simultaneous",
                        "execution_evidence": "gdk_published_with_feedback_not_physical_settlement",
                        "gripper_status": self.status()["right_gripper"],
                        "native_trajectory": native_snapshot,
                    }
                self._record(command, result)
                self.commands.task_done()
                command = None
        except Exception as error:
            if not (isinstance(error, StopPublishing) and self.stop_event.is_set()):
                self.fail(f"{type(error).__name__}: {error}")
        finally:
            reason = self.fatal_error or "publishing stopped before command completion"
            if command is not None:
                self._record(
                    command,
                    {
                        "command_id": command.command_id,
                        "ok": False,
                        "accepted": False,
                        "error": reason,
                    },
                )
                self.commands.task_done()
            while True:
                try:
                    pending = self.commands.get_nowait()
                except queue.Empty:
                    break
                self._record(
                    pending,
                    {
                        "command_id": pending.command_id,
                        "ok": False,
                        "accepted": False,
                        "error": reason,
                    },
                )
                self.commands.task_done()
            self.activation_done.set()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                self.fail("control worker did not stop")
                raise RuntimeError(self.fatal_error)

    def disconnect(self, owner: object, *, shutdown: bool = False):
        with self.lock:
            if self.owner is not owner:
                return
            if self.activation_state != "fault":
                self.activation_state = "stopped" if shutdown else "fault"
                if not shutdown:
                    self.fatal_error = "activated client disconnected; restart service to retry"
            self.stop_event.set()
        self.stop()


def serve_connection(service: ContinuousActionService, connection: socket.socket, deadline: float):
    owner, shutdown = object(), False
    try:
        with connection, connection.makefile("rb") as stream:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while time.monotonic() < deadline:
                connection.settimeout(min(60.0, max(0.01, deadline - time.monotonic())))
                try:
                    payload = recv_line(stream)
                    if payload is None:
                        break
                    operation = payload.get("op")
                    if operation == "info":
                        response = service.info()
                    elif operation == "status":
                        response = service.status(
                            compact=payload.get("compact") is True,
                            after_command_id=payload.get("after_command_id"),
                        )
                    elif operation == "activate":
                        response = service.activate(payload, owner)
                    elif operation == "execute_h1_gripper":
                        response = service.execute(payload, owner)
                    elif operation == "execute_h16_gripper":
                        response = service.execute_chunk(payload, owner)
                    elif operation == "shutdown":
                        if service.owner is not None and service.owner is not owner:
                            raise RuntimeError("shutdown requires the active owner")
                        service.disconnect(owner, shutdown=True)
                        shutdown = True
                        response = {"ok": True, "shutdown": True}
                    else:
                        raise ValueError("unsupported operation")
                except (OSError, EOFError):
                    break
                except Exception as error:
                    response = {
                        "ok": False,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                try:
                    send_response(connection, response)
                except OSError:
                    break
                if shutdown:
                    break
    finally:
        service.disconnect(owner, shutdown=shutdown)
    return shutdown


def serve(service: ContinuousActionService, host: str, port: int, session_limit_s: float):
    require_loopback(host)
    deadline = time.monotonic() + session_limit_s
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(4)
        listener.settimeout(0.5)
        print(json.dumps({"event": "standby_listener_ready", **service.info()}), flush=True)
        try:
            while time.monotonic() < deadline and not service.stop_event.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                if serve_connection(service, connection, deadline):
                    break
        finally:
            service.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-control", action="store_true")
    parser.add_argument("--contact-guard-config", help="site-confirmed right-EEF limits JSON")
    parser.add_argument("--initial-gripper-command", type=float, required=True)
    parser.add_argument("--workspace-min", type=float, nargs=3, required=True)
    parser.add_argument("--workspace-max", type=float, nargs=3, required=True)
    parser.add_argument("--required-motion-mode", type=int, default=1)
    parser.add_argument("--required-control-mode", type=int, choices=(2, 3), default=3)
    parser.add_argument("--freeze-compensation-after-calibration", action="store_true")
    parser.add_argument("--calibration-duration-s", type=float, default=2.0)
    parser.add_argument("--gripper-joint-name", default="right_gripper_joint1")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--session-limit-s", type=float, default=1800.0)
    args = parser.parse_args()
    require_loopback(args.bind_host)
    if not 1 <= args.port <= 65535 or not 0 < args.session_limit_s <= 1800:
        parser.error("invalid port or session duration")
    contact_guard = ContactGuard.from_file(args.contact_guard_config) if args.contact_guard_config else None
    if args.enable_control and contact_guard is not None:
        contact_guard.ensure_ready()
    import agibot_gdk

    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise RuntimeError("gdk_init failed")
    service = None
    try:
        robot, tf = agibot_gdk.Robot(), agibot_gdk.TF()
        time.sleep(2.0)
        service = ContinuousActionService(
            robot,
            tf,
            initial_gripper_command=args.initial_gripper_command,
            workspace_min=args.workspace_min,
            workspace_max=args.workspace_max,
            enable_control=args.enable_control,
            required_motion_mode=args.required_motion_mode,
            calibration_duration_s=args.calibration_duration_s,
            gripper_joint_name=args.gripper_joint_name,
            contact_guard=contact_guard,
            native_collision_latch=NativeCollisionLatch(args.required_control_mode),
            freeze_compensation_after_calibration=args.freeze_compensation_after_calibration,
        )
        serve(service, args.bind_host, args.port, args.session_limit_s)
    finally:
        if service is not None:
            service.stop()
        agibot_gdk.gdk_release()


if __name__ == "__main__":
    main()
