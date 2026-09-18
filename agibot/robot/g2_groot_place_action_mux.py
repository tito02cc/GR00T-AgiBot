#!/usr/bin/env python3
"""Stable action endpoint that serializes G2 Cartesian-arm and tool owners.

The tested separate-API path on this GDK 3.3.8 robot did not execute
``move_ee_pos`` while a process publishing ``end_effector_pose_control`` was
alive. This does not rule out GDK's native combined trajectory API.
This supervisor owns no GDK objects.
It starts in nonmoving standby with only the feedback/tool daemon. An explicit
activation request starts the arm child; inspecting info/status does not.
It stops that child for the placement release
sequence, commands the right tool through its dedicated daemon, verifies real
position feedback, then restarts the unchanged Cartesian arm bridge.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any


CONFIRMATION = "ENABLE_G2_GROOT_PLACE_ACTION_MUX"
ACTIVATION_CONFIRMATION = "ACTIVATE_G2_GROOT_PLACE_ARM"
ACTIVATION_PROTOCOL = "explicit_activate_v1"
SCHEMA = "g2_groot_persistent_h1_action_bridge_v3"
OPEN_THRESHOLD = -0.72
CLOSED_THRESHOLD = -0.03
GRIPPER_TARGET_TOLERANCE = 0.01
ARM_DRAIN_TIMEOUT_S = 8.0
# Measured on G2A0104C300179 on 2026-09-07: after the Cartesian arm child
# exits, ``move_ee_pos`` keeps returning 0 while the jaw does not move for
# about 4.4 s, then the jaw completes its travel in about 0.62 s.  The dead
# period is consistent with delayed ownership/discovery release; its internal
# cause is not proven. It and actuator travel are budgeted separately. A 5 s budget covering
# both left 21 ms of margin and is what produced the historical
# "physical gripper did not reach open threshold" failures.
GRIPPER_OWNERSHIP_RELEASE_TIMEOUT_S = 8.0
GRIPPER_TRAVEL_TIMEOUT_S = 3.0
# Settled feedback reads exactly 0.0, and the first real motion sample was
# 0.0012, so this clears quantisation without missing the start of travel.
GRIPPER_MOTION_EPSILON = 0.002
GRIPPER_COMMAND_REFRESH_S = 0.25
GRIPPER_POLL_PERIOD_S = 0.05
# Enough to hold the whole worst-case release trace at the poll period, so the
# ownership dead period is never truncated out of the evidence.
GRIPPER_SAMPLE_LIMIT = 256
GDK_OWNER_HANDOFF_SETTLE_S = 0.75
# Inbound runner requests carry one pose plus metadata, so they stay small.
MAX_REQUEST_LINE_BYTES = 16384
# Child status responses grow with the arm child's 32-entry ``recent_results``
# deque: each entry holds two 7-float poses plus metadata and a gripper status.
# At the previous shared 16 KiB ceiling the response was truncated from the
# 25th accumulated result onward, which is partway through the second H16
# chunk of a normal run.  ``request`` then raised "invalid response from port
# 9201" for every ``status`` call, so the runner's completion wait could never
# observe a receipt and failed the chunk with a settle timeout no matter how
# long its timeout was.  Match the runner client's own 4 MiB ceiling.
MAX_CHILD_RESPONSE_BYTES = 4 * 1024 * 1024


def gripper_fault(observation: dict[str, Any]) -> str | None:
    """Classify unambiguous tool hardware faults from one daemon observation.

    Only the two GDK error codes are treated as faults.  ``motor_status`` 2 is
    the normal holding state of a loaded omnipicker and appears throughout a
    release, and the effort of a workpiece-holding gripper already sits close
    to the state machine's 25.0 advisory limit, so neither is a safe abort
    condition here.  Both are still recorded for diagnosis.
    """
    if int(observation["motor_err_code"]) != 0:
        return f"motor_err_code={int(observation['motor_err_code'])}"
    if int(observation["whole_end_error"]) != 0:
        return f"whole_end_error={int(observation['whole_end_error'])}"
    return None


def command_float(value: float) -> str:
    """Format subprocess numeric arguments without exponent notation.

    ``argparse`` can classify a small negative value such as ``-8e-05`` as an
    option token instead of one of the values of a ``nargs`` argument.
    """
    return format(float(value), ".17f")


def request(port: int, payload: dict[str, Any], timeout: float = 30.0) -> dict:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall((json.dumps(payload) + "\n").encode())
        line = sock.makefile("rb").readline(MAX_CHILD_RESPONSE_BYTES + 1)
    if not line or len(line) > MAX_CHILD_RESPONSE_BYTES:
        raise RuntimeError(f"invalid response from port {port}")
    response = json.loads(line)
    if not response.get("ok"):
        raise RuntimeError(f"port {port} rejected request: {response}")
    return response


class ProcessMux:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.directory = Path(__file__).resolve().parent
        self.daemon: subprocess.Popen | None = None
        self.arm: subprocess.Popen | None = None
        self.cached_arm_status: dict[str, Any] | None = None
        self.release_phase = "closed"
        self.last_gripper_target = 0.0
        self.gripper_command_count = 0
        self.last_forwarded_arm_command_id: str | None = None
        self.completed_results: deque[dict[str, Any]] = deque(maxlen=128)
        self.handoff_error: str | None = None
        self.last_handoff: dict[str, Any] | None = None
        self.seen_command_ids: set[str] = set()
        self.activation_state = "standby"
        self.activation_error: str | None = None
        self.activation_owner: object | None = None

    def arm_alive(self) -> bool:
        return self.arm is not None and self.arm.poll() is None

    def activation_fields(self) -> dict[str, Any]:
        alive = self.arm_alive()
        if self.activation_state == "active" and not alive:
            self.activation_state = "fault"
            self.activation_error = (
                self.handoff_error or "activated arm bridge is not running; restart mux to retry"
            )
        return {
            "activation_protocol": ACTIVATION_PROTOCOL,
            "activation_state": self.activation_state,
            "arm_owner_active": alive,
        }

    def claim_owner(self, connection_id: object | None) -> None:
        if connection_id is None:
            return
        if self.activation_owner is not None and self.activation_owner is not connection_id:
            raise RuntimeError("arm activation belongs to another connection")
        self.activation_owner = connection_id

    def activate(
        self, payload: dict[str, Any], connection_id: object | None = None
    ) -> dict[str, Any]:
        if payload.get("confirm") != ACTIVATION_CONFIRMATION:
            raise ValueError(f"activate requires confirm={ACTIVATION_CONFIRMATION}")
        self.activation_fields()
        if self.activation_state == "active":
            self.claim_owner(connection_id)
            return {"ok": True, **self.activation_fields()}
        if self.activation_state != "standby":
            raise RuntimeError(
                self.activation_error
                or f"cannot activate from {self.activation_state}; restart mux"
            )
        self.claim_owner(connection_id)
        self.activation_state = "activating"
        try:
            self.start_arm()
            if not self.arm_alive():
                raise RuntimeError("arm bridge exited during activation")
        except Exception as error:
            self.activation_state = "fault"
            self.activation_error = (
                f"arm activation failed: {type(error).__name__}: {error}; restart mux to retry"
            )
            self.stop_arm()
            raise RuntimeError(self.activation_error) from error
        self.activation_state = "active"
        return {"ok": True, **self.activation_fields()}

    def disconnect(self, connection_id: object, *, shutdown: bool = False) -> None:
        if self.activation_owner is not connection_id:
            return
        # The activated model connection owns physical publishing. A failed
        # model or disconnected runner must not leave an idle holding child.
        self.activation_state = "stopped" if shutdown else "fault"
        self.activation_error = (
            None if shutdown else "activated client disconnected; restart mux to retry"
        )
        self.stop_arm()
        self.activation_owner = None

    @staticmethod
    def finish_child(process: subprocess.Popen, timeout: float = 3.0) -> None:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)

    def remember_results(self, status: dict[str, Any]) -> None:
        known = {item["command_id"] for item in self.completed_results}
        for item in status.get("recent_results", []):
            if item.get("command_id") and item["command_id"] not in known:
                self.completed_results.append(dict(item))
                known.add(item["command_id"])

    def wait_arm_idle(self) -> dict[str, Any]:
        started = time.monotonic()
        while True:
            status = self.arm_request({"op": "status"})
            if status.get("fatal_error"):
                raise RuntimeError(str(status["fatal_error"]))
            self.remember_results(status)
            receipt = next(
                (
                    item
                    for item in self.completed_results
                    if item["command_id"] == self.last_forwarded_arm_command_id
                ),
                None,
            )
            if status.get("queue_depth") == 0 and (
                self.last_forwarded_arm_command_id is None or receipt is not None
            ):
                if receipt is not None and (
                    receipt.get("accepted") is not True
                    or receipt.get("ok") is False
                    or receipt.get("error")
                ):
                    raise RuntimeError(f"arm execution failed before gripper handoff: {receipt}")
                self.cached_arm_status = status
                return {
                    "command_id": self.last_forwarded_arm_command_id,
                    "result": receipt,
                    "elapsed_s": time.monotonic() - started,
                }
            if time.monotonic() - started >= ARM_DRAIN_TIMEOUT_S:
                raise TimeoutError(
                    f"arm execution receipt missing before gripper handoff: "
                    f"{self.last_forwarded_arm_command_id}"
                )
            time.sleep(0.01)

    def _wait_port(self, port: int, process: subprocess.Popen, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"child exited with code {process.returncode}")
            try:
                request(
                    port, {"op": "ping"} if port == self.args.gripper_port else {"op": "info"}, 1.0
                )
                return
            except Exception as error:
                last_error = error
                time.sleep(0.1)
        raise RuntimeError(f"port {port} did not become ready: {last_error}")

    def start_daemon(self) -> None:
        self.daemon = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(self.directory / "g2_groot_right_gripper_command_daemon.py"),
                "--bind-host",
                "127.0.0.1",
                "--port",
                str(self.args.gripper_port),
                "--session-limit-s",
                str(self.args.session_limit_s + 30.0),
                "--confirm",
                "ENABLE_G2_GROOT_RIGHT_GRIPPER_COMMAND_DAEMON",
            ]
        )
        try:
            self._wait_port(self.args.gripper_port, self.daemon, 8.0)
        except Exception:
            self.finish_child(self.daemon, 0.0)
            self.daemon = None
            raise
        initial = self.gripper_status()["last_observation"]["raw_position"]
        if initial <= OPEN_THRESHOLD:
            self.release_phase = "open"
            self.last_gripper_target = initial

    def start_arm(self, *, restart: bool = False) -> None:
        command = [
            sys.executable,
            "-u",
            str(self.directory / "g2_groot_persistent_h1_action_bridge.py"),
            "--bind-host",
            "127.0.0.1",
            "--port",
            str(self.args.backend_port),
            "--enable-control",
            "--required-motion-mode",
            str(self.args.required_motion_mode),
            "--shutdown-gripper-action",
            "hold",
            "--right-session-max-translation-m",
            "0.50",
            "--right-session-max-rotation-deg",
            "180",
            "--workspace-min",
            *map(str, self.args.workspace_min),
            "--workspace-max",
            *map(str, self.args.workspace_max),
            "--startup-countdown-s",
            "0" if restart else "5",
            # Releasing a held workpiece changes the arm load enough that the
            # previous 0.3 s restart window cannot reliably reconverge the
            # Cartesian compensation.  Use the same complete calibration on
            # both startup and post-release reacquisition.
            "--calibration-duration-s",
            "2.0",
            "--session-limit-s",
            str(self.args.session_limit_s),
            "--confirm",
            "ENABLE_G2_GROOT_PERSISTENT_RIGHT_ARM_H1",
        ]
        # Do not seed the post-release child with the compensation the arm
        # adapted to while holding the workpiece.  That vector belongs to the
        # loaded arm, it can already sit at the 6.5 mm cap, and the child
        # rejects an over-cap seed at startup, which surfaces to the runner
        # only as "child exited with code 1".  The 2.0 s calibration above
        # re-derives it for the released load.  The pre-teardown value is kept
        # in ``cached_arm_status`` and reported for comparison.
        if not restart and self.args.initial_translation_compensation is not None:
            command.extend(
                [
                    "--initial-translation-compensation",
                    *map(
                        command_float,
                        self.args.initial_translation_compensation,
                    ),
                    "--initial-rotation-compensation",
                    *map(
                        command_float,
                        self.args.initial_rotation_compensation,
                    ),
                ]
            )
        self.arm = subprocess.Popen(command)
        try:
            self._wait_port(self.args.backend_port, self.arm, 15.0)
        except Exception:
            self.finish_child(self.arm, 0.0)
            self.arm = None
            raise

    def stop_arm(self) -> None:
        if self.arm is None:
            return
        try:
            self.cached_arm_status = request(self.args.backend_port, {"op": "status"}, 2.0)
            self.remember_results(self.cached_arm_status)
            request(self.args.backend_port, {"op": "shutdown"}, 2.0)
        except Exception:
            pass
        self.finish_child(self.arm, 5.0)
        self.arm = None
        # Ensure DDS discovery has removed the Cartesian publisher before the
        # transaction-style tool command is issued.
        time.sleep(GDK_OWNER_HANDOFF_SETTLE_S)

    def gripper_status(self) -> dict[str, Any]:
        raw = request(self.args.gripper_port, {"op": "status"}, 2.0)
        position = float(raw["position"])
        observation = {
            "monotonic_s": time.monotonic(),
            "raw_position": position,
            "motor_status": int(raw["status"]),
            "motor_err_code": int(raw["err_code"]),
            "whole_end_error": int(raw["whole_end_error"]),
            "effort": float(raw["effort"]),
        }
        return {
            "closure_enabled": True,
            "feedback_encoding": "native_radians",
            "desired": self.last_gripper_target,
            "last_completed": position,
            "active_target": None,
            "recovery_requested": False,
            # Report the real tool health.  A hardcoded ``None`` here made the
            # runner's preflight gripper check unreachable.
            "fault": gripper_fault(observation),
            "last_observation": observation,
        }

    def arm_request(self, payload: dict[str, Any]) -> dict:
        if not self.arm_alive():
            raise RuntimeError("arm bridge is not running")
        response = request(self.args.backend_port, payload)
        if payload.get("op") == "execute_h1":
            if response.get("accepted") is not True:
                raise RuntimeError(f"arm did not accept waypoint: {response}")
            self.last_forwarded_arm_command_id = payload["command_id"]
        return response

    def status(self) -> dict[str, Any]:
        activation = self.activation_fields()
        arm_owner_active = activation["arm_owner_active"]
        arm = (
            self.arm_request({"op": "status"})
            if arm_owner_active
            else dict(self.cached_arm_status or {})
        )
        self.remember_results(arm)
        gripper = self.gripper_status()
        position = gripper["last_observation"]["raw_position"]
        self.release_phase = (
            "open"
            if position <= OPEN_THRESHOLD
            else "closed"
            if position >= CLOSED_THRESHOLD
            else "partial"
        )
        arm.update(
            {
                "ok": True,
                "schema": SCHEMA,
                # Never claim readiness, or a fresh ``live_pose``, while no
                # Cartesian arm child owns GDK.
                "ready": self.activation_state == "active"
                and arm_owner_active
                and not self.handoff_error
                and bool(arm.get("ready", False)),
                "fatal_error": self.activation_error
                or self.handoff_error
                or arm.get("fatal_error")
                or (None if self.activation_state in ("standby", "stopped") or arm_owner_active
                    else "arm bridge is not running"),
                **activation,
                "live_pose_stale": not arm_owner_active,
                "right_gripper_enabled": True,
                "right_gripper": gripper,
                "release_phase": self.release_phase,
                "recent_results": list(self.completed_results),
                "last_handoff": self.last_handoff,
            }
        )
        return arm

    def info(self) -> dict[str, Any]:
        activation = self.activation_fields()
        # Standby inspection is deliberately independent of the arm child.
        # These describe the unchanged legacy protocol, not physical readiness.
        arm = {
            "server_time_ns": time.time_ns(),
            "bind_loopback_only": True,
            "right_arm_enabled": True,
            "left_arm_enabled": False,
            "head_waist_chassis_enabled": False,
            "maximum_horizon": 16,
            "model_waypoint_hz": 10.0,
            "control_hz": 50.0,
            "interpolation_ticks": 5,
            "maximum_translation_step_m": 0.25,
            "maximum_rotation_step_rad": math.pi,
            "maximum_command_age_s": 5.0,
            "right_session_max_translation_m": 0.50,
            "right_session_max_rotation_rad": math.pi,
            "acknowledgement": "immediate_queue_acceptance",
            "execution_results": "status.recent_results",
        }
        if activation["arm_owner_active"]:
            arm.update(self.arm_request({"op": "info"}))
        arm.update(
            {
                "ok": True,
                "schema": SCHEMA,
                "mode": "cartesian_arm_serialized_place_release",
                "operations": [
                    "info", "status", "activate", "execute_h1_gripper", "set_gripper", "shutdown"
                ],
                **activation,
                "right_gripper_enabled": True,
                "right_gripper_protected_closure_enabled": True,
                "right_gripper_physical_zero_closure_enabled": True,
                "right_gripper_command_range": [-0.785, 0.0],
                "shutdown_gripper_action": "hold",
                "gripper_command_mode": "process_handoff",
                "gripper_policy_mode": "endpoint_process_handoff",
                "gripper_partial_targets": "deferred_until_endpoint",
            }
        )
        return arm

    def gripper_command(self, target: float) -> None:
        request(self.args.gripper_port, {"op": "command", "target": target}, 3.0)
        self.gripper_command_count += 1

    def open_gripper(self, target: float) -> dict[str, Any]:
        return self.move_gripper(target)

    def move_gripper(self, target: float) -> dict[str, Any]:
        """Execute the requested opening or closing target with actual feedback."""
        samples: list[dict[str, Any]] = []
        commands_before = self.gripper_command_count
        started = time.monotonic()

        def observe() -> dict[str, Any]:
            observation = self.gripper_status()["last_observation"]
            samples.append(
                {
                    "elapsed_s": time.monotonic() - started,
                    "raw_position": observation["raw_position"],
                    "motor_status": observation["motor_status"],
                    "motor_err_code": observation["motor_err_code"],
                    "whole_end_error": observation["whole_end_error"],
                    "effort": observation["effort"],
                }
            )
            return observation

        observation = observe()
        start_position = observation["raw_position"]
        closing = target > start_position
        self.gripper_command(target)
        next_refresh = started + GRIPPER_COMMAND_REFRESH_S
        motion_started: float | None = None
        stationary_since = started
        previous_position = start_position

        def summary(final: dict[str, Any]) -> dict[str, Any]:
            now = time.monotonic()
            return {
                "target": target,
                "start_position": start_position,
                "opened_position": final["raw_position"],
                "final_position": final["raw_position"],
                "direction": "closing" if closing else "opening",
                "travel": start_position - final["raw_position"],
                "elapsed_s": now - started,
                "ownership_release_s": (
                    None if motion_started is None else motion_started - started
                ),
                "travel_s": (None if motion_started is None else now - motion_started),
                "commands_sent": self.gripper_command_count - commands_before,
                "samples": samples[-GRIPPER_SAMPLE_LIMIT:],
            }

        while True:
            observation = observe()
            fault = gripper_fault(observation)
            if fault is not None:
                raise RuntimeError(
                    f"tool hardware fault during release: {fault}; "
                    f"target={target!r} start={start_position!r} "
                    f"position={observation['raw_position']!r}"
                )
            position = observation["raw_position"]
            if abs(position - target) <= GRIPPER_TARGET_TOLERANCE:
                return summary(observation)
            now = time.monotonic()
            if motion_started is None and abs(position - start_position) >= GRIPPER_MOTION_EPSILON:
                motion_started = now
            if abs(position - previous_position) >= GRIPPER_MOTION_EPSILON:
                stationary_since = now
                previous_position = position
            if (
                closing
                and motion_started is not None
                and position - start_position >= GRIPPER_MOTION_EPSILON
                and observation["motor_status"] in (2, 3)
                and now - stationary_since >= 0.2
            ):
                return {**summary(observation), "contact_hold": True}
            if motion_started is None:
                if now - started >= GRIPPER_OWNERSHIP_RELEASE_TIMEOUT_S:
                    # No jaw motion at all: the command is being accepted and
                    # silently dropped, which means a Cartesian owner is still
                    # present.  This is not an actuator problem.
                    raise RuntimeError(
                        "gripper never started moving, so GDK tool ownership "
                        "was not released: "
                        f"target={target!r} start={start_position!r} "
                        f"position={position!r} "
                        f"motor_status={observation['motor_status']} "
                        f"effort={observation['effort']!r} "
                        f"commands_sent="
                        f"{self.gripper_command_count - commands_before} "
                        f"waited_s={now - started!r} "
                        f"limit_s={GRIPPER_OWNERSHIP_RELEASE_TIMEOUT_S!r}"
                    )
            elif now - motion_started >= GRIPPER_TRAVEL_TIMEOUT_S:
                # The jaw moved but did not finish: mechanically slow or
                # obstructed, which is an actuator or workpiece problem.
                raise RuntimeError(
                    "gripper started moving but did not reach target "
                    f"within {GRIPPER_TARGET_TOLERANCE}: target={target!r} "
                    f"start={start_position!r} position={position!r} "
                    f"travel={start_position - position!r} "
                    f"motor_status={observation['motor_status']} "
                    f"effort={observation['effort']!r} "
                    f"ownership_release_s={motion_started - started!r} "
                    f"travel_s={now - motion_started!r} "
                    f"limit_s={GRIPPER_TRAVEL_TIMEOUT_S!r}"
                )
            if now >= next_refresh:
                self.gripper_command(target)
                next_refresh = now + GRIPPER_COMMAND_REFRESH_S
            time.sleep(GRIPPER_POLL_PERIOD_S)

    def release_and_resume(
        self, target: float, arm_payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Complete this row's EEF once, hand off to the tool, then resume.

        A gripper endpoint must not happen at the previous row's EEF pose.
        The original arm child validates and executes the triggering row
        before any tool command. A restarted child only receives future rows.
        """
        if self.handoff_error:
            raise RuntimeError(self.handoff_error)
        started = time.monotonic()
        pre_release_compensation = None
        policy_timestamp_ns = None if arm_payload is None else arm_payload.get("timestamp_ns")
        self.last_handoff = {
            "target": target,
            "policy_timestamp_ns": policy_timestamp_ns,
            "status": "STARTED",
        }
        try:
            response = (
                {"ok": True} if arm_payload is None else dict(self.arm_request(dict(arm_payload)))
            )
            drain = self.wait_arm_idle()
            arm_pose_before_tool = list((self.cached_arm_status or {}).get("live_pose", []))
            self.last_handoff.update(arm_drain=drain, arm_pose_before_tool=arm_pose_before_tool)
            self.stop_arm()
            if self.cached_arm_status is not None:
                pre_release_compensation = {
                    "translation_compensation_m": self.cached_arm_status.get(
                        "translation_compensation_m"
                    ),
                    "rotation_compensation_xyzw": self.cached_arm_status.get(
                        "rotation_compensation_xyzw"
                    ),
                }
            release = self.move_gripper(target)
            self.last_handoff.update(release)
            self.release_phase = (
                "open"
                if release["final_position"] <= OPEN_THRESHOLD
                else "closed"
                if release["final_position"] >= CLOSED_THRESHOLD
                else "partial"
            )
            self.start_arm(restart=True)
            resumed = self.arm_request({"op": "status"})
            if resumed.get("ready") is not True or resumed.get("fatal_error"):
                raise RuntimeError(f"arm restart is not ready after gripper handoff: {resumed}")
            self.remember_results(resumed)
        except Exception as error:
            self.handoff_error = f"gripper handoff failed: {type(error).__name__}: {error}"
            self.activation_state = "fault"
            self.activation_error = self.handoff_error
            try:
                self.stop_arm()
            except Exception as stop_error:
                self.handoff_error += (
                    f"; arm shutdown failed: {type(stop_error).__name__}: {stop_error}"
                )
                self.activation_error = self.handoff_error
            self.last_handoff.update(status="FAILED", error=self.handoff_error)
            raise RuntimeError(self.handoff_error) from error
        handoff = {
            **release,
            "handoff_s": time.monotonic() - started,
            "policy_timestamp_ns": policy_timestamp_ns,
            "arm_row_completed_before_tool": arm_payload is not None,
            "arm_resumed_ready": True,
            "pre_release_compensation": pre_release_compensation,
            "arm_pose_before_tool": arm_pose_before_tool,
            "arm_drain": drain,
            "status": "COMPLETED",
        }
        self.last_handoff = handoff
        response["gripper_handoff"] = handoff
        if target <= OPEN_THRESHOLD:
            response["release"] = handoff
        return response

    @staticmethod
    def validate_gripper_target(value: Any) -> float:
        target = float(value)
        if not math.isfinite(target) or not -0.7851 <= target <= 0.0001:
            raise ValueError("target_gripper must be finite and within [-0.785, 0]")
        return target

    def set_gripper(self, target: float) -> dict[str, Any]:
        self.activation_fields()
        if self.activation_state not in ("standby", "active"):
            raise RuntimeError(
                self.activation_error
                or f"cannot set gripper from {self.activation_state}; restart mux"
            )
        target = self.validate_gripper_target(target)
        self.last_gripper_target = target
        if self.activation_state == "standby":
            result = self.move_gripper(target)
            self.last_handoff = {**result, "status": "COMPLETED", "standby_tool_only": True}
            return {"ok": True, "gripper_handoff": self.last_handoff, **self.activation_fields()}
        return self.release_and_resume(target, None)

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.activation_fields()
        if self.activation_state != "active":
            raise RuntimeError(self.activation_error or "arm is not activated; send activate first")
        if self.handoff_error:
            raise RuntimeError(self.handoff_error)
        command_id = payload.get("command_id")
        if not isinstance(command_id, str) or not command_id.strip() or len(command_id) > 128:
            raise ValueError("command_id is required and must be <=128 chars")
        if command_id in self.seen_command_ids:
            raise ValueError(f"duplicate command_id: {command_id}")
        self.seen_command_ids.add(command_id)
        target = self.validate_gripper_target(payload["target_gripper"])
        self.last_gripper_target = target
        arm_payload = dict(payload)
        arm_payload["op"] = "execute_h1"
        arm_payload.pop("target_gripper", None)

        gripper = self.gripper_status()
        if gripper["fault"]:
            raise RuntimeError(f"tool hardware fault: {gripper['fault']}")
        position = gripper["last_observation"]["raw_position"]
        self.release_phase = (
            "open"
            if position <= OPEN_THRESHOLD
            else "closed"
            if position >= CLOSED_THRESHOLD
            else "partial"
        )
        endpoint = target <= OPEN_THRESHOLD or target >= CLOSED_THRESHOLD
        satisfied = bool(
            (target <= OPEN_THRESHOLD and position <= OPEN_THRESHOLD)
            or (target >= CLOSED_THRESHOLD and position >= CLOSED_THRESHOLD)
            or abs(position - target) <= GRIPPER_TARGET_TOLERANCE
        )
        contact = self.last_handoff or {}
        contact_satisfied = bool(
            contact.get("contact_hold")
            and target >= contact["target"] - GRIPPER_TARGET_TOLERANCE
            and abs(position - contact["final_position"]) <= GRIPPER_TARGET_TOLERANCE
        )
        if endpoint and not satisfied and not contact_satisfied:
            response = self.release_and_resume(target, arm_payload)
            execution = "handoff"
        else:
            response = self.arm_request(arm_payload)
            execution = "satisfied" if satisfied or contact_satisfied else "deferred_until_endpoint"
        response["gripper_execution"] = execution
        response["target_gripper"] = target
        return response

    def close(self) -> None:
        self.stop_arm()
        if self.daemon is not None:
            try:
                request(self.args.gripper_port, {"op": "shutdown"}, 2.0)
            except Exception:
                pass
            self.finish_child(self.daemon)
            self.daemon = None


def send_response(connection: socket.socket, payload: dict[str, Any]) -> bool:
    try:
        connection.sendall((json.dumps(payload) + "\n").encode())
        return True
    except OSError:
        return False


def serve_connection(mux: ProcessMux, connection: socket.socket, deadline: float) -> bool:
    """Serve one persistent session; only its physical owner is stopped on EOF."""
    connection_id = object()
    stopping = False
    try:
        with connection, connection.makefile("rb") as stream:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while not stopping and time.monotonic() < deadline:
                connection.settimeout(min(60.0, max(0.01, deadline - time.monotonic())))
                try:
                    line = stream.readline(MAX_REQUEST_LINE_BYTES + 1)
                except OSError:
                    break
                if not line:
                    break
                if len(line) > MAX_REQUEST_LINE_BYTES:
                    send_response(connection, {"ok": False, "message": "request too large"})
                    break
                try:
                    payload = json.loads(line)
                    operation = payload.get("op")
                    if operation == "info":
                        response = mux.info()
                    elif operation == "status":
                        response = mux.status()
                    elif operation == "activate":
                        response = mux.activate(payload, connection_id)
                    elif operation == "execute_h1_gripper":
                        if mux.activation_state == "active":
                            mux.claim_owner(connection_id)
                        response = mux.execute(payload)
                    elif operation == "set_gripper":
                        if mux.activation_state == "active":
                            mux.claim_owner(connection_id)
                        response = mux.set_gripper(payload["target_gripper"])
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
                if not send_response(connection, response):
                    break
    finally:
        mux.disconnect(connection_id, shutdown=stopping)
    return stopping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--backend-port", type=int, default=9201)
    parser.add_argument("--gripper-port", type=int, default=9300)
    parser.add_argument("--required-motion-mode", type=int, default=1)
    parser.add_argument("--workspace-min", nargs=3, type=float, required=True)
    parser.add_argument("--workspace-max", nargs=3, type=float, required=True)
    parser.add_argument("--initial-translation-compensation", nargs=3, type=float)
    parser.add_argument("--initial-rotation-compensation", nargs=4, type=float)
    parser.add_argument("--session-limit-s", type=float, default=1800.0)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.bind_host not in ("127.0.0.1", "localhost"):
        parser.error("mux must bind to loopback")
    if args.confirm != CONFIRMATION:
        parser.error(f"physical control requires --confirm {CONFIRMATION}")
    if not 30 <= args.session_limit_s <= 3570:
        parser.error("--session-limit-s must be in [30, 3570]")
    if (args.initial_translation_compensation is None) != (
        args.initial_rotation_compensation is None
    ):
        parser.error("both initial compensation vectors are required together")

    mux = ProcessMux(args)
    deadline = time.monotonic() + args.session_limit_s
    try:
        mux.start_daemon()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.bind_host, args.port))
            listener.listen(4)
            listener.settimeout(0.5)
            print(json.dumps({"event": "mux_ready", "port": args.port}), flush=True)
            stopping = False
            while time.monotonic() < deadline and not stopping:
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                stopping = serve_connection(mux, connection, deadline)
        return 0
    finally:
        mux.close()


if __name__ == "__main__":
    raise SystemExit(main())
