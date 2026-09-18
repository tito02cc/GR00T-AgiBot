#!/usr/bin/env python3
"""Protocol-faithful stand-in for the Cartesian arm child, for bench testing.

This process owns no GDK objects and publishes nothing, so it requests no arm
motion at all.  It exists so the real placement action mux can perform its real
ownership handoff against the real omnipicker daemon on a robot whose arm is
not positioned for the task.

It mirrors the parts of ``g2_groot_persistent_h1_action_bridge.py`` that the
mux and the runner actually depend on:

* the same CLI surface the mux launches the child with;
* the 5 s command staleness limit, which is what rejects a handoff replay that
  still carries the original policy timestamp;
* the 6.5 mm seeded-compensation cap, whose ``ValueError`` is what turns into
  "child exited with code 1" for the caller;
* the ``recent_results`` / ``queue_depth`` contract the runner settles against;
* the 5 interpolation ticks per 100 ms waypoint execution cadence.

Everything it reports about physical pose is simulated and labelled as such.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any

CONFIRMATION = "ENABLE_G2_GROOT_PERSISTENT_RIGHT_ARM_H1"
ARM_GRIPPER_SCHEMA = "g2_groot_persistent_h1_action_bridge_v3"
MAX_COMMAND_AGE_S = 5.0
MAX_TOTAL_TRANSLATION_COMPENSATION_M = 0.0065
MAX_TOTAL_ROTATION_COMPENSATION_RAD = 0.05
MAX_TARGET_STEP_M = 0.05
MAX_TARGET_ROTATION_RAD = 0.35
CONTROL_HZ = 50.0
MODEL_PERIOD_S = 0.1
MODEL_TICKS = 5
MAX_LINE_BYTES = 16384


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "simulated": True, **fields}), flush=True)


def distance(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((first[i] - second[i]) ** 2 for i in range(3)))


def quaternion_angle(first: list[float], second: list[float]) -> float:
    dot = abs(sum(first[3 + i] * second[3 + i] for i in range(4)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


class SimulatedArm:
    """Tracks a simulated desired/live pose and an execution result history."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        # The real controller anchors its session to the live TF pose at
        # startup, so a child restarted mid-task continues from wherever the
        # arm physically is.  A restarted stub must do the same or the first
        # replayed waypoint looks like a large jump.  The persisted pose stands
        # in for that TF read.
        anchor = list(args.simulated_start_pose)
        state = Path(args.simulated_pose_state)
        if state.exists():
            try:
                stored = json.loads(state.read_text())["pose"]
                if isinstance(stored, list) and len(stored) == 7:
                    anchor = [float(value) for value in stored]
                    emit("anchored_to_persisted_pose", pose=anchor)
            except Exception as error:
                emit("pose_state_unreadable", message=str(error))
        self.desired = list(anchor)
        self.live = list(anchor)
        self.session_start = list(anchor)
        self.recent_results: deque[dict[str, Any]] = deque(maxlen=32)
        self.queue_depth = 0
        self.tick_count = 0
        self.calibrated = False
        self.fatal_error: str | None = None
        self.last_command_id = ""
        self.last_result: dict[str, Any] | None = None
        self.started_monotonic = time.monotonic()
        self.translation_compensation = [0.0, 0.0, 0.0]
        self.rotation_compensation = [0.0, 0.0, 0.0, 1.0]
        self._seed_compensation()

    def _seed_compensation(self) -> None:
        """Reject an over-cap seed exactly like the real controller does.

        The real ``PersistentRightArmController.__init__`` raises here, the
        child exits non-zero, and the mux surfaces only "child exited with
        code 1".  Reproducing that is the point of this stub.
        """
        if self.args.initial_translation_compensation is None:
            return
        seed = list(self.args.initial_translation_compensation)
        norm = math.sqrt(sum(value * value for value in seed))
        if norm > MAX_TOTAL_TRANSLATION_COMPENSATION_M + 1e-9:
            raise ValueError("initial translation compensation exceeds limit")
        if norm > MAX_TOTAL_TRANSLATION_COMPENSATION_M:
            scale = MAX_TOTAL_TRANSLATION_COMPENSATION_M / norm
            seed = [value * scale for value in seed]
        self.translation_compensation = seed
        rotation = list(self.args.initial_rotation_compensation)
        angle = 2.0 * math.acos(max(-1.0, min(1.0, abs(rotation[3]))))
        if angle > MAX_TOTAL_ROTATION_COMPENSATION_RAD:
            raise ValueError("initial rotation compensation exceeds limit")
        self.rotation_compensation = rotation
        emit(
            "seeded_compensation",
            translation_compensation_m=self.translation_compensation,
            rotation_compensation_xyzw=self.rotation_compensation,
        )

    def calibrate(self) -> None:
        if self.args.startup_countdown_s > 0:
            for remaining in range(int(self.args.startup_countdown_s), 0, -1):
                emit("physical_countdown", remaining_s=remaining)
                time.sleep(1.0)
        emit("calibration_start", duration_s=self.args.calibration_duration_s)
        time.sleep(self.args.calibration_duration_s)
        with self.lock:
            self.calibrated = True
        emit("calibration_complete", calibrated=True)

    def validate_target(self, request: dict[str, Any]) -> tuple[list[float], float]:
        target = request.get("target_pose")
        if not isinstance(target, list) or len(target) != 7:
            raise ValueError("target_pose must contain seven values")
        target = [float(value) for value in target]
        if not all(math.isfinite(value) for value in target):
            raise ValueError("target_pose must be finite")
        with self.lock:
            reference = list(self.desired)
        step = distance(reference, target)
        if step > MAX_TARGET_STEP_M:
            raise ValueError(
                f"translation step {step:.6f}m exceeds {MAX_TARGET_STEP_M:.6f}m"
            )
        rotation = quaternion_angle(reference, target)
        if rotation > MAX_TARGET_ROTATION_RAD:
            raise ValueError(
                f"rotation step {rotation:.6f}rad exceeds "
                f"{MAX_TARGET_ROTATION_RAD:.6f}rad"
            )
        timestamp_ns = int(request.get("timestamp_ns", 0))
        if timestamp_ns <= 0:
            raise ValueError("timestamp_ns is required")
        age_s = max(0.0, (time.time_ns() - timestamp_ns) / 1e9)
        if age_s > MAX_COMMAND_AGE_S:
            raise ValueError(
                f"command age {age_s:.3f}s exceeds {MAX_COMMAND_AGE_S:.3f}s"
            )
        for index in range(3):
            if not (
                self.args.workspace_min[index] - 0.01
                <= target[index]
                <= self.args.workspace_max[index] + 0.01
            ):
                raise ValueError("compensated command outside extended workspace")
        return target, age_s

    def execute(self, command_id: str, target: list[float], gripper: Any) -> None:
        """Consume one waypoint over the real 5 tick / 100 ms budget."""
        submitted = time.monotonic()
        with self.lock:
            self.queue_depth += 1
        time.sleep(MODEL_TICKS / CONTROL_HZ)
        with self.lock:
            self.desired = list(target)
            # A simulated arm converges exactly; no physical error is invented.
            self.live = list(target)
            self.tick_count += MODEL_TICKS
            self.queue_depth -= 1
            result = {
                "command_id": command_id,
                "accepted": True,
                "queue_delay_s": 0.0,
                "duration_s": time.monotonic() - submitted,
                "ticks": MODEL_TICKS,
                "target_pose": list(target),
                "live_pose_at_100ms": list(self.live),
                "position_error_at_100ms_m": 0.0,
                "rotation_error_at_100ms_rad": 0.0,
                "gripper_target": gripper,
                "gripper_status": None,
                "simulated": True,
            }
            self.last_command_id = command_id
            self.last_result = result
            self.recent_results.append(result)
            pose = list(self.live)
        self._persist_pose(pose)

    def _persist_pose(self, pose: list[float]) -> None:
        """Stand in for the physical pose a restarted child would read from TF."""
        try:
            Path(self.args.simulated_pose_state).write_text(
                json.dumps({"pose": pose}), encoding="utf-8"
            )
        except Exception as error:
            emit("pose_state_unwritable", message=str(error))

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "schema": ARM_GRIPPER_SCHEMA,
                "simulated_arm": True,
                "ready": self.calibrated and self.fatal_error is None,
                "fatal_error": self.fatal_error,
                "control_hz": CONTROL_HZ,
                "model_waypoint_hz": 1.0 / MODEL_PERIOD_S,
                "interpolation_ticks": MODEL_TICKS,
                "desired_pose": list(self.desired),
                "last_accepted_pose": list(self.desired),
                "live_pose": list(self.live),
                "live_target_position_error_m": 0.0,
                "live_target_rotation_error_rad": 0.0,
                "translation_compensation_m": list(self.translation_compensation),
                "rotation_compensation_xyzw": list(self.rotation_compensation),
                "tick_count": self.tick_count,
                "queue_depth": self.queue_depth,
                "last_command_id": self.last_command_id,
                "last_result": self.last_result,
                "recent_results": list(self.recent_results),
                "session_age_s": time.monotonic() - self.started_monotonic,
                "right_gripper_enabled": False,
                "right_gripper": None,
            }

    def info(self) -> dict[str, Any]:
        return {
            "server_time_ns": time.time_ns(),
            "schema": ARM_GRIPPER_SCHEMA,
            "simulated_arm": True,
            "mode": "simulated_persistent_right_arm_h1",
            "operations": ["info", "status", "execute_h1", "shutdown"],
            "bind_loopback_only": True,
            "right_arm_enabled": True,
            "right_gripper_enabled": False,
            "right_gripper_protected_closure_enabled": True,
            "right_gripper_physical_zero_closure_enabled": True,
            "right_gripper_command_range": [-0.785, 0.0],
            "left_arm_enabled": False,
            "head_waist_chassis_enabled": False,
            "maximum_horizon": 16,
            "model_waypoint_hz": 10.0,
            "control_hz": 50.0,
            "interpolation_ticks": MODEL_TICKS,
            "maximum_translation_step_m": MAX_TARGET_STEP_M,
            "maximum_rotation_step_rad": MAX_TARGET_ROTATION_RAD,
            "maximum_command_age_s": MAX_COMMAND_AGE_S,
            "shutdown_gripper_action": "hold",
            "gripper_command_mode": "external_daemon",
            "acknowledgement": "immediate_queue_acceptance",
            "execution_results": "status.recent_results",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9201)
    parser.add_argument("--enable-control", action="store_true")
    parser.add_argument("--required-motion-mode", type=int, default=1)
    parser.add_argument("--shutdown-gripper-action", default="hold")
    parser.add_argument("--right-session-max-translation-m", type=float, default=0.5)
    parser.add_argument("--right-session-max-rotation-deg", type=float, default=180.0)
    parser.add_argument("--workspace-min", nargs=3, type=float, required=True)
    parser.add_argument("--workspace-max", nargs=3, type=float, required=True)
    parser.add_argument("--startup-countdown-s", type=float, default=0.0)
    parser.add_argument("--calibration-duration-s", type=float, default=2.0)
    parser.add_argument("--session-limit-s", type=float, default=1800.0)
    parser.add_argument("--initial-translation-compensation", nargs=3, type=float)
    parser.add_argument("--initial-rotation-compensation", nargs=4, type=float)
    parser.add_argument(
        "--simulated-start-pose",
        nargs=7,
        type=float,
        default=[
            0.5013904571533203,
            -0.1765020340681076,
            1.0539140701293945,
            0.5225321066771904,
            -0.001878945069661977,
            0.8526121988491865,
            0.0030175205845201216,
        ],
        help="pose this stub reports as its starting live/desired pose",
    )
    parser.add_argument(
        "--simulated-pose-state",
        default="/tmp/sim_arm_child_pose.json",
        help="file standing in for the TF pose a restarted child would read",
    )
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.bind_host not in ("127.0.0.1", "localhost"):
        parser.error("stub must bind to loopback")
    if args.confirm != CONFIRMATION:
        parser.error(f"this stub mirrors the child's --confirm {CONFIRMATION}")
    return args


def main() -> int:
    args = parse_args()
    emit(
        "stub_start",
        note="no GDK objects, no publishers, no arm motion is requested",
        port=args.port,
        calibration_duration_s=args.calibration_duration_s,
        startup_countdown_s=args.startup_countdown_s,
        seeded_translation_compensation=args.initial_translation_compensation,
    )
    try:
        arm = SimulatedArm(args)
    except ValueError as error:
        # Same observable outcome as the real child: non-zero exit, which the
        # mux reports as "child exited with code 1".
        emit("fatal", error_type="ValueError", message=str(error))
        return 1

    arm.calibrate()
    deadline = time.monotonic() + args.session_limit_s
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.bind_host, args.port))
        listener.listen(4)
        listener.settimeout(0.5)
        emit("stub_ready", port=args.port)
        stopping = False
        while time.monotonic() < deadline and not stopping:
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            with connection, connection.makefile("rb") as stream:
                while not stopping:
                    line = stream.readline(MAX_LINE_BYTES + 1)
                    if not line:
                        break
                    try:
                        request = json.loads(line)
                        operation = request.get("op")
                        if operation == "info":
                            response = {"ok": True, **arm.info()}
                        elif operation == "status":
                            response = {"ok": True, **arm.status()}
                        elif operation in ("execute_h1", "execute_h1_gripper"):
                            command_id = str(request.get("command_id", "")).strip()
                            if not command_id or len(command_id) > 128:
                                raise ValueError(
                                    "command_id is required and must be <=128 chars"
                                )
                            target, age_s = arm.validate_target(request)
                            arm.execute(
                                command_id, target, request.get("target_gripper")
                            )
                            response = {
                                "ok": True,
                                "command_age_s": age_s,
                                "command_id": command_id,
                                "accepted": True,
                                "queued": True,
                                "gripper_target": request.get("target_gripper"),
                            }
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
                    connection.sendall((json.dumps(response) + "\n").encode())
                    if stopping:
                        break
    emit("stub_stopped", tick_count=arm.tick_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
