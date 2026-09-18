#!/usr/bin/env python3
"""Persistent 50 Hz G2 right-arm EEF controller and guarded validation.

The controller keeps one GDK session alive from calibration through waypoint
execution.  It accepts 10 Hz absolute EEF waypoints and interpolates five
20 ms control ticks between them.  Default operation is strictly read-only.
No gripper, left-arm, head, waist, or chassis command is implemented here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time
from typing import Any

import agibot_gdk
from g2_groot_recover_pre_wbc_pose import clamp, normalize, q_to_rv, qinv, qmul, rv_to_q, send
from g2_groot_trajectory_tracking_probe import (
    LEFT_FRAME,
    RIGHT_FRAME,
    competing_controllers,
    emit,
    pose_values,
    quaternion_angle,
    require_safe_state,
    snapshot,
    translation_distance,
)
import numpy as np


CONFIRMATION = "TEST_G2_GROOT_PERSISTENT_50HZ_RIGHT_ARM"
CONTROL_HZ = 50.0
CONTROL_PERIOD_S = 1.0 / CONTROL_HZ
MODEL_PERIOD_S = 0.1
MODEL_TICKS = 5

WORKSPACE_MIN = np.asarray([0.448, -0.333, 0.987])
WORKSPACE_MAX = np.asarray([0.754, -0.197, 1.217])
SEED_TRANSLATION_COMPENSATION = np.asarray(
    [-0.0009471253, 0.0004121550, 0.0040800230]
)
SEED_ROTATION_COMPENSATION = normalize(
    [0.0005071716, 0.0065928102, -0.0006921396, 0.9999778990]
)

TRANSLATION_GAIN = 0.12
ROTATION_GAIN = 0.10
MAX_TRANSLATION_CORRECTION_PER_TICK_M = 0.00015
MAX_ROTATION_CORRECTION_PER_TICK_RAD = 0.0005
MAX_TOTAL_TRANSLATION_COMPENSATION_M = 0.0065
MAX_TOTAL_ROTATION_COMPENSATION_RAD = 0.05

LEFT_SESSION_MAX_TRANSLATION_M = 0.0005
LEFT_SESSION_MAX_ROTATION_RAD = 0.005
RIGHT_SESSION_MAX_TRANSLATION_M = 0.004
RIGHT_SESSION_MAX_ROTATION_RAD = 0.05
APPROACH_SESSION_MAX_TRANSLATION_M = 0.240
APPROACH_SESSION_MAX_ROTATION_RAD = math.radians(20.0)
MAX_ALLOWED_SESSION_TRANSLATION_M = 0.550
MAX_ALLOWED_SESSION_ROTATION_RAD = math.radians(180.0)
# One command is one native 10 Hz policy waypoint.  The controller performs
# the only required interpolation: five 50 Hz GDK targets in that 100 ms.
MAX_TARGET_STEP_M = 0.25
MAX_TARGET_ROTATION_RAD = math.radians(180.0)
MAX_LIVE_TARGET_ERROR_M = 0.25
MAX_LIVE_TARGET_ERROR_RAD = math.radians(180.0)
CALIBRATION_MAX_TRANSIENT_M = 0.020
CALIBRATION_MAX_TRANSIENT_RAD = 0.05
CONVERGENCE_POSITION_M = 0.003
CONVERGENCE_ROTATION_RAD = 0.02


def interpolate_quaternion(first: np.ndarray, second: np.ndarray, alpha: float):
    q0 = normalize(first)
    q1 = normalize(second)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    return normalize((1.0 - alpha) * q0 + alpha * q1)


@dataclass
class TickMetrics:
    left_translation_m: float
    left_rotation_rad: float
    right_session_translation_m: float
    right_session_rotation_rad: float
    target_position_error_m: float
    target_rotation_error_rad: float

    def as_dict(self):
        return vars(self).copy()


class PersistentRightArmController:
    def __init__(
        self,
        robot: Any,
        tf: Any,
        *,
        right_session_max_translation_m: float = RIGHT_SESSION_MAX_TRANSLATION_M,
        right_session_max_rotation_rad: float = RIGHT_SESSION_MAX_ROTATION_RAD,
        initial_translation_compensation: Any | None = None,
        initial_rotation_compensation: Any | None = None,
        workspace_min: Any = WORKSPACE_MIN,
        workspace_max: Any = WORKSPACE_MAX,
        required_motion_mode: int = 5,
    ):
        if not 0 < right_session_max_translation_m <= MAX_ALLOWED_SESSION_TRANSLATION_M:
            raise ValueError("invalid right session translation limit")
        if not 0 < right_session_max_rotation_rad <= MAX_ALLOWED_SESSION_ROTATION_RAD:
            raise ValueError("invalid right session rotation limit")
        self.robot = robot
        self.tf = tf
        self.right_session_max_translation_m = right_session_max_translation_m
        self.right_session_max_rotation_rad = right_session_max_rotation_rad
        self.workspace_min = np.asarray(workspace_min, dtype=np.float64).copy()
        self.workspace_max = np.asarray(workspace_max, dtype=np.float64).copy()
        if self.workspace_min.shape != (3,) or self.workspace_max.shape != (3,):
            raise ValueError("workspace bounds must contain three values")
        if not np.all(np.isfinite(self.workspace_min)) or not np.all(
            np.isfinite(self.workspace_max)
        ):
            raise ValueError("workspace bounds must be finite")
        if np.any(self.workspace_min >= self.workspace_max):
            raise ValueError("workspace minimum must be below maximum")
        self.required_motion_mode = int(required_motion_mode)
        self.left_session_start = pose_values(tf, LEFT_FRAME)
        self.right_session_start = pose_values(tf, RIGHT_FRAME)
        self.desired = self.right_session_start.copy()
        self.translation_compensation = np.asarray(
            SEED_TRANSLATION_COMPENSATION
            if initial_translation_compensation is None
            else initial_translation_compensation,
            dtype=np.float64,
        ).copy()
        if self.translation_compensation.shape != (3,):
            raise ValueError("initial translation compensation must have 3 values")
        if not np.all(np.isfinite(self.translation_compensation)):
            raise ValueError("initial translation compensation must be finite")
        translation_compensation_norm = float(
            np.linalg.norm(self.translation_compensation)
        )
        if (
            translation_compensation_norm
            > MAX_TOTAL_TRANSLATION_COMPENSATION_M + 1e-9
        ):
            raise ValueError("initial translation compensation exceeds limit")
        # A compensation produced by ``clamp`` can round a few ulps above the
        # same bound when serialized through JSON and parsed by a restarted
        # controller.  Project only that numerical excess back to the exact
        # limit; materially out-of-range seeds are still rejected above.
        if translation_compensation_norm > MAX_TOTAL_TRANSLATION_COMPENSATION_M:
            self.translation_compensation *= (
                MAX_TOTAL_TRANSLATION_COMPENSATION_M
                / translation_compensation_norm
            )
        self.rotation_compensation = normalize(
            SEED_ROTATION_COMPENSATION
            if initial_rotation_compensation is None
            else initial_rotation_compensation
        )
        identity = np.asarray([0.0, 0.0, 0.0, 1.0])
        qdot = abs(float(np.dot(identity, self.rotation_compensation)))
        seed_rotation_angle = 2.0 * math.acos(max(-1.0, min(1.0, qdot)))
        if seed_rotation_angle > MAX_TOTAL_ROTATION_COMPENSATION_RAD:
            raise ValueError("initial rotation compensation exceeds limit")
        self.tick_count = 0
        self.calibrated = False
        self.maxima: dict[str, float] = {}
        self.next_tick = time.monotonic()

    def _command(self, desired: np.ndarray):
        command = desired.copy()
        command[:3] += self.translation_compensation
        command[3:7] = qmul(self.rotation_compensation, normalize(desired[3:7]))
        if np.any(command[:3] < self.workspace_min - 0.01) or np.any(
            command[:3] > self.workspace_max + 0.01
        ):
            raise RuntimeError("compensated command outside extended workspace")
        return command

    def _adapt(self, desired: np.ndarray, live: np.ndarray):
        tstep = clamp(
            TRANSLATION_GAIN * (desired[:3] - live[:3]),
            MAX_TRANSLATION_CORRECTION_PER_TICK_M,
        )
        self.translation_compensation = clamp(
            self.translation_compensation + tstep,
            MAX_TOTAL_TRANSLATION_COMPENSATION_M,
        )
        qerror = qmul(normalize(desired[3:7]), qinv(live[3:7]))
        rstep = clamp(
            ROTATION_GAIN * q_to_rv(qerror),
            MAX_ROTATION_CORRECTION_PER_TICK_RAD,
        )
        self.rotation_compensation = qmul(
            rv_to_q(rstep), self.rotation_compensation
        )
        identity = np.asarray([0.0, 0.0, 0.0, 1.0])
        qdot = abs(float(np.dot(identity, self.rotation_compensation)))
        comp_angle = 2.0 * math.acos(max(-1.0, min(1.0, qdot)))
        if comp_angle > MAX_TOTAL_ROTATION_COMPENSATION_RAD:
            raise RuntimeError("rotation compensation exceeded limit")

    def _metrics(self, desired: np.ndarray, left: np.ndarray, right: np.ndarray):
        return TickMetrics(
            left_translation_m=translation_distance(self.left_session_start, left),
            left_rotation_rad=quaternion_angle(self.left_session_start, left),
            right_session_translation_m=translation_distance(
                self.right_session_start, right
            ),
            right_session_rotation_rad=quaternion_angle(
                self.right_session_start, right
            ),
            target_position_error_m=translation_distance(desired, right),
            target_rotation_error_rad=quaternion_angle(desired, right),
        )

    def _guard(self, desired: np.ndarray, metrics: TickMetrics):
        if not np.isfinite(desired).all():
            raise RuntimeError("non-finite desired target")

    def tick(self, desired: np.ndarray):
        desired = np.asarray(desired, dtype=np.float64).copy()
        desired[3:7] = normalize(desired[3:7])
        left = pose_values(self.tf, LEFT_FRAME)
        right = pose_values(self.tf, RIGHT_FRAME)
        metrics = self._metrics(desired, left, right)
        self._guard(desired, metrics)
        # Keep the original motion algorithm, but do not publish another
        # target after the official controller has reported a fault.
        require_safe_state(snapshot(self.robot, self.tf), self.required_motion_mode)
        self._adapt(desired, right)
        send(self.robot, self._command(desired))
        for key, value in metrics.as_dict().items():
            self.maxima[key] = max(self.maxima.get(key, 0.0), value)
        self.desired = desired
        self.tick_count += 1
        if self.tick_count % MODEL_TICKS == 0:
            require_safe_state(
                snapshot(self.robot, self.tf), self.required_motion_mode
            )
        self.next_tick += CONTROL_PERIOD_S
        wait = self.next_tick - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        elif wait < -CONTROL_PERIOD_S:
            raise RuntimeError(f"50 Hz control deadline missed by {-wait:.6f}s")
        return metrics

    def hold_ticks(self, ticks: int):
        last = None
        for _ in range(ticks):
            last = self.tick(self.desired)
        return last

    def calibrate(self, duration_s: float = 2.0):
        anchor = pose_values(self.tf, RIGHT_FRAME)
        self.desired = anchor.copy()
        max_transient_m = 0.0
        max_transient_rad = 0.0
        for _ in range(int(round(duration_s * CONTROL_HZ))):
            metrics = self.tick(anchor)
            max_transient_m = max(max_transient_m, metrics.target_position_error_m)
            max_transient_rad = max(max_transient_rad, metrics.target_rotation_error_rad)
            if max_transient_m > CALIBRATION_MAX_TRANSIENT_M:
                raise RuntimeError("calibration translation transient exceeded limit")
            if max_transient_rad > CALIBRATION_MAX_TRANSIENT_RAD:
                raise RuntimeError("calibration rotation transient exceeded limit")
        final_pose = pose_values(self.tf, RIGHT_FRAME)
        final_position = translation_distance(anchor, final_pose)
        final_rotation = quaternion_angle(anchor, final_pose)
        if final_position > CONVERGENCE_POSITION_M:
            raise RuntimeError("calibration position convergence failed")
        if final_rotation > CONVERGENCE_ROTATION_RAD:
            raise RuntimeError("calibration rotation convergence failed")
        self.calibrated = True
        result = {
            "anchor_pose": anchor.tolist(),
            "final_pose": final_pose.tolist(),
            "final_position_error_m": final_position,
            "final_rotation_error_rad": final_rotation,
            "max_transient_m": max_transient_m,
            "max_transient_rad": max_transient_rad,
            "translation_compensation_m": self.translation_compensation.tolist(),
            "rotation_compensation_xyzw": self.rotation_compensation.tolist(),
        }
        emit("calibration_complete", **result)
        return result

    def move_to(self, target: np.ndarray, duration_s: float):
        if not self.calibrated:
            raise RuntimeError("controller is not calibrated")
        target = np.asarray(target, dtype=np.float64).copy()
        target[3:7] = normalize(target[3:7])
        start = self.desired.copy()
        distance = translation_distance(start, target)
        rotation = quaternion_angle(start, target)
        if distance > MAX_TARGET_STEP_M:
            raise RuntimeError("target translation step exceeds limit")
        if rotation > MAX_TARGET_ROTATION_RAD:
            raise RuntimeError("target rotation step exceeds limit")
        ticks = max(1, int(round(duration_s * CONTROL_HZ)))
        emitted = []
        for index in range(ticks):
            alpha = float(index + 1) / ticks
            desired = np.empty(7)
            desired[:3] = start[:3] * (1.0 - alpha) + target[:3] * alpha
            desired[3:7] = interpolate_quaternion(start[3:7], target[3:7], alpha)
            emitted.append(desired.tolist())
            self.tick(desired)
        return {
            "start_pose": start.tolist(),
            "target_pose": target.tolist(),
            "duration_s": duration_s,
            "ticks": ticks,
            "emitted_50hz_targets": emitted,
        }

    def verify_target(self, target: np.ndarray, hold_s: float = 0.5):
        self.desired = np.asarray(target, dtype=np.float64).copy()
        self.hold_ticks(int(round(hold_s * CONTROL_HZ)))
        live = pose_values(self.tf, RIGHT_FRAME)
        result = {
            "target_pose": self.desired.tolist(),
            "live_pose": live.tolist(),
            "position_error_m": translation_distance(self.desired, live),
            "rotation_error_rad": quaternion_angle(self.desired, live),
        }
        result["passed"] = (
            result["position_error_m"] <= CONVERGENCE_POSITION_M
            and result["rotation_error_rad"] <= CONVERGENCE_ROTATION_RAD
        )
        emit("target_verification", **result)
        if not result["passed"]:
            raise RuntimeError("target convergence verification failed")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-validation", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if args.execute_validation and args.confirm != CONFIRMATION:
        parser.error(f"--execute-validation requires --confirm {CONFIRMATION}")
    if not args.execute_validation and args.confirm:
        parser.error("--confirm is only valid with --execute-validation")
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise RuntimeError("gdk_init failed")
    try:
        robot, tf = agibot_gdk.Robot(), agibot_gdk.TF()
        time.sleep(2.0)
        before = snapshot(robot, tf)
        require_safe_state(before)
        time.sleep(0.30)
        stable = snapshot(robot, tf)
        require_safe_state(stable)
        if translation_distance(before["right"], stable["right"]) > 0.0002:
            raise RuntimeError("robot is not stationary")
        competitors = competing_controllers()
        emit(
            "preflight",
            execute=args.execute_validation,
            mode=stable["mode"],
            whole=stable["whole"],
            start_pose=stable["right"].round(9).tolist(),
            competing_controllers=competitors,
            control_hz=CONTROL_HZ,
            model_hz=1.0 / MODEL_PERIOD_S,
            interpolation_ticks=MODEL_TICKS,
            sequence=[
                "calibrate_and_zero_hold",
                "0p5mm_slow_roundtrip",
                "1p0mm_slow_roundtrip",
                "0p5mm_10hz_roundtrip",
            ],
        )
        if not args.execute_validation:
            return 0
        if competitors:
            raise RuntimeError(f"competing controllers found: {competitors}")
        for remaining in range(5, 0, -1):
            emit("physical_countdown", remaining_s=remaining)
            time.sleep(1.0)

        controller = PersistentRightArmController(robot, tf)
        calibration = controller.calibrate(2.0)
        origin = controller.desired.copy()
        zero_hold = controller.verify_target(origin, 0.5)
        tests = []
        for name, distance_m, duration_s in (
            ("0p5mm_slow", 0.0005, 0.5),
            ("1p0mm_slow", 0.0010, 0.5),
            ("0p5mm_10hz", 0.0005, MODEL_PERIOD_S),
        ):
            target = origin.copy()
            target[0] += distance_m
            emit(
                "validation_segment_start",
                name=name,
                distance_m=distance_m,
                duration_s=duration_s,
            )
            outbound_command = controller.move_to(target, duration_s)
            outbound = controller.verify_target(target, 0.5)
            return_command = controller.move_to(origin, duration_s)
            returned = controller.verify_target(origin, 0.5)
            tests.append(
                {
                    "name": name,
                    "outbound_command": outbound_command,
                    "outbound": outbound,
                    "return_command": return_command,
                    "returned": returned,
                }
            )
            emit("validation_segment_complete", name=name)
        final = snapshot(robot, tf)
        require_safe_state(final)
        result = {
            "passed": True,
            "calibration": calibration,
            "zero_hold": zero_hold,
            "tests": tests,
            "final_pose": final["right"].tolist(),
            "final_origin_position_error_m": translation_distance(
                origin, final["right"]
            ),
            "final_origin_rotation_error_rad": quaternion_angle(
                origin, final["right"]
            ),
            "tick_count": controller.tick_count,
            "max": controller.maxima,
            "translation_compensation_m": controller.translation_compensation.tolist(),
            "rotation_compensation_xyzw": controller.rotation_compensation.tolist(),
            "mode": final["mode"],
            "whole": final["whole"],
        }
        emit("result", **result)
        return 0
    finally:
        agibot_gdk.gdk_release()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        emit("fatal", error_type=type(error).__name__, message=str(error))
        raise
