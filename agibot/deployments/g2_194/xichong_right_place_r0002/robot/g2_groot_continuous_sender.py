"""Candidate continuous sender: arm EEF and tool in one official GDK request.

This adapter uses ``Robot.trajectory_tracking_control`` only. The installed
GDK 3.3.8 Python binding accepts ``right_arm/ABS_POSE`` (XYZ + quaternion XYZW)
and ``right_effector/ABS_JOINT`` in the same trajectory action. The binding
ignores ``infer_timestamp`` and ``robot_states``; neither is used for feedback.

The caller owns the GDK session, feedback checks, EEF interpolation, and 50 Hz
schedule. A call publishes one 20 ms point; it never sleeps or retries. Tool
targets interpolate over the same five ticks as each 10 Hz EEF waypoint. Idle
calls hold the most recently commanded tool target. No construction or target
update publishes a command. Unit tests verify the request mapping, not actual
hardware compatibility or physical tracking.

Before the first point, initialize the official IK reference orientations from
the live ``arm_l_link3`` / ``arm_r_link3`` TF frames. These are redundancy
references, not EEF target orientations. The installed adapter otherwise uses
hard-coded startup references. The official setter only updates Robot's local
reference poses; it does not publish arm or gripper commands.

This module is independent of the legacy backend. Optional before_publish()
checks run immediately before each SDK call, never at construction or reference
initialization. Monotonic timing measures the Python SDK call, not DDS receipt
or actuator response. Reference timestamps are read only through the official
TF.get_latest_timestamp when available; that separate read is not an atomic
timestamp of the quaternion sample. No clock value is substituted for a missing
source timestamp. None of these diagnostics establish hardware acceptance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import operator
import threading
import time
from typing import Any


CONTROL_PERIOD_S = 0.02
MODEL_TICKS = 5
OPEN_GRIPPER = -0.785
CLOSED_GRIPPER = 0.0
GRIPPER_FLOAT_TOLERANCE = 1e-4
QUATERNION_NORM_TOLERANCE = 1e-3
REFERENCE_BASE_FRAME = "base_link"
REFERENCE_FRAMES = ("arm_l_link3", "arm_r_link3")


def _native_gripper(value: Any) -> float:
    """Keep native radians, correcting only endpoint floating-point overshoot."""
    if isinstance(value, bool):
        raise ValueError("gripper target must be a finite native-radian number")
    try:
        target = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("gripper target must be a finite native-radian number") from exc
    if not math.isfinite(target):
        raise ValueError("gripper target must be finite")
    if not OPEN_GRIPPER - GRIPPER_FLOAT_TOLERANCE <= target <= GRIPPER_FLOAT_TOLERANCE:
        raise ValueError(f"gripper target must be in [{OPEN_GRIPPER}, {CLOSED_GRIPPER}]")
    return min(CLOSED_GRIPPER, max(OPEN_GRIPPER, target))


def _pose_xyzw(pose: Sequence[float]) -> list[float]:
    try:
        if len(pose) != 7:
            raise ValueError("pose must contain 7 values: XYZ + quaternion XYZW")
        if any(isinstance(value, bool) for value in pose):
            raise ValueError("pose values must be finite numbers")
        values = [float(value) for value in pose]
    except (TypeError, OverflowError) as exc:
        raise ValueError("pose must contain 7 finite values: XYZ + quaternion XYZW") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pose values must be finite")
    norm = math.hypot(*values[3:])
    if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise ValueError("pose quaternion must be unit length (XYZW)")
    return values[:3] + [value / norm for value in values[3:]]


def _reference_quaternion(tf: Any, frame: str) -> tuple[float, ...]:
    """Read the named official TF orientation without substituting an EEF pose."""
    try:
        rotation = tf.get_tf_from_base_link(frame).rotation
        raw = [rotation.x, rotation.y, rotation.z, rotation.w]
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"missing reference TF quaternion for {frame}") from exc
    try:
        values = _pose_xyzw([0.0, 0.0, 0.0, *raw])[3:]
    except ValueError as exc:
        raise ValueError(f"invalid reference TF quaternion for {frame}: {exc}") from exc
    return tuple(values)


class NativeTrajectorySender:
    """Callable single-request arm/tool sender; SDK failure latches the session.

    ``sent_target`` is the last *attempted* tool request, not measured feedback.
    ``accepted_target`` means the SDK returned zero, not that the tool reached
    it. Following an SDK exception the outcome is unknown, so this instance
    refuses further dispatch rather than silently repeating a physical action.
    """

    def __init__(
        self,
        initial_gripper: float,
        *,
        before_publish: Callable[[], None] | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ):
        initial = _native_gripper(initial_gripper)
        if before_publish is not None and not callable(before_publish):
            raise TypeError("before_publish must be callable or None")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self._before_publish = before_publish
        self._clock_ns = clock_ns
        self._last_call_started_monotonic_ns: int | None = None
        self._last_call_finished_monotonic_ns: int | None = None
        self._last_call_interval_ns: int | None = None
        self._lock = threading.RLock()
        self._initial_gripper = initial
        self._current_target = initial
        self._requested_target = initial
        self._ramp_start = initial
        self._ramp_ticks = 0
        self._ramp_index = 0
        self._sent_target: float | None = None
        self._accepted_target: float | None = None
        self._call_count = 0
        self._success_count = 0
        self._last_result: Any = None
        self._fault: str | None = None
        self._reference_robot: Any = None
        self._reference_quaternions: dict[str, tuple[float, ...]] = {}
        self._reference_setter_result: Any = None
        self._reference_timestamps_ns: dict[str, int | None] = {}
        self._reference_timestamp_errors: dict[str, str] = {}
        self._reference_timestamp_source: str | None = None

    def _require_healthy(self) -> None:
        if self._fault is not None:
            raise RuntimeError(f"native trajectory sender fault latched: {self._fault}")

    def _reference_snapshot(self) -> dict[str, Any]:
        return {
            "initialized": self._reference_robot is not None,
            "base_frame": REFERENCE_BASE_FRAME,
            "source": "live_tf_at_startup",
            "reference_quaternions_xyzw": {
                frame: list(values) for frame, values in self._reference_quaternions.items()
            },
            "setter_result": self._reference_setter_result,
            "reference_timestamps_ns": dict(self._reference_timestamps_ns),
            "reference_timestamp_source": self._reference_timestamp_source,
            "reference_timestamp_errors": dict(self._reference_timestamp_errors),
        }

    def initialize_references(self, robot: Any, tf: Any) -> dict[str, Any]:
        """Freeze live link3 IK references before any trajectory is published.

        Both TF orientations are read and validated before the official local
        setter is called. Repeating setup on the same Robot before publishing
        is idempotent: it does not change those references. After publishing,
        or for a different Robot, a new sender is required.
        """
        with self._lock:
            self._require_healthy()
            if self._call_count:
                raise RuntimeError("cannot initialize references after trajectory publication")
            if self._reference_robot is not None:
                if robot is not self._reference_robot:
                    raise RuntimeError("native trajectory references belong to a different Robot")
                return self._reference_snapshot()
            references = {}
            timestamps: dict[str, int | None] = {}
            timestamp_errors = {}
            timestamp_reader = getattr(tf, "get_latest_timestamp", None)
            for frame in REFERENCE_FRAMES:
                references[frame] = _reference_quaternion(tf, frame)
                timestamps[frame] = None
                if callable(timestamp_reader):
                    try:
                        raw_timestamp = timestamp_reader(frame)
                        if isinstance(raw_timestamp, bool):
                            raise ValueError("boolean source timestamp")
                        timestamp_ns = operator.index(raw_timestamp)
                        if timestamp_ns <= 0:
                            raise ValueError("source timestamp must be positive")
                        timestamps[frame] = timestamp_ns
                    except Exception as exc:
                        # Optional diagnostics must not invent freshness or
                        # silently substitute the local acquisition time.
                        timestamp_errors[frame] = f"{type(exc).__name__}: {exc}"
            self._reference_timestamps_ns = timestamps
            self._reference_timestamp_errors = timestamp_errors
            self._reference_timestamp_source = (
                "TF.get_latest_timestamp" if callable(timestamp_reader) else None
            )
            args = [value for frame in REFERENCE_FRAMES for value in references[frame]]
            try:
                result = robot.set_reference_frame_poses(*args)
                self._reference_setter_result = result
                if result != 0:
                    raise RuntimeError(f"set_reference_frame_poses returned {result!r}")
            except Exception as exc:
                self._fault = f"{type(exc).__name__}: {exc}"
                raise
            self._reference_quaternions = references
            self._reference_robot = robot
            return self._reference_snapshot()

    def set_target(self, target: float, ticks: int = MODEL_TICKS) -> None:
        """Stage the next tool waypoint, retaining all intermediate openings."""
        canonical_target = _native_gripper(target)
        if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 1:
            raise ValueError("ticks must be a positive integer")
        with self._lock:
            self._require_healthy()
            self._requested_target = canonical_target
            self._ramp_start = self._current_target
            self._ramp_ticks = ticks
            self._ramp_index = 0

    def __call__(
        self, robot: Any, pose: Sequence[float], *, latest_start_monotonic_ns: int | None = None
    ) -> int:
        """Send one combined action; caller must call once per 20 ms tick."""
        pose_values = _pose_xyzw(pose)
        with self._lock:
            self._require_healthy()
            if self._reference_robot is None:
                raise RuntimeError(
                    "initialize_references(robot, tf) is required before publication"
                )
            if robot is not self._reference_robot:
                raise RuntimeError("native trajectory references belong to a different Robot")
            next_index = min(self._ramp_index + 1, self._ramp_ticks)
            if next_index == self._ramp_ticks:
                next_target = self._requested_target
            else:
                alpha = next_index / self._ramp_ticks
                next_target = self._ramp_start + alpha * (self._requested_target - self._ramp_start)
            action = {
                "right_arm": {"control_type": "ABS_POSE", "action_data": pose_values},
                "right_effector": {
                    "control_type": "ABS_JOINT",
                    "action_data": [next_target],
                },
            }
            if self._before_publish is not None:
                try:
                    self._before_publish()
                except Exception as exc:
                    self._fault = f"before_publish: {type(exc).__name__}: {exc}"
                    raise
            # The callback does not count as an attempted SDK publication.
            # Capture the start after it, so its work is not reported as SDK latency.
            started_ns = self._clock_ns()
            if latest_start_monotonic_ns is not None and started_ns > latest_start_monotonic_ns:
                self._fault = "publication deadline missed after feedback callback"
                raise RuntimeError(self._fault)
            previous_started_ns = self._last_call_started_monotonic_ns
            self._last_call_interval_ns = (
                None if previous_started_ns is None else started_ns - previous_started_ns
            )
            self._last_call_started_monotonic_ns = started_ns
            self._last_call_finished_monotonic_ns = None
            self._call_count += 1
            self._sent_target = next_target
            self._last_result = None
            try:
                result = robot.trajectory_tracking_control(
                    0,
                    {},
                    [action],
                    robot_link="base_link",
                    trajectory_reference_time=CONTROL_PERIOD_S,
                )
                self._last_result = result
                if result != 0:
                    raise RuntimeError(f"trajectory_tracking_control returned {result!r}")
            except Exception as exc:
                self._fault = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self._last_call_finished_monotonic_ns = self._clock_ns()
            self._current_target = next_target
            self._accepted_target = next_target
            self._ramp_index = next_index
            self._success_count += 1
            return 0

    def snapshot(self) -> dict[str, Any]:
        """Return command-level diagnostics; none is physical gripper feedback."""
        with self._lock:
            return {
                "backend": "gdk_trajectory_tracking_control",
                "initial_gripper": self._initial_gripper,
                "requested_target": self._requested_target,
                "sent_target": self._sent_target,
                "accepted_target": self._accepted_target,
                "current_target": self._current_target,
                "remaining_ticks": self._ramp_ticks - self._ramp_index,
                "control_period_s": CONTROL_PERIOD_S,
                "last_call_started_monotonic_ns": self._last_call_started_monotonic_ns,
                "last_call_finished_monotonic_ns": self._last_call_finished_monotonic_ns,
                "last_call_interval_ns": self._last_call_interval_ns,
                "last_call_duration_ns": (
                    None
                    if self._last_call_started_monotonic_ns is None
                    or self._last_call_finished_monotonic_ns is None
                    else self._last_call_finished_monotonic_ns
                    - self._last_call_started_monotonic_ns
                ),
                "call_count": self._call_count,
                "success_count": self._success_count,
                "last_result": self._last_result,
                "fault": self._fault,
                "reference_initialization": self._reference_snapshot(),
            }
