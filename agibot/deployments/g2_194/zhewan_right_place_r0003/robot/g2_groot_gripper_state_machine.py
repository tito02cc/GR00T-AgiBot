#!/usr/bin/env python3
"""Pure fail-closed state machine for asynchronous G2 omnipicker commands.

The GR00T and GDK command conventions are identical: -0.785 is fully open
and 0.0 is fully closed. G2 releases have exposed feedback either as an
approximately 0..120 travel value or directly in native command radians.
The encoding is explicit because zero has opposite meanings in those forms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


OPEN_COMMAND = -0.785
CLOSED_COMMAND = 0.0
RAW_CLOSED = 120.0
FEEDBACK_RAW_0_120 = "raw_0_120"
FEEDBACK_NATIVE_RADIANS = "native_radians"
FEEDBACK_ENCODINGS = (FEEDBACK_RAW_0_120, FEEDBACK_NATIVE_RADIANS)
CLOSURE_THRESHOLD = -0.55
TRANSITIONAL_STATUS = 1
BLOCKED_STATUS = 2
POST_CONTACT_STATUS = 3
SETTLED_STATUS = 0


@dataclass(frozen=True)
class GripperObservation:
    monotonic_s: float
    raw_position: float
    motor_status: int
    motor_err_code: int
    whole_end_error: int
    effort: float = 0.0


@dataclass(frozen=True)
class GripperDecision:
    command: float | None
    reason: str
    desired: float
    active_target: float | None
    recovery_requested: bool
    fault: str | None


def command_to_raw(command: float) -> float:
    value = float(command)
    if not math.isfinite(value) or not OPEN_COMMAND <= value <= CLOSED_COMMAND:
        raise ValueError(f"invalid omnipicker command: {value}")
    return RAW_CLOSED * (value - OPEN_COMMAND) / (CLOSED_COMMAND - OPEN_COMMAND)


def raw_to_command(raw_position: float) -> float:
    value = float(raw_position)
    if not math.isfinite(value):
        raise ValueError("non-finite omnipicker feedback")
    fraction = min(1.0, max(0.0, value / RAW_CLOSED))
    return OPEN_COMMAND * (1.0 - fraction)


def feedback_to_command(position: float, encoding: str) -> float:
    """Convert version-specific feedback into the GR00T/GDK command domain."""
    value = float(position)
    if not math.isfinite(value):
        raise ValueError("non-finite omnipicker feedback")
    if encoding == FEEDBACK_RAW_0_120:
        return raw_to_command(value)
    if encoding == FEEDBACK_NATIVE_RADIANS:
        if not OPEN_COMMAND - 0.02 <= value <= CLOSED_COMMAND + 0.02:
            raise ValueError("native omnipicker feedback is out of range")
        return min(CLOSED_COMMAND, max(OPEN_COMMAND, value))
    raise ValueError(f"unknown omnipicker feedback encoding: {encoding}")


class GripperStateMachine:
    """Serialize last-desired targets onto one asynchronous GDK actuator."""

    def __init__(
        self,
        *,
        closure_enabled: bool = False,
        maximum_command_step: float = 0.2,
        command_deadband: float = 0.015,
        settle_tolerance: float = 0.01,
        motion_timeout_s: float = 1.5,
        contact_detection_enabled: bool = False,
        contact_target_threshold: float = -0.1,
        contact_shortfall: float = 0.02,
        contact_min_abs_effort: float = 2.0,
        contact_confirm_s: float = 0.15,
        maximum_abs_effort: float = 25.0,
        feedback_encoding: str = FEEDBACK_RAW_0_120,
    ) -> None:
        for value, name in (
            (maximum_command_step, "maximum_command_step"),
            (command_deadband, "command_deadband"),
            (settle_tolerance, "settle_tolerance"),
            (motion_timeout_s, "motion_timeout_s"),
            (contact_shortfall, "contact_shortfall"),
            (contact_min_abs_effort, "contact_min_abs_effort"),
            (contact_confirm_s, "contact_confirm_s"),
            (maximum_abs_effort, "maximum_abs_effort"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if maximum_command_step > CLOSED_COMMAND - OPEN_COMMAND:
            raise ValueError("maximum_command_step exceeds actuator range")
        self.closure_enabled = bool(closure_enabled)
        self.maximum_command_step = float(maximum_command_step)
        self.command_deadband = float(command_deadband)
        self.settle_tolerance = float(settle_tolerance)
        self.motion_timeout_s = float(motion_timeout_s)
        self.contact_detection_enabled = bool(contact_detection_enabled)
        if self.contact_detection_enabled and not self.closure_enabled:
            raise ValueError("contact detection requires closure_enabled")
        if not OPEN_COMMAND <= contact_target_threshold <= CLOSED_COMMAND:
            raise ValueError("invalid contact_target_threshold")
        self.contact_target_threshold = float(contact_target_threshold)
        self.contact_shortfall = float(contact_shortfall)
        self.contact_min_abs_effort = float(contact_min_abs_effort)
        self.contact_confirm_s = float(contact_confirm_s)
        self.maximum_abs_effort = float(maximum_abs_effort)
        if feedback_encoding not in FEEDBACK_ENCODINGS:
            raise ValueError(f"invalid feedback_encoding: {feedback_encoding}")
        self.feedback_encoding = feedback_encoding
        self.desired = OPEN_COMMAND
        self.last_completed = OPEN_COMMAND
        self.active_target: float | None = None
        self.active_direction: int | None = None
        self.active_since_s: float | None = None
        self.active_start_feedback: float | None = None
        self.recovery_requested = False
        self.recovery_reason: str | None = None
        self.fault: str | None = None
        self.command_count = 0
        self.active_saw_transition = False
        self.contact_candidate_since_s: float | None = None
        self.contact_latched = False
        self.contact_feedback: float | None = None
        self.contact_effort: float | None = None
        self.last_closure_outcome: str | None = None

    def update_desired(self, command: float) -> None:
        value = float(command)
        if not math.isfinite(value) or not OPEN_COMMAND <= value <= CLOSED_COMMAND:
            raise ValueError(f"gripper target outside [{OPEN_COMMAND}, 0]: {value}")
        if not self.closure_enabled and value >= CLOSURE_THRESHOLD:
            raise ValueError("gripper closure is not enabled")
        if self.recovery_requested or self.fault:
            if value > OPEN_COMMAND + self.command_deadband:
                raise RuntimeError("gripper is latched in fail-open recovery")
        self.desired = value

    def request_safe_open(self, reason: str) -> None:
        text = str(reason).strip()
        if not text:
            raise ValueError("safe-open reason is required")
        self.desired = OPEN_COMMAND
        self.recovery_requested = True
        self.recovery_reason = text

    def _decision(self, command: float | None, reason: str) -> GripperDecision:
        return GripperDecision(
            command=command,
            reason=reason,
            desired=self.desired,
            active_target=self.active_target,
            recovery_requested=self.recovery_requested,
            fault=self.fault,
        )

    def _validate_observation(self, observation: GripperObservation) -> str | None:
        if not math.isfinite(observation.monotonic_s):
            return "non_finite_monotonic_time"
        if not math.isfinite(observation.raw_position):
            return "non_finite_feedback"
        if not math.isfinite(observation.effort):
            return "non_finite_effort"
        try:
            feedback_to_command(observation.raw_position, self.feedback_encoding)
        except ValueError:
            return "feedback_out_of_range"
        if observation.motor_status not in (
            SETTLED_STATUS,
            TRANSITIONAL_STATUS,
            BLOCKED_STATUS,
            POST_CONTACT_STATUS,
        ):
            return f"unknown_motor_status_{observation.motor_status}"
        if observation.motor_err_code != 0:
            return f"motor_error_{observation.motor_err_code}"
        if observation.whole_end_error != 0:
            return f"whole_end_error_{observation.whole_end_error}"
        if abs(observation.effort) > self.maximum_abs_effort:
            return "effort_limit_exceeded"
        return None

    def _start_command(
        self, target: float, now_s: float, reason: str, feedback: float
    ) -> GripperDecision:
        self.active_target = float(target)
        self.active_direction = 1 if target >= feedback else -1
        self.active_since_s = float(now_s)
        self.active_start_feedback = float(feedback)
        self.active_saw_transition = False
        self.contact_candidate_since_s = None
        self.command_count += 1
        return self._decision(self.active_target, reason)

    def _next_bounded_target(self) -> float | None:
        delta = self.desired - self.last_completed
        if abs(delta) <= self.command_deadband:
            return None
        bounded = max(-self.maximum_command_step, min(self.maximum_command_step, delta))
        return max(OPEN_COMMAND, min(CLOSED_COMMAND, self.last_completed + bounded))

    def step(self, observation: GripperObservation) -> GripperDecision:
        health_fault = self._validate_observation(observation)
        if health_fault:
            self.fault = health_fault
            return self._decision(None, "health_fault_no_command")

        feedback = feedback_to_command(
            observation.raw_position, self.feedback_encoding
        )
        if self.fault:
            return self._decision(None, "fault_latched_no_command")

        if self.recovery_requested and self.active_target is None:
            if (
                observation.motor_status == SETTLED_STATUS
                and abs(feedback - OPEN_COMMAND) <= self.settle_tolerance
            ):
                self.last_completed = OPEN_COMMAND
                return self._decision(None, "already_open_recovery_latched")
            return self._start_command(
                OPEN_COMMAND,
                observation.monotonic_s,
                f"safe_open:{self.recovery_reason}",
                feedback,
            )

        if self.recovery_requested and self.active_target != OPEN_COMMAND:
            return self._start_command(
                OPEN_COMMAND,
                observation.monotonic_s,
                f"safe_open:{self.recovery_reason}",
                feedback,
            )

        if self.active_target is not None:
            reached_target = abs(feedback - self.active_target) <= self.settle_tolerance
            if reached_target and observation.motor_status in (
                BLOCKED_STATUS, POST_CONTACT_STATUS
            ):
                # Position feedback is authoritative. Some GDK releases keep
                # status=2 after an otherwise completed tool command.
                self.last_completed = feedback
                if self.active_target >= self.contact_target_threshold:
                    self.last_closure_outcome = "target_reached_no_contact_evidence"
                completed_open_recovery = bool(
                    self.recovery_requested and self.active_target == OPEN_COMMAND
                )
                self.active_target = None
                self.active_direction = None
                self.active_since_s = None
                self.active_start_feedback = None
                if completed_open_recovery:
                    return self._decision(None, "safe_open_completed_latched")
                self.last_closure_outcome = "hardware_blocked_or_holding"
                reason = (
                    "hardware_post_contact_hold"
                    if observation.motor_status == POST_CONTACT_STATUS
                    else "hardware_blocked_or_holding"
                )
                return self._decision(None, reason)
            elif observation.motor_status in (BLOCKED_STATUS, POST_CONTACT_STATUS):
                if (
                    self.active_direction == 1
                    and self.active_start_feedback is not None
                    and feedback - self.active_start_feedback > self.settle_tolerance
                ):
                    self.last_completed = feedback
                    self.last_closure_outcome = "hardware_blocked_or_holding"
                    self.contact_latched = True
                    self.contact_feedback = feedback
                    self.contact_effort = observation.effort
                    self.active_target = None
                    self.active_direction = None
                    self.active_since_s = None
                    self.active_start_feedback = None
                    reason = (
                        "hardware_post_contact_hold"
                        if observation.motor_status == POST_CONTACT_STATUS
                        else "hardware_blocked_or_holding"
                    )
                    return self._decision(None, reason)
                # During opening, status=2 may simply be the previous holding
                # state. Keep waiting for actual position feedback to change.
            if observation.motor_status == TRANSITIONAL_STATUS:
                self.active_saw_transition = True
            settled = bool(
                observation.motor_status == SETTLED_STATUS
                and reached_target
            )
            if settled:
                self.last_completed = feedback
                if self.active_target >= self.contact_target_threshold:
                    self.last_closure_outcome = "target_reached_no_contact_evidence"
                completed_open_recovery = bool(
                    self.recovery_requested and self.active_target == OPEN_COMMAND
                )
                self.active_target = None
                self.active_direction = None
                self.active_since_s = None
                self.active_start_feedback = None
                if completed_open_recovery:
                    return self._decision(None, "safe_open_completed_latched")
            else:
                assert self.active_since_s is not None
                elapsed = observation.monotonic_s - self.active_since_s
                if elapsed < 0:
                    self.fault = "monotonic_time_regressed"
                    return self._decision(None, "clock_fault_no_command")
                if elapsed >= self.motion_timeout_s:
                    self.fault = (
                        "opening_motion_timeout"
                        if self.active_direction == -1
                        else "closing_motion_timeout"
                    )
                    self.last_completed = feedback
                    if self.active_direction == 1:
                        self.last_closure_outcome = "closing_motion_timeout"
                    self.active_target = None
                    self.active_direction = None
                    self.active_since_s = None
                    self.active_start_feedback = None
                    return self._decision(None, f"{self.fault}_fault")
                contact_shortfall = self.active_target - feedback
                contact_candidate = bool(
                    self.contact_detection_enabled
                    and self.active_direction == 1
                    and self.active_start_feedback is not None
                    and feedback - self.active_start_feedback > self.settle_tolerance
                    and self.active_target >= self.contact_target_threshold
                    and self.active_saw_transition
                    and observation.motor_status == SETTLED_STATUS
                    and contact_shortfall >= self.contact_shortfall
                    and abs(observation.effort) >= self.contact_min_abs_effort
                )
                if contact_candidate:
                    if self.contact_candidate_since_s is None:
                        self.contact_candidate_since_s = observation.monotonic_s
                    elif (
                        observation.monotonic_s - self.contact_candidate_since_s
                        >= self.contact_confirm_s
                    ):
                        self.contact_latched = True
                        self.contact_feedback = feedback
                        self.contact_effort = observation.effort
                        self.last_completed = feedback
                        self.last_closure_outcome = "position_shortfall_contact_candidate"
                        self.active_target = None
                        self.active_direction = None
                        self.active_since_s = None
                        self.active_start_feedback = None
                        return self._decision(None, "contact_candidate_latched")
                else:
                    self.contact_candidate_since_s = None
                return self._decision(None, "motion_in_progress")

        if self.recovery_requested:
            return self._decision(None, "safe_open_latched")
        if self.fault:
            return self._decision(None, "fault_latched_no_command")
        if self.contact_latched:
            assert self.contact_feedback is not None
            if self.desired >= self.contact_feedback - self.command_deadband:
                return self._decision(None, "contact_latched_hold")
            self.contact_latched = False
            self.contact_candidate_since_s = None
            self.last_completed = feedback
        target = self._next_bounded_target()
        if target is None:
            return self._decision(None, "within_deadband")
        return self._start_command(
            target, observation.monotonic_s, "new_target", feedback
        )

    def status(self) -> dict:
        return {
            "closure_enabled": self.closure_enabled,
            "feedback_encoding": self.feedback_encoding,
            "desired": self.desired,
            "last_completed": self.last_completed,
            "active_target": self.active_target,
            "active_direction": self.active_direction,
            "active_since_s": self.active_since_s,
            "active_start_feedback": self.active_start_feedback,
            "recovery_requested": self.recovery_requested,
            "recovery_reason": self.recovery_reason,
            "fault": self.fault,
            "command_count": self.command_count,
            "active_saw_transition": self.active_saw_transition,
            "contact_candidate_since_s": self.contact_candidate_since_s,
            "contact_latched": self.contact_latched,
            "contact_feedback": self.contact_feedback,
            "contact_effort": self.contact_effort,
            "last_closure_outcome": self.last_closure_outcome,
            "limits": {
                "maximum_command_step": self.maximum_command_step,
                "command_deadband": self.command_deadband,
                "settle_tolerance": self.settle_tolerance,
                "motion_timeout_s": self.motion_timeout_s,
                "contact_detection_enabled": self.contact_detection_enabled,
                "contact_target_threshold": self.contact_target_threshold,
                "contact_shortfall": self.contact_shortfall,
                "contact_min_abs_effort": self.contact_min_abs_effort,
                "contact_confirm_s": self.contact_confirm_s,
                "maximum_abs_effort": self.maximum_abs_effort,
            },
        }


def decision_as_dict(decision: GripperDecision) -> dict:
    return asdict(decision)
