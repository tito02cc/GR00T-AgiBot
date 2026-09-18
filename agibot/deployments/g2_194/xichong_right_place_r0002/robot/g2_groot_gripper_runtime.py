#!/usr/bin/env python3
"""Hardware-callback adapter around the pure G2 gripper state machine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from g2_groot_gripper_state_machine import (
    FEEDBACK_RAW_0_120,
    OPEN_COMMAND,
    GripperDecision,
    GripperObservation,
    GripperStateMachine,
)


PROVISIONAL_MAXIMUM_COMMAND = -0.760
PROTECTED_MAXIMUM_COMMAND = 0.0


class GripperRuntime:
    """Poll feedback and execute at most one serialized hardware command."""

    def __init__(
        self,
        read_observation: Callable[[], GripperObservation],
        send_command: Callable[[float], None],
        *,
        closure_enabled: bool = False,
        maximum_allowed_command: float = PROVISIONAL_MAXIMUM_COMMAND,
        feedback_encoding: str = FEEDBACK_RAW_0_120,
    ) -> None:
        if not OPEN_COMMAND <= maximum_allowed_command <= 0.0:
            raise ValueError("invalid maximum_allowed_command")
        if not closure_enabled and maximum_allowed_command >= -0.55:
            raise ValueError("provisional runtime cannot enable closure range")
        self.read_observation = read_observation
        self.send_command = send_command
        self.maximum_allowed_command = float(maximum_allowed_command)
        self.machine = GripperStateMachine(
            closure_enabled=closure_enabled,
            maximum_command_step=0.785,
            # Ignore sub-0.015 rad policy noise while the fingers are already
            # settled. On GDK 3.3.8 every tool transaction briefly changes
            # the global motion-control status, so milliradian chatter must
            # not interrupt the simultaneous 50 Hz arm stream.
            command_deadband=0.015,
            contact_detection_enabled=False,
            maximum_abs_effort=1000.0,
            # Begin contact detection as soon as the policy enters the
            # training-defined closure region.  Waiting until -0.1 would turn
            # an earlier obstruction into a 1.5 s motion timeout instead of a
            # bounded contact latch.
            contact_target_threshold=-0.55,
            feedback_encoding=feedback_encoding,
        )
        self.last_observation: GripperObservation | None = None
        self.last_decision: GripperDecision | None = None
        self.command_events: list[dict[str, Any]] = []

    def update_desired(self, command: float) -> None:
        value = float(command)
        if value > self.maximum_allowed_command:
            raise ValueError(
                f"gripper target {value} exceeds approved maximum "
                f"{self.maximum_allowed_command}"
            )
        self.machine.update_desired(value)

    def request_safe_open(self, reason: str) -> None:
        self.machine.request_safe_open(reason)

    def poll(self) -> GripperDecision:
        observation = self.read_observation()
        decision = self.machine.step(observation)
        self.last_observation = observation
        self.last_decision = decision
        if decision.command is not None:
            self.send_command(decision.command)
            self.command_events.append(
                {
                    "monotonic_s": observation.monotonic_s,
                    "command": decision.command,
                    "reason": decision.reason,
                }
            )
        return decision

    @property
    def safe_open_complete(self) -> bool:
        return bool(
            self.machine.recovery_requested
            and self.machine.active_target is None
            and self.machine.last_completed == OPEN_COMMAND
            and self.machine.fault is None
        )

    def status(self) -> dict[str, Any]:
        return {
            **self.machine.status(),
            "maximum_allowed_command": self.maximum_allowed_command,
            "safe_open_complete": self.safe_open_complete,
            "last_observation": (
                asdict(self.last_observation) if self.last_observation else None
            ),
            "last_decision": (
                asdict(self.last_decision) if self.last_decision else None
            ),
            "command_events": list(self.command_events[-16:]),
        }
