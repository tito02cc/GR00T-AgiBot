from __future__ import annotations

import math
import unittest

from g2_groot_gripper_state_machine import (
    CLOSED_COMMAND,
    FEEDBACK_NATIVE_RADIANS,
    OPEN_COMMAND,
    GripperObservation,
    GripperStateMachine,
    command_to_raw,
    feedback_to_command,
    raw_to_command,
)


def obs(
    time_s: float,
    command: float = OPEN_COMMAND,
    *,
    status: int = 0,
    motor_error: int = 0,
    whole_error: int = 0,
    effort: float = 0.0,
) -> GripperObservation:
    return GripperObservation(
        monotonic_s=time_s,
        raw_position=command_to_raw(command),
        motor_status=status,
        motor_err_code=motor_error,
        whole_end_error=whole_error,
        effort=effort,
    )


class GripperStateMachineTest(unittest.TestCase):
    def assertCommand(self, actual: float | None, expected: float) -> None:
        self.assertIsNotNone(actual)
        self.assertAlmostEqual(float(actual), expected, places=12)

    def test_mapping_endpoints_and_roundtrip(self) -> None:
        self.assertEqual(command_to_raw(OPEN_COMMAND), 0.0)
        self.assertEqual(command_to_raw(CLOSED_COMMAND), 120.0)
        for value in (-0.785, -0.76, -0.55, -0.2, 0.0):
            with self.subTest(value=value):
                self.assertAlmostEqual(raw_to_command(command_to_raw(value)), value)

    def test_native_feedback_mapping_is_identity(self) -> None:
        for value in (-0.785, -0.55, -0.2, 0.0):
            with self.subTest(value=value):
                self.assertAlmostEqual(
                    feedback_to_command(value, FEEDBACK_NATIVE_RADIANS), value
                )

    def test_native_zero_is_closed_not_open(self) -> None:
        machine = GripperStateMachine(
            closure_enabled=True,
            feedback_encoding=FEEDBACK_NATIVE_RADIANS,
        )
        machine.desired = CLOSED_COMMAND
        machine.last_completed = CLOSED_COMMAND
        decision = machine.step(
            GripperObservation(
                monotonic_s=0.0,
                raw_position=0.0,
                motor_status=0,
                motor_err_code=0,
                whole_end_error=0,
            )
        )
        self.assertIsNone(decision.command)
        self.assertEqual(decision.reason, "within_deadband")

    def test_idle_open_emits_no_command(self) -> None:
        machine = GripperStateMachine()
        decision = machine.step(obs(0.0))
        self.assertIsNone(decision.command)
        self.assertEqual(decision.reason, "within_deadband")

    def test_last_desired_is_not_queued_while_motion_is_active(self) -> None:
        machine = GripperStateMachine()
        machine.update_desired(-0.76)
        self.assertCommand(machine.step(obs(0.0)).command, -0.76)
        machine.update_desired(-0.74)
        moving = machine.step(obs(0.2, -0.77, status=1))
        self.assertIsNone(moving.command)
        self.assertEqual(moving.reason, "motion_in_progress")
        self.assertCommand(machine.step(obs(0.5, -0.76)).command, -0.74)
        self.assertEqual(machine.command_count, 2)

    def test_bounded_steps_serialize_a_large_target(self) -> None:
        machine = GripperStateMachine(
            closure_enabled=True, maximum_command_step=0.2
        )
        machine.update_desired(0.0)
        self.assertCommand(machine.step(obs(0.0)).command, -0.585)
        self.assertCommand(machine.step(obs(0.5, -0.585)).command, -0.385)

    def test_closure_is_rejected_when_not_enabled(self) -> None:
        machine = GripperStateMachine(closure_enabled=False)
        with self.assertRaisesRegex(ValueError, "closure is not enabled"):
            machine.update_desired(-0.5)

    def test_motion_timeout_faults_without_claiming_target_reached(self) -> None:
        machine = GripperStateMachine(motion_timeout_s=1.0)
        machine.update_desired(-0.76)
        self.assertCommand(machine.step(obs(0.0)).command, -0.76)
        timeout = machine.step(obs(1.1, -0.77, status=1))
        self.assertIsNone(timeout.command)
        self.assertEqual(timeout.reason, "closing_motion_timeout_fault")
        self.assertEqual(timeout.fault, "closing_motion_timeout")
        self.assertAlmostEqual(machine.last_completed, -0.77)
        self.assertEqual(machine.last_closure_outcome, "closing_motion_timeout")
        self.assertFalse(machine.recovery_requested)
        hold = machine.step(obs(1.2, -0.77, status=2))
        self.assertIsNone(hold.command)
        self.assertEqual(hold.reason, "fault_latched_no_command")

    def test_blocked_contact_retains_feedback_and_holds_newer_close_target(self) -> None:
        machine = GripperStateMachine(closure_enabled=True)
        machine.update_desired(-0.50)
        self.assertCommand(machine.step(obs(0.0)).command, -0.585)
        machine.update_desired(-0.30)
        blocked = machine.step(obs(0.2, -0.60, status=2))
        self.assertIsNone(blocked.command)
        self.assertEqual(blocked.reason, "hardware_blocked_or_holding")
        self.assertAlmostEqual(machine.last_completed, -0.60)
        self.assertAlmostEqual(machine.contact_feedback, -0.60)
        self.assertTrue(machine.contact_latched)
        hold = machine.step(obs(0.3, -0.60, status=2))
        self.assertIsNone(hold.command)
        self.assertEqual(hold.reason, "contact_latched_hold")
        self.assertEqual(machine.command_count, 1)

    def test_no_motion_blocked_feedback_cannot_fake_close_completion(self) -> None:
        for status in (2, 3):
            with self.subTest(status=status):
                machine = GripperStateMachine(
                    closure_enabled=True,
                    maximum_command_step=0.785,
                    motion_timeout_s=1.0,
                )
                machine.update_desired(CLOSED_COMMAND)
                self.assertCommand(machine.step(obs(0.0)).command, CLOSED_COMMAND)
                waiting = machine.step(obs(0.2, OPEN_COMMAND, status=status, effort=8.0))
                self.assertIsNone(waiting.command)
                self.assertEqual(waiting.reason, "motion_in_progress")
                self.assertFalse(machine.contact_latched)
                self.assertEqual(machine.last_completed, OPEN_COMMAND)
                timeout = machine.step(obs(1.1, OPEN_COMMAND, status=status, effort=8.0))
                self.assertEqual(timeout.fault, "closing_motion_timeout")
                self.assertEqual(machine.last_completed, OPEN_COMMAND)
                self.assertIsNone(machine.active_target)
                self.assertEqual(machine.command_count, 1)

    def test_blocked_contact_does_not_repeat_target_and_allows_partial_reopen(self) -> None:
        machine = GripperStateMachine(closure_enabled=True, maximum_command_step=0.785)
        machine.update_desired(CLOSED_COMMAND)
        self.assertCommand(machine.step(obs(0.0)).command, CLOSED_COMMAND)
        machine.step(obs(0.2, -0.20, status=2))
        machine.update_desired(CLOSED_COMMAND)
        hold = machine.step(obs(0.3, -0.20, status=0))
        self.assertIsNone(hold.command)
        self.assertEqual(hold.reason, "contact_latched_hold")
        machine.update_desired(-0.40)
        self.assertCommand(machine.step(obs(0.4, -0.20, status=2)).command, -0.40)
        self.assertFalse(machine.contact_latched)
        self.assertFalse(machine.recovery_requested)
        self.assertIsNone(machine.step(obs(0.6, -0.40)).command)
        self.assertAlmostEqual(machine.last_completed, -0.40)
        self.assertEqual(machine.command_count, 2)

    def test_transient_status_three_with_zero_error_is_post_contact_hold(self) -> None:
        machine = GripperStateMachine(
            closure_enabled=True, maximum_command_step=0.785
        )
        machine.update_desired(-0.01)
        self.assertCommand(machine.step(obs(0.0)).command, -0.01)
        held = machine.step(obs(0.2, -0.01, status=3, effort=22.5))
        self.assertIsNone(held.command)
        self.assertEqual(held.reason, "hardware_post_contact_hold")
        self.assertIsNone(held.fault)
        settled = machine.step(obs(0.4, -0.01, status=0, effort=22.5))
        self.assertIsNone(settled.command)
        self.assertEqual(settled.reason, "within_deadband")

    def test_status_three_with_gdk_error_remains_fatal(self) -> None:
        machine = GripperStateMachine(closure_enabled=True)
        machine.update_desired(-0.01)
        decision = machine.step(obs(0.0, -0.01, status=3, motor_error=7))
        self.assertIsNone(decision.command)
        self.assertEqual(decision.fault, "motor_error_7")

    def test_explicit_safe_open_preempts_active_close(self) -> None:
        machine = GripperStateMachine()
        machine.update_desired(-0.76)
        machine.step(obs(0.0))
        machine.request_safe_open("bridge_shutdown")
        decision = machine.step(obs(0.1, -0.77, status=1))
        self.assertCommand(decision.command, OPEN_COMMAND)
        self.assertEqual(decision.reason, "safe_open:bridge_shutdown")

    def test_safe_open_when_already_open_sends_no_command(self) -> None:
        machine = GripperStateMachine()
        machine.request_safe_open("shutdown")
        decision = machine.step(obs(0.0, OPEN_COMMAND))
        self.assertIsNone(decision.command)
        self.assertEqual(decision.reason, "already_open_recovery_latched")

    def test_health_faults_never_emit_recovery_commands(self) -> None:
        cases = [
            (obs(0.0, motor_error=7), "motor_error_7"),
            (obs(0.0, whole_error=9), "whole_end_error_9"),
            (obs(0.0, status=4), "unknown_motor_status_4"),
            (GripperObservation(0.0, math.nan, 0, 0, 0), "non_finite_feedback"),
        ]
        for observation, fault in cases:
            with self.subTest(fault=fault):
                machine = GripperStateMachine()
                machine.update_desired(-0.76)
                decision = machine.step(observation)
                self.assertIsNone(decision.command)
                self.assertEqual(decision.reason, "health_fault_no_command")
                self.assertEqual(decision.fault, fault)

    def test_safe_open_timeout_faults_without_claiming_completion(self) -> None:
        machine = GripperStateMachine(motion_timeout_s=1.0)
        machine.update_desired(-0.76)
        machine.step(obs(0.0))
        machine.request_safe_open("test")
        self.assertCommand(
            machine.step(obs(0.1, -0.77, status=1)).command, OPEN_COMMAND
        )
        timeout = machine.step(obs(1.2, -0.77, status=1))
        self.assertIsNone(timeout.command)
        self.assertEqual(timeout.reason, "opening_motion_timeout_fault")
        self.assertEqual(timeout.fault, "opening_motion_timeout")
        hold = machine.step(obs(1.3, -0.77, status=2))
        self.assertIsNone(hold.command)
        self.assertEqual(hold.reason, "fault_latched_no_command")

    def test_blocked_status_cannot_fake_open_completion(self) -> None:
        machine = GripperStateMachine(
            closure_enabled=True,
            maximum_command_step=0.785,
            motion_timeout_s=1.0,
        )
        machine.desired = -0.01
        machine.last_completed = -0.01
        machine.update_desired(OPEN_COMMAND)
        self.assertCommand(machine.step(obs(0.0, -0.01, status=2)).command, OPEN_COMMAND)
        waiting = machine.step(obs(0.2, -0.01, status=2))
        self.assertIsNone(waiting.command)
        self.assertEqual(waiting.reason, "motion_in_progress")
        self.assertAlmostEqual(machine.last_completed, -0.01)
        timeout = machine.step(obs(1.1, -0.01, status=2))
        self.assertEqual(timeout.fault, "opening_motion_timeout")

    def _advance_to_final_close(self, machine: GripperStateMachine) -> None:
        machine.update_desired(-0.008070454)
        self.assertCommand(machine.step(obs(0.0)).command, -0.585)
        self.assertCommand(machine.step(obs(0.5, -0.585)).command, -0.385)
        self.assertCommand(machine.step(obs(1.0, -0.385)).command, -0.185)
        self.assertCommand(
            machine.step(obs(1.5, -0.185)).command, -0.008070454
        )

    def test_contact_candidate_requires_transition_and_persistence(self) -> None:
        machine = GripperStateMachine(
            closure_enabled=True,
            contact_detection_enabled=True,
            contact_confirm_s=0.15,
        )
        self._advance_to_final_close(machine)
        early = machine.step(obs(1.7, -0.20, status=0, effort=8.0))
        self.assertEqual(early.reason, "motion_in_progress")
        moving = machine.step(obs(1.8, -0.15, status=1, effort=8.0))
        self.assertEqual(moving.reason, "motion_in_progress")
        candidate = machine.step(obs(1.9, -0.15, status=0, effort=8.0))
        self.assertEqual(candidate.reason, "motion_in_progress")
        latched = machine.step(obs(2.06, -0.15, status=0, effort=8.0))
        self.assertEqual(latched.reason, "contact_candidate_latched")
        self.assertTrue(machine.contact_latched)
        hold = machine.step(obs(2.2, -0.15, status=0, effort=8.0))
        self.assertEqual(hold.reason, "contact_latched_hold")
        self.assertIsNone(hold.command)

    def test_target_reached_is_not_contact_evidence(self) -> None:
        machine = GripperStateMachine(
            closure_enabled=True, contact_detection_enabled=True
        )
        self._advance_to_final_close(machine)
        decision = machine.step(obs(2.0, -0.008070454, status=0, effort=22.5))
        self.assertIsNone(decision.command)
        self.assertFalse(machine.contact_latched)
        self.assertEqual(
            machine.last_closure_outcome, "target_reached_no_contact_evidence"
        )

    def test_policy_open_after_contact_allows_next_close(self) -> None:
        machine = GripperStateMachine(
            closure_enabled=True,
            contact_detection_enabled=True,
            contact_confirm_s=0.1,
        )
        self._advance_to_final_close(machine)
        machine.step(obs(1.7, -0.15, status=1, effort=8.0))
        machine.step(obs(1.8, -0.15, status=0, effort=8.0))
        machine.step(obs(1.91, -0.15, status=0, effort=8.0))
        self.assertTrue(machine.contact_latched)
        machine.update_desired(OPEN_COMMAND)
        recovery = machine.step(obs(2.0, -0.15, status=0, effort=8.0))
        self.assertCommand(recovery.command, -0.35)
        self.assertEqual(recovery.reason, "new_target")
        self.assertFalse(machine.recovery_requested)
        self.assertFalse(machine.contact_latched)
        self.assertCommand(machine.step(obs(2.1, -0.35)).command, -0.55)
        self.assertCommand(machine.step(obs(2.2, -0.55)).command, -0.75)
        self.assertCommand(machine.step(obs(2.3, -0.75)).command, OPEN_COMMAND)
        self.assertIsNone(machine.step(obs(2.4, OPEN_COMMAND)).command)
        machine.update_desired(CLOSED_COMMAND)
        self.assertCommand(machine.step(obs(2.5, OPEN_COMMAND)).command, -0.585)

    def test_effort_and_transition_without_travel_cannot_fake_contact(self) -> None:
        machine = GripperStateMachine(
            closure_enabled=True,
            maximum_command_step=0.785,
            contact_detection_enabled=True,
            contact_confirm_s=0.1,
            motion_timeout_s=1.0,
        )
        machine.update_desired(CLOSED_COMMAND)
        self.assertCommand(machine.step(obs(0.0)).command, CLOSED_COMMAND)
        machine.step(obs(0.1, OPEN_COMMAND, status=1, effort=8.0))
        machine.step(obs(0.2, OPEN_COMMAND, status=0, effort=8.0))
        waiting = machine.step(obs(0.4, OPEN_COMMAND, status=0, effort=8.0))
        self.assertEqual(waiting.reason, "motion_in_progress")
        self.assertFalse(machine.contact_latched)
        self.assertEqual(machine.last_completed, OPEN_COMMAND)
        self.assertEqual(
            machine.step(obs(1.1, OPEN_COMMAND, effort=8.0)).fault,
            "closing_motion_timeout",
        )

    def test_effort_limit_is_a_fail_closed_health_fault(self) -> None:
        machine = GripperStateMachine(
            closure_enabled=True, contact_detection_enabled=True
        )
        machine.update_desired(-0.5)
        decision = machine.step(obs(0.0, effort=25.1))
        self.assertIsNone(decision.command)
        self.assertEqual(decision.fault, "effort_limit_exceeded")

    def test_contact_configuration_requires_closure(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires closure_enabled"):
            GripperStateMachine(contact_detection_enabled=True)


if __name__ == "__main__":
    unittest.main()
