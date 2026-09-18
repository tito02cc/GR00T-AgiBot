from __future__ import annotations

import unittest

from g2_groot_gripper_runtime import (
    PROTECTED_MAXIMUM_COMMAND,
    GripperRuntime,
)
from g2_groot_gripper_state_machine import (
    OPEN_COMMAND,
    GripperObservation,
    command_to_raw,
)


class Feedback:
    def __init__(self) -> None:
        self.time_s = 0.0
        self.command = OPEN_COMMAND
        self.status = 0
        self.motor_error = 0
        self.whole_error = 0

    def read(self) -> GripperObservation:
        return GripperObservation(
            self.time_s,
            command_to_raw(self.command),
            self.status,
            self.motor_error,
            self.whole_error,
        )


class GripperRuntimeTest(unittest.TestCase):
    def test_provisional_limit_rejects_unapproved_closure(self) -> None:
        feedback = Feedback()
        runtime = GripperRuntime(feedback.read, lambda _: None)
        with self.assertRaisesRegex(ValueError, "approved maximum"):
            runtime.update_desired(-0.75)

    def test_poll_sends_only_state_machine_decisions(self) -> None:
        feedback = Feedback()
        sent: list[float] = []
        runtime = GripperRuntime(feedback.read, sent.append)
        runtime.update_desired(-0.76)
        runtime.poll()
        runtime.poll()
        self.assertEqual(sent, [-0.76])
        feedback.time_s = 0.5
        feedback.command = -0.76
        runtime.poll()
        runtime.poll()
        self.assertEqual(sent, [-0.76])

    def test_shutdown_preempts_and_latches_open(self) -> None:
        feedback = Feedback()
        sent: list[float] = []
        runtime = GripperRuntime(feedback.read, sent.append)
        runtime.update_desired(-0.76)
        runtime.poll()
        feedback.time_s = 0.1
        feedback.command = -0.77
        feedback.status = 1
        runtime.request_safe_open("shutdown")
        runtime.poll()
        self.assertEqual(sent, [-0.76, OPEN_COMMAND])
        feedback.time_s = 0.6
        feedback.command = OPEN_COMMAND
        feedback.status = 0
        runtime.poll()
        self.assertTrue(runtime.safe_open_complete)

    def test_hardware_error_emits_no_command(self) -> None:
        feedback = Feedback()
        sent: list[float] = []
        runtime = GripperRuntime(feedback.read, sent.append)
        runtime.update_desired(-0.76)
        feedback.motor_error = 11
        decision = runtime.poll()
        self.assertIsNone(decision.command)
        self.assertEqual(sent, [])
        self.assertEqual(runtime.status()["fault"], "motor_error_11")

    def test_closure_runtime_accepts_full_policy_range_without_contact_remap(self) -> None:
        feedback = Feedback()
        runtime = GripperRuntime(
            feedback.read,
            lambda _: None,
            closure_enabled=True,
            maximum_allowed_command=PROTECTED_MAXIMUM_COMMAND,
        )
        self.assertFalse(runtime.machine.contact_detection_enabled)
        self.assertEqual(runtime.machine.contact_target_threshold, -0.55)
        self.assertEqual(
            runtime.status()["maximum_allowed_command"],
            PROTECTED_MAXIMUM_COMMAND,
        )
        runtime.update_desired(PROTECTED_MAXIMUM_COMMAND)
        self.assertEqual(runtime.machine.desired, 0.0)

    def test_blocked_hardware_hold_does_not_resend_gripper_command(self) -> None:
        feedback = Feedback()
        sent: list[float] = []
        runtime = GripperRuntime(
            feedback.read,
            sent.append,
            closure_enabled=True,
            maximum_allowed_command=PROTECTED_MAXIMUM_COMMAND,
        )
        runtime.update_desired(-0.2)
        runtime.poll()
        feedback.time_s = 0.2
        feedback.command = -0.25
        feedback.status = 2
        runtime.poll()
        runtime.poll()
        self.assertEqual(len(sent), 1)
        self.assertAlmostEqual(sent[0], -0.2)


if __name__ == "__main__":
    unittest.main()
