#!/usr/bin/env python3
"""Tests for the serialized place-release ownership handoff in the action mux.

The mux owns no GDK objects, so the real robot children are replaced with a
recording fake.  These cover the failures observed on hardware: dropped model
rows during the release ramp, stale command timestamps after the handoff, and
open-wait timeouts that reported no usable cause.
"""

import argparse
import json
import socket
import threading
import time
import unittest
from unittest import mock

import g2_groot_place_action_mux as mux


CLOSED = -0.006917812500000031
FULL_OPEN = -0.7850000262260437  # float32 round-trip of the -0.785 bound


def make_args(**overrides) -> argparse.Namespace:
    values = {
        "bind_host": "127.0.0.1",
        "port": 9200,
        "backend_port": 9201,
        "gripper_port": 9300,
        "required_motion_mode": 1,
        "workspace_min": [0.4172, -0.2392, 0.9750],
        "workspace_max": [0.8403, -0.1031, 1.2674],
        "initial_translation_compensation": None,
        "initial_rotation_compensation": None,
        "session_limit_s": 1800.0,
        "confirm": mux.CONFIRMATION,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class FakeRobot:
    """Stand-in for the gripper daemon and the Cartesian arm child."""

    def __init__(
        self,
        *,
        position: float = CLOSED,
        travel_per_poll: float = 0.25,
        motor_err_code: int = 0,
        whole_end_error: int = 0,
    ):
        self.position = position
        self.travel_per_poll = travel_per_poll
        self.motor_err_code = motor_err_code
        self.whole_end_error = whole_end_error
        self.commanded: list[float] = []
        self.arm_payloads: list[dict] = []
        self.commanded_target: float | None = None
        self.events: list[tuple] = []
        self.arm_pose = [0.5, -0.17, 1.05, 0.52, 0.0, 0.85, 0.0]

    def gripper(self, payload: dict) -> dict:
        operation = payload.get("op")
        if operation == "command":
            target = float(payload["target"])
            if not -0.785 - 1e-4 <= target <= 1e-4:
                raise RuntimeError(f"port 9300 rejected request: {target}")
            self.commanded.append(target)
            self.commanded_target = target
            self.events.append(("tool", target))
            return {"ok": True, "target": target, "commanded": target}
        if operation == "status":
            if self.commanded_target is not None:
                gap = self.position - self.commanded_target
                if gap > 0:
                    self.position -= min(gap, self.travel_per_poll)
                elif gap < 0:
                    self.position += min(-gap, self.travel_per_poll)
            return {
                "ok": True,
                "position": self.position,
                "status": 2,
                "err_code": self.motor_err_code,
                "whole_end_error": self.whole_end_error,
                "effort": 22.4,
            }
        if operation == "ping":
            return {"ok": True, "mode": "right_gripper_only"}
        if operation == "shutdown":
            return {"ok": True, "shutdown": True}
        raise RuntimeError(f"unsupported gripper op {operation}")

    def arm(self, payload: dict) -> dict:
        operation = payload.get("op")
        if operation in ("info", "status"):
            return {
                "ok": True,
                "ready": True,
                "live_pose": list(self.arm_pose),
                "translation_compensation_m": [0.0065, 0.0, 0.0],
                "rotation_compensation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "queue_depth": 0,
                "recent_results": [
                    {"command_id": row["command_id"], "target_pose": row["target_pose"],
                     "accepted": True}
                    for row in self.arm_payloads[-32:]
                ],
            }
        if operation == "execute_h1":
            self.arm_payloads.append(dict(payload))
            self.arm_pose = list(payload["target_pose"])
            self.events.append(("arm", payload["command_id"]))
            return {"ok": True, "accepted": True}
        if operation == "shutdown":
            return {"ok": True, "shutdown": True}
        raise RuntimeError(f"unsupported arm op {operation}")


class MuxHandoffTest(unittest.TestCase):
    def build(self, robot: FakeRobot, **overrides) -> mux.ProcessMux:
        instance = mux.ProcessMux(make_args(**overrides))
        instance.arm = mock.Mock()
        instance.arm.poll.return_value = None
        # Existing handoff cases begin after the explicit activation phase.
        instance.activation_state = "active"

        def dispatch(port, payload, timeout=30.0):
            if port == instance.args.gripper_port:
                return robot.gripper(payload)
            return robot.arm(payload)

        patcher = mock.patch.object(mux, "request", side_effect=dispatch)
        patcher.start()
        self.addCleanup(patcher.stop)

        def restart(owner, **kwargs):
            owner.arm = mock.Mock()
            owner.arm.poll.return_value = None

        started = mock.patch.object(mux.ProcessMux, "start_arm", autospec=True, side_effect=restart)
        self.start_arm = started.start()
        self.addCleanup(started.stop)
        return instance

    def row(self, gripper: float, command_id: str = "place-c0-h0") -> dict:
        return {
            "op": "execute_h1_gripper",
            "command_id": command_id,
            "timestamp_ns": 1_000_000_000,
            "target_pose": [0.5, -0.17, 1.05, 0.52, 0.0, 0.85, 0.0],
            "target_gripper": gripper,
        }

    def test_closed_phase_rows_reach_the_arm_unchanged(self) -> None:
        robot = FakeRobot()
        instance = self.build(robot)
        response = instance.execute(self.row(CLOSED))
        self.assertTrue(response["accepted"])
        self.assertEqual(len(robot.arm_payloads), 1)
        self.assertEqual(robot.arm_payloads[0]["op"], "execute_h1")
        self.assertNotIn("target_gripper", robot.arm_payloads[0])
        self.assertEqual(robot.commanded, [])
        self.assertEqual(instance.release_phase, "closed")

    def test_release_ramp_rows_are_not_dropped(self) -> None:
        """Partial ramp values must keep executing the model's EEF action.

        The previous dead band between -0.05 and -0.72 tore the arm child down
        and returned a synthetic receipt, so those rows never moved the arm and
        produced no execution result for the runner to settle against.
        """
        robot = FakeRobot()
        instance = self.build(robot)
        for index, value in enumerate((-0.06, -0.21, -0.44, -0.68)):
            with self.subTest(gripper=value):
                instance.execute(self.row(value, f"place-c1-h{index}"))
        self.assertEqual(len(robot.arm_payloads), 4)
        self.assertEqual(instance.release_phase, "closed")
        self.assertEqual(robot.commanded, [])
        self.start_arm.assert_not_called()

    def test_full_open_row_performs_handoff_and_still_executes_the_arm(self) -> None:
        robot = FakeRobot()
        instance = self.build(robot)
        with mock.patch.object(instance, "stop_arm") as stop_arm:
            response = instance.execute(self.row(FULL_OPEN, "place-c1-h5"))
        stop_arm.assert_called_once_with()
        self.start_arm.assert_called_once()
        self.assertEqual(self.start_arm.call_args.kwargs, {"restart": True})
        self.assertEqual(instance.release_phase, "open")
        self.assertEqual(robot.commanded[0], FULL_OPEN)
        self.assertEqual(len(robot.arm_payloads), 1)
        self.assertLessEqual(response["release"]["opened_position"], mux.OPEN_THRESHOLD)

    def test_handoff_sends_original_timestamp_once_before_tool_not_replayed(self) -> None:
        robot = FakeRobot()
        instance = self.build(robot)
        with mock.patch.object(instance, "stop_arm"):
            response = instance.execute(self.row(FULL_OPEN, "place-c1-h5"))
        self.assertEqual(robot.arm_payloads[0]["timestamp_ns"], 1_000_000_000)
        self.assertEqual(response["release"]["policy_timestamp_ns"], 1_000_000_000)
        self.assertNotIn("replayed_timestamp_ns", response["release"])
        self.assertTrue(response["release"]["arm_row_completed_before_tool"])
        self.assertEqual(robot.arm_payloads[0]["command_id"], "place-c1-h5")
        self.assertEqual(robot.events[:2], [("arm", "place-c1-h5"), ("tool", FULL_OPEN)])

    def test_no_jaw_motion_is_reported_as_an_ownership_failure(self) -> None:
        """Measured on hardware: move_ee_pos returns 0 and the jaw does not move
        while a Cartesian owner is still present.  That is not an actuator
        problem and must not be reported as one."""
        robot = FakeRobot(travel_per_poll=0.0)
        instance = self.build(robot)
        with (
            mock.patch.object(instance, "stop_arm"),
            mock.patch.object(mux, "GRIPPER_OWNERSHIP_RELEASE_TIMEOUT_S", 0.2),
            mock.patch.object(mux, "GRIPPER_POLL_PERIOD_S", 0.01),
            self.assertRaises(RuntimeError) as caught,
        ):
            instance.execute(self.row(FULL_OPEN))
        message = str(caught.exception)
        self.assertIn("ownership", message)
        self.assertIn("never started moving", message)
        for field in ("start=", "position=", "commands_sent=", "waited_s="):
            self.assertIn(field, message)

    def test_stalled_travel_is_reported_separately_from_ownership(self) -> None:
        """A jaw that starts moving but stalls is an actuator problem."""
        robot = FakeRobot(travel_per_poll=0.004)
        instance = self.build(robot)
        with (
            mock.patch.object(instance, "stop_arm"),
            mock.patch.object(mux, "GRIPPER_TRAVEL_TIMEOUT_S", 0.15),
            mock.patch.object(mux, "GRIPPER_POLL_PERIOD_S", 0.01),
            self.assertRaises(RuntimeError) as caught,
        ):
            instance.execute(self.row(FULL_OPEN))
        message = str(caught.exception)
        self.assertIn("started moving but did not reach target", message)
        self.assertIn("ownership_release_s=", message)
        self.assertIn("travel_s=", message)

    def test_ownership_dead_period_does_not_consume_the_travel_budget(self) -> None:
        """The 4.4 s ownership dead period measured on hardware must not eat
        into the jaw's own travel allowance."""
        robot = FakeRobot(travel_per_poll=0.0)
        instance = self.build(robot)
        with (
            mock.patch.object(instance, "stop_arm"),
            mock.patch.object(mux, "GRIPPER_POLL_PERIOD_S", 0.005),
            mock.patch.object(mux, "GRIPPER_OWNERSHIP_RELEASE_TIMEOUT_S", 0.4),
        ):
            # Release ownership partway through the dead period.
            def release_later() -> None:
                time.sleep(0.2)
                robot.travel_per_poll = 0.25

            import threading

            threading.Thread(target=release_later, daemon=True).start()
            response = instance.execute(self.row(FULL_OPEN))
        release = response["release"]
        self.assertIsNotNone(release["ownership_release_s"])
        self.assertIsNotNone(release["travel_s"])
        self.assertLessEqual(release["opened_position"], mux.OPEN_THRESHOLD)

    def test_hardware_measured_budgets_have_margin(self) -> None:
        """Hardware on 2026-09-07: 4.36 s ownership dead period, 0.62 s travel."""
        self.assertGreaterEqual(mux.GRIPPER_OWNERSHIP_RELEASE_TIMEOUT_S, 4.36 * 1.5)
        self.assertGreaterEqual(mux.GRIPPER_TRAVEL_TIMEOUT_S, 0.62 * 3.0)
        # The whole handoff must still fit the runner's 25 s chunk budget
        # alongside the arm child restart measured at about 5.4 s.
        self.assertLess(
            mux.GRIPPER_OWNERSHIP_RELEASE_TIMEOUT_S
            + mux.GRIPPER_TRAVEL_TIMEOUT_S
            + 5.4
            + mux.GDK_OWNER_HANDOFF_SETTLE_S,
            25.0,
        )

    def test_tool_hardware_fault_aborts_the_release(self) -> None:
        robot = FakeRobot(travel_per_poll=0.0, motor_err_code=7)
        instance = self.build(robot)
        with (
            mock.patch.object(instance, "stop_arm"),
            mock.patch.object(mux, "GRIPPER_OWNERSHIP_RELEASE_TIMEOUT_S", 0.2),
            mock.patch.object(mux, "GRIPPER_POLL_PERIOD_S", 0.01),
            self.assertRaises(RuntimeError) as caught,
        ):
            instance.execute(self.row(FULL_OPEN))
        self.assertIn("motor_err_code=7", str(caught.exception))

    def test_failed_release_leaves_the_arm_down_instead_of_dropping_rows(self) -> None:
        robot = FakeRobot(travel_per_poll=0.0)
        instance = self.build(robot)
        with (
            mock.patch.object(instance, "stop_arm"),
            mock.patch.object(mux, "GRIPPER_OWNERSHIP_RELEASE_TIMEOUT_S", 0.05),
            mock.patch.object(mux, "GRIPPER_POLL_PERIOD_S", 0.01),
        ):
            with self.assertRaises(RuntimeError):
                instance.execute(self.row(FULL_OPEN))
        instance.arm = None
        self.assertIsNotNone(instance.handoff_error)
        with self.assertRaises(RuntimeError) as caught:
            instance.execute(self.row(FULL_OPEN, "place-c1-h6"))
        self.assertIn("gripper handoff failed", str(caught.exception))

    def test_already_open_session_sends_every_row_to_the_arm(self) -> None:
        robot = FakeRobot(position=-0.783)
        instance = self.build(robot)
        instance.release_phase = "open"
        instance.execute(self.row(FULL_OPEN, "resume-h0"))
        self.assertEqual(len(robot.arm_payloads), 1)
        self.assertEqual(robot.commanded, [])
        self.start_arm.assert_not_called()

    def test_stale_open_phase_cannot_hide_an_external_closure(self) -> None:
        robot = FakeRobot(position=CLOSED)
        instance = self.build(robot)
        instance.release_phase = "open"
        with mock.patch.object(instance, "stop_arm"):
            result = instance.execute(self.row(FULL_OPEN))
        self.assertEqual(result["gripper_execution"], "handoff")
        self.assertTrue(robot.commanded)
        self.assertAlmostEqual(robot.position, FULL_OPEN)

    def test_model_close_from_open_runs_arm_then_tool(self) -> None:
        robot = FakeRobot(position=FULL_OPEN)
        instance = self.build(robot)
        with mock.patch.object(instance, "stop_arm"):
            result = instance.execute(self.row(0.0))
        self.assertEqual(result["gripper_handoff"]["direction"], "closing")
        self.assertEqual(robot.commanded, [0.0])
        self.assertAlmostEqual(robot.position, 0.0)
        self.assertEqual(len(robot.arm_payloads), 1)
        self.assertEqual(robot.events[:2], [("arm", "place-c0-h0"), ("tool", 0.0)])

    def test_partial_policy_target_is_explicitly_deferred(self) -> None:
        instance = self.build(FakeRobot())
        result = instance.execute(self.row(-0.35))
        self.assertEqual(result["target_gripper"], -0.35)
        self.assertEqual(result["gripper_execution"], "deferred_until_endpoint")
        self.assertEqual(instance.info()["gripper_policy_mode"], "endpoint_process_handoff")

    def test_manual_close_uses_same_mux_without_an_arm_waypoint(self) -> None:
        robot = FakeRobot(position=FULL_OPEN)
        instance = self.build(robot)
        with mock.patch.object(instance, "stop_arm"):
            result = instance.set_gripper(0.0)
        self.assertEqual(result["gripper_handoff"]["status"], "COMPLETED")
        self.assertAlmostEqual(robot.position, 0.0)
        self.assertEqual(robot.arm_payloads, [])

    def test_handoff_waits_for_triggering_row_execution_receipt(self) -> None:
        robot = FakeRobot()
        instance = self.build(robot)
        instance.execute(self.row(CLOSED, "previous-row"))
        original = instance.arm_request
        status_reads = 0

        def delayed(payload):
            nonlocal status_reads
            result = original(payload)
            if payload["op"] == "status":
                status_reads += 1
                if status_reads < 3:
                    result["queue_depth"] = 1
                    result["recent_results"] = []
            return result

        def stopped():
            self.assertGreaterEqual(status_reads, 3)

        with (
            mock.patch.object(instance, "arm_request", side_effect=delayed),
            mock.patch.object(instance, "stop_arm", side_effect=stopped),
        ):
            result = instance.execute(self.row(FULL_OPEN, "release-row"))
        self.assertEqual(result["release"]["arm_drain"]["result"]["command_id"], "release-row")

    def test_missing_receipt_stops_arm_without_sending_tool_command(self) -> None:
        robot = FakeRobot()
        instance = self.build(robot)
        original = instance.arm_request

        def no_receipt(payload):
            result = original(payload)
            if payload["op"] == "status":
                result["recent_results"] = []
            return result

        with (
            mock.patch.object(instance, "arm_request", side_effect=no_receipt),
            mock.patch.object(mux, "ARM_DRAIN_TIMEOUT_S", 0.02),
            mock.patch.object(instance, "stop_arm") as stop,
            self.assertRaisesRegex(RuntimeError, "receipt missing"),
        ):
            instance.execute(self.row(FULL_OPEN))
        stop.assert_called_once()
        self.assertEqual(robot.commanded, [])
        self.assertEqual(instance.activation_state, "fault")
        with self.assertRaisesRegex(RuntimeError, "receipt missing"):
            instance.execute(self.row(FULL_OPEN, "next-row"))
        self.assertEqual(len(robot.arm_payloads), 1)

    def test_receipts_survive_child_restart(self) -> None:
        robot = FakeRobot()
        instance = self.build(robot)
        instance.execute(self.row(CLOSED, "before-restart"))
        instance.wait_arm_idle()
        robot.arm_payloads.clear()
        instance.execute(self.row(CLOSED, "after-restart"))
        ids = [item["command_id"] for item in instance.status()["recent_results"]]
        self.assertEqual(ids, ["before-restart", "after-restart"])

    def test_open_waits_for_target_tolerance_before_arm_restart(self) -> None:
        robot = FakeRobot(travel_per_poll=0.02)
        instance = self.build(robot)
        with (
            mock.patch.object(instance, "stop_arm"),
            mock.patch.object(mux, "GRIPPER_POLL_PERIOD_S", 0.001),
        ):
            result = instance.execute(self.row(FULL_OPEN))
        self.assertLessEqual(
            abs(result["release"]["final_position"] - FULL_OPEN), mux.GRIPPER_TARGET_TOLERANCE
        )

    def test_dead_child_is_reported_as_not_ready(self) -> None:
        instance = self.build(FakeRobot())
        instance.arm.poll.return_value = 1
        instance.cached_arm_status = {"ready": True, "recent_results": []}
        status = instance.status()
        self.assertFalse(status["ready"])
        self.assertFalse(status["arm_owner_active"])
        self.assertIsNotNone(status["fatal_error"])

    def test_duplicate_id_after_restart_never_reexecutes(self) -> None:
        robot = FakeRobot()
        instance = self.build(robot)
        instance.execute(self.row(CLOSED, "old-command"))
        with mock.patch.object(instance, "stop_arm"):
            instance.execute(self.row(FULL_OPEN, "release-command"))
        commands = list(robot.commanded)
        row_count = len(robot.arm_payloads)
        with self.assertRaisesRegex(ValueError, "duplicate command_id"):
            instance.execute(self.row(0.0, "old-command"))
        self.assertEqual(robot.commanded, commands)
        self.assertEqual(len(robot.arm_payloads), row_count)

    def test_triggering_arm_row_rejection_never_commands_tool_and_latches_fault(self) -> None:
        robot = FakeRobot()
        instance = self.build(robot)
        original = instance.arm_request

        def reject_row(payload):
            if payload["op"] == "execute_h1":
                raise RuntimeError("arm row rejected")
            return original(payload)

        with (
            mock.patch.object(instance, "stop_arm"),
            mock.patch.object(instance, "arm_request", side_effect=reject_row),
            self.assertRaisesRegex(RuntimeError, "arm row rejected"),
        ):
            instance.execute(self.row(FULL_OPEN))
        self.assertEqual(instance.last_handoff["status"], "FAILED")
        self.assertNotIn("final_position", instance.last_handoff)
        self.assertEqual(robot.commanded, [])
        self.assertEqual(robot.arm_payloads, [])
        self.assertFalse(instance.status()["ready"])
        with self.assertRaisesRegex(RuntimeError, "arm row rejected"):
            instance.execute(self.row(FULL_OPEN, "next-row"))
        self.assertEqual(robot.arm_payloads, [])

    def test_failed_or_unaccepted_completion_receipt_never_commands_tool(self) -> None:
        for failure in ({"accepted": False}, {"ok": False}, {"error": "execution failed"}):
            with self.subTest(failure=failure):
                self.doCleanups()
                robot = FakeRobot()
                instance = self.build(robot)
                original = instance.arm_request

                def failed_receipt(payload):
                    response = original(payload)
                    if payload["op"] == "status":
                        response["recent_results"][-1].update(failure)
                    return response

                with (
                    mock.patch.object(instance, "arm_request", side_effect=failed_receipt),
                    mock.patch.object(
                        instance, "stop_arm", side_effect=lambda: setattr(instance, "arm", None)
                    ),
                    self.assertRaisesRegex(RuntimeError, "execution failed before gripper handoff"),
                ):
                    instance.execute(self.row(FULL_OPEN))
                self.assertEqual(robot.commanded, [])
                self.assertIsNone(instance.arm)
                self.assertEqual(instance.activation_state, "fault")
                with self.assertRaises(RuntimeError):
                    instance.execute(self.row(FULL_OPEN, "after-failure"))
                self.assertEqual(len(robot.arm_payloads), 1)

    def test_restart_failure_preserves_tool_result_stops_child_and_rejects_later_rows(self) -> None:
        for failure in ("start", "ready", "fatal"):
            with self.subTest(failure=failure):
                self.doCleanups()
                robot = FakeRobot()
                instance = self.build(robot)
                resumed = False
                original = instance.arm_request

                def restart(*, restart):
                    nonlocal resumed
                    resumed = True
                    instance.arm = mock.Mock()
                    instance.arm.poll.return_value = None
                    if failure == "start":
                        raise RuntimeError("restart failed")

                def resumed_status(payload):
                    response = original(payload)
                    if resumed and payload["op"] == "status":
                        if failure == "ready":
                            response["ready"] = False
                        if failure == "fatal":
                            response["fatal_error"] = "worker failed after restart"
                    return response

                with (
                    mock.patch.object(instance, "start_arm", side_effect=restart),
                    mock.patch.object(
                        instance, "stop_arm", side_effect=lambda: setattr(instance, "arm", None)
                    ) as stop,
                    mock.patch.object(instance, "arm_request", side_effect=resumed_status),
                    self.assertRaisesRegex(RuntimeError, "restart"),
                ):
                    instance.execute(self.row(FULL_OPEN, "release-row"))
                self.assertEqual(stop.call_count, 2)
                self.assertIsNone(instance.arm)
                self.assertEqual(instance.activation_state, "fault")
                self.assertEqual(instance.last_handoff["status"], "FAILED")
                self.assertLessEqual(instance.last_handoff["final_position"], mux.OPEN_THRESHOLD)
                self.assertEqual([row["command_id"] for row in robot.arm_payloads], ["release-row"])
                with self.assertRaises(RuntimeError):
                    instance.execute(self.row(FULL_OPEN, "next-row"))
                self.assertEqual(len(robot.arm_payloads), 1)

    def test_failed_child_start_clears_process_reference(self) -> None:
        instance = mux.ProcessMux(make_args())
        with (
            mock.patch.object(mux.subprocess, "Popen"),
            mock.patch.object(
                instance, "_wait_port", side_effect=RuntimeError("child startup failed")
            ),
            self.assertRaisesRegex(RuntimeError, "startup failed"),
        ):
            instance.start_arm(restart=True)
        self.assertIsNone(instance.arm)

    def test_disconnected_client_does_not_raise_out_of_response_sender(self) -> None:
        connection = mock.Mock()
        connection.sendall.side_effect = BrokenPipeError()
        self.assertFalse(mux.send_response(connection, {"ok": True}))

    def test_gripper_health_is_reported_not_hardcoded(self) -> None:
        robot = FakeRobot(whole_end_error=9)
        instance = self.build(robot)
        self.assertEqual(instance.gripper_status()["fault"], "whole_end_error=9")
        robot.whole_end_error = 0
        self.assertIsNone(instance.gripper_status()["fault"])

    def test_status_does_not_claim_readiness_without_an_arm_owner(self) -> None:
        robot = FakeRobot()
        instance = self.build(robot)
        instance.cached_arm_status = robot.arm({"op": "status"})
        instance.arm = None
        status = instance.status()
        self.assertFalse(status["ready"])
        self.assertFalse(status["arm_owner_active"])
        self.assertTrue(status["live_pose_stale"])

    def test_restart_does_not_reuse_the_loaded_arm_compensation(self) -> None:
        """A capped holding-state compensation made the child exit at startup."""
        robot = FakeRobot()
        instance = mux.ProcessMux(make_args())
        instance.cached_arm_status = robot.arm({"op": "status"})
        with (
            mock.patch.object(mux.subprocess, "Popen") as popen,
            mock.patch.object(mux.ProcessMux, "_wait_port"),
        ):
            instance.start_arm(restart=True)
        command = popen.call_args.args[0]
        self.assertNotIn("--initial-translation-compensation", command)
        self.assertIn("--calibration-duration-s", command)
        self.assertEqual(command[command.index("--calibration-duration-s") + 1], "2.0")


class MuxActivationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = FakeRobot()
        self.instance = mux.ProcessMux(make_args())

        def dispatch(port, payload, timeout=30.0):
            if port == self.instance.args.gripper_port:
                return self.robot.gripper(payload)
            return self.robot.arm(payload)

        request_patcher = mock.patch.object(mux, "request", side_effect=dispatch)
        self.request = request_patcher.start()
        self.addCleanup(request_patcher.stop)

        def start(**kwargs):
            self.instance.arm = mock.Mock()
            self.instance.arm.poll.return_value = None

        start_patcher = mock.patch.object(self.instance, "start_arm", side_effect=start)
        self.start_arm = start_patcher.start()
        self.addCleanup(start_patcher.stop)

        def stop():
            self.instance.arm = None

        stop_patcher = mock.patch.object(self.instance, "stop_arm", side_effect=stop)
        self.stop_arm = stop_patcher.start()
        self.addCleanup(stop_patcher.stop)

    def activate(self, owner=None):
        return self.instance.activate(
            {"op": "activate", "confirm": mux.ACTIVATION_CONFIRMATION}, owner
        )

    def exchange(self, payloads):
        failures = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(2.0)

            def serve():
                try:
                    connection, _ = listener.accept()
                    mux.serve_connection(self.instance, connection, time.monotonic() + 3.0)
                except Exception as error:
                    failures.append(error)

            server = threading.Thread(target=serve, daemon=True)
            server.start()
            responses = []
            with socket.create_connection(listener.getsockname(), timeout=2.0) as connection:
                with connection.makefile("rb") as stream:
                    for payload in payloads:
                        connection.sendall((json.dumps(payload) + "\n").encode())
                        responses.append(json.loads(stream.readline()))
            server.join(timeout=3.0)
            self.assertFalse(server.is_alive())
            self.assertEqual(failures, [])
            return responses

    def test_main_startup_starts_only_daemon(self) -> None:
        with (
            mock.patch.object(mux.argparse.ArgumentParser, "parse_args", return_value=make_args()),
            mock.patch.object(mux, "ProcessMux", return_value=self.instance),
            mock.patch.object(self.instance, "start_daemon") as start_daemon,
            mock.patch.object(mux.socket, "socket") as socket_class,
            mock.patch.object(mux, "serve_connection", return_value=True),
        ):
            socket_class.return_value.__enter__.return_value.accept.return_value = (
                mock.Mock(), ("127.0.0.1", 12345)
            )
            self.assertEqual(mux.main(), 0)
        start_daemon.assert_called_once_with()
        self.start_arm.assert_not_called()
        self.assertEqual(self.instance.activation_state, "standby")

    def test_standby_info_status_are_nonmoving_and_not_fatal(self) -> None:
        info = self.instance.info()
        status = self.instance.status()
        for response in (info, status):
            self.assertEqual(response["activation_protocol"], "explicit_activate_v1")
            self.assertEqual(response["activation_state"], "standby")
            self.assertFalse(response["arm_owner_active"])
        self.assertFalse(status["ready"])
        self.assertIsNone(status["fatal_error"])
        self.assertIn("activate", info["operations"])
        self.assertEqual(info["maximum_horizon"], 16)
        self.assertEqual(info["model_waypoint_hz"], 10.0)
        self.assertEqual(info["gripper_policy_mode"], "endpoint_process_handoff")
        self.start_arm.assert_not_called()
        self.assertEqual(self.robot.commanded, [])
        self.assertEqual(self.robot.arm_payloads, [])
        self.assertTrue(all(call.args[0] == 9300 for call in self.request.call_args_list))

    def test_activate_is_explicit_and_idempotent(self) -> None:
        with self.assertRaisesRegex(ValueError, "confirm"):
            self.instance.activate({"op": "activate"})
        self.start_arm.assert_not_called()
        owner = object()
        first = self.activate(owner)
        second = self.activate(owner)
        self.assertEqual(first, second)
        self.assertTrue(first["ok"])
        self.assertEqual(first["activation_state"], "active")
        self.assertTrue(first["arm_owner_active"])
        self.start_arm.assert_called_once_with()
        self.assertTrue(self.instance.status()["ready"])

    def test_activate_failure_is_latched_and_not_retried(self) -> None:
        self.start_arm.side_effect = RuntimeError("calibration failed")
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "calibration failed"):
                self.activate()
        self.start_arm.assert_called_once_with()
        self.assertEqual(self.instance.activation_state, "fault")
        self.assertFalse(self.instance.status()["ready"])
        self.assertIn("calibration failed", self.instance.status()["fatal_error"])

    def test_execute_before_activation_has_no_arm_or_tool_commands(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not activated"):
            self.instance.execute({"command_id": "blocked", "target_gripper": -0.785})
        self.assertEqual(self.robot.commanded, [])
        self.assertEqual(self.robot.arm_payloads, [])
        self.assertEqual(self.instance.seen_command_ids, set())
        self.request.assert_not_called()
        self.start_arm.assert_not_called()

    def test_standby_set_gripper_uses_daemon_without_arm_takeover(self) -> None:
        with mock.patch.object(mux, "GRIPPER_POLL_PERIOD_S", 0.001):
            response = self.instance.set_gripper(-0.785)
        self.assertTrue(response["ok"])
        self.assertTrue(response["gripper_handoff"]["standby_tool_only"])
        self.assertLessEqual(self.robot.position, mux.OPEN_THRESHOLD)
        self.assertEqual(self.instance.activation_state, "standby")
        self.start_arm.assert_not_called()
        self.stop_arm.assert_not_called()
        self.assertEqual(self.robot.arm_payloads, [])

    def test_readonly_connection_disconnect_keeps_standby(self) -> None:
        responses = self.exchange([{"op": "info"}, {"op": "status"}])
        self.assertTrue(all(response["ok"] for response in responses))
        self.assertEqual(self.instance.activation_state, "standby")
        self.start_arm.assert_not_called()
        self.stop_arm.assert_not_called()
        # The later model connection can still activate this same mux.
        self.assertTrue(self.activate()["ok"])

    def test_model_disconnect_stops_arm_and_locks_session(self) -> None:
        responses = self.exchange([
            {"op": "activate", "confirm": mux.ACTIVATION_CONFIRMATION},
            {"op": "status"},
        ])
        self.assertTrue(responses[0]["ok"])
        self.assertTrue(responses[1]["ready"])
        self.stop_arm.assert_called_once_with()
        self.assertEqual(self.instance.activation_state, "fault")
        self.assertFalse(self.instance.arm_alive())
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            self.activate()
        self.start_arm.assert_called_once_with()
        self.assertEqual(self.robot.commanded, [])

    def test_readonly_disconnect_does_not_stop_another_owner(self) -> None:
        owner = object()
        self.activate(owner)
        self.exchange([{"op": "info"}, {"op": "status"}])
        self.assertEqual(self.instance.activation_state, "active")
        self.assertIs(self.instance.activation_owner, owner)
        self.stop_arm.assert_not_called()

    def test_normal_shutdown_stops_owner_without_disconnect_fault(self) -> None:
        responses = self.exchange([
            {"op": "activate", "confirm": mux.ACTIVATION_CONFIRMATION},
            {"op": "shutdown"},
        ])
        self.assertTrue(responses[1]["shutdown"])
        self.assertEqual(self.instance.activation_state, "stopped")
        self.assertIsNone(self.instance.activation_error)
        self.assertIsNone(self.instance.activation_owner)
        self.assertIsNone(self.instance.status()["fatal_error"])
        self.assertFalse(self.instance.status()["ready"])
        self.stop_arm.assert_called_once_with()

    def test_rejected_pre_activation_requests_do_not_claim_connection(self) -> None:
        responses = self.exchange([
            {"op": "activate", "confirm": "incorrect"},
            {"op": "execute_h1_gripper", "command_id": "no", "target_gripper": -0.785},
        ])
        self.assertTrue(all(not response["ok"] for response in responses))
        self.assertEqual(self.instance.activation_state, "standby")
        self.assertIsNone(self.instance.activation_owner)
        self.start_arm.assert_not_called()
        self.stop_arm.assert_not_called()
        self.assertEqual(self.robot.commanded, [])
        self.assertEqual(self.robot.arm_payloads, [])

    def test_another_connection_cannot_take_over_activation(self) -> None:
        self.activate(object())
        responses = self.exchange([
            {"op": "activate", "confirm": mux.ACTIVATION_CONFIRMATION},
        ])
        self.assertFalse(responses[0]["ok"])
        self.assertIn("another connection", responses[0]["message"])
        self.stop_arm.assert_not_called()
        self.start_arm.assert_called_once_with()


class GripperFaultTest(unittest.TestCase):
    def observation(self, **overrides) -> dict:
        values = {
            "monotonic_s": 1.0,
            "raw_position": CLOSED,
            "motor_status": 2,
            "motor_err_code": 0,
            "whole_end_error": 0,
            "effort": 22.4,
        }
        values.update(overrides)
        return values

    def test_holding_status_and_effort_are_not_faults(self) -> None:
        self.assertIsNone(mux.gripper_fault(self.observation()))
        self.assertIsNone(mux.gripper_fault(self.observation(motor_status=3)))
        self.assertIsNone(mux.gripper_fault(self.observation(effort=24.9)))

    def test_error_codes_are_faults(self) -> None:
        self.assertEqual(mux.gripper_fault(self.observation(motor_err_code=4)), "motor_err_code=4")
        self.assertEqual(
            mux.gripper_fault(self.observation(whole_end_error=1)), "whole_end_error=1"
        )


class ChildResponseCeilingTest(unittest.TestCase):
    """A truncated child status made every settle wait fail on the 2nd chunk."""

    def test_child_response_ceiling_matches_the_runner_client(self) -> None:
        from g2_groot_h1_bridge_client import MAX_RESPONSE_BYTES

        self.assertEqual(mux.MAX_CHILD_RESPONSE_BYTES, MAX_RESPONSE_BYTES)

    def test_child_response_ceiling_clears_a_full_results_deque(self) -> None:
        """32 recent_results measured at 20615 bytes against the old 16 KiB cap."""
        self.assertGreater(mux.MAX_CHILD_RESPONSE_BYTES, 32 * 1024)

    def test_inbound_request_limit_stays_tight(self) -> None:
        self.assertEqual(mux.MAX_REQUEST_LINE_BYTES, 16384)

    def test_oversized_child_status_is_read_not_rejected(self) -> None:
        """A status response larger than the old ceiling must pass through."""
        payload = {
            "ok": True,
            "ready": True,
            "queue_depth": 0,
            "recent_results": [
                {
                    "command_id": f"probe-h{index}",
                    "target_pose": [0.5, -0.17, 1.05, 0.52, 0.0, 0.85, 0.0],
                    "live_pose_at_100ms": [0.5, -0.17, 1.05, 0.52, 0.0, 0.85, 0.0],
                    "padding": "x" * 400,
                }
                for index in range(32)
            ],
        }
        encoded = (json.dumps(payload) + "\n").encode()
        self.assertGreater(len(encoded), 16384)

        import threading

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]

            def serve() -> None:
                connection, _ = listener.accept()
                with connection, connection.makefile("rb") as stream:
                    stream.readline(65536)
                    connection.sendall(encoded)

            server = threading.Thread(target=serve, daemon=True)
            server.start()
            response = mux.request(port, {"op": "status"}, 5.0)
            server.join(timeout=5)

        self.assertEqual(len(response["recent_results"]), 32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
