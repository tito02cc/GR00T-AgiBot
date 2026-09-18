#!/usr/bin/env python3
"""Tests for the right-omnipicker command daemon's actuator range handling.

``agibot_gdk`` only exists on the robot, so a minimal stub stands in for the
GDK types the daemon touches.  The cases below pin the float32 boundary
behaviour that rejected the model's own fully-open command on hardware.
"""

import importlib
import io
import json
import socket
import sys
import types
import unittest
from unittest import mock

import numpy as np


def install_gdk_stub() -> types.ModuleType:
    module = types.ModuleType("agibot_gdk")

    class JointState:
        def __init__(self):
            self.position = 0.0

    class JointStates:
        def __init__(self):
            self.group = ""
            self.target_type = ""
            self.states = []
            self.nums = 0

    class GDKRes:
        kSuccess = 0

    module.JointState = JointState
    module.JointStates = JointStates
    module.GDKRes = GDKRes
    module.gdk_init = lambda: GDKRes.kSuccess
    module.gdk_release = lambda: GDKRes.kSuccess
    module.Robot = object
    sys.modules["agibot_gdk"] = module
    return module


install_gdk_stub()
daemon = importlib.import_module("g2_groot_right_gripper_command_daemon")


class RecordingRobot:
    def __init__(self, result_code: int = 0):
        self.result_code = result_code
        self.commanded: list[float] = []
        self.groups: list[tuple[str, str, int]] = []

    def move_ee_pos(self, joints) -> int:
        self.commanded.append(joints.states[0].position)
        self.groups.append((joints.group, joints.target_type, joints.nums))
        return self.result_code


class ClampCommandTest(unittest.TestCase):
    def test_float32_open_bound_is_accepted_and_clamped(self) -> None:
        """The policy path stores the -0.785 clip in a float32 action array.

        That round-trip yields -0.7850000262260437, which the previous closed
        interval check rejected as out of range, aborting the release.
        """
        saturated = float(np.float32(-0.785))
        self.assertLess(saturated, daemon.OPEN_COMMAND)
        self.assertEqual(daemon.clamp_command(saturated), daemon.OPEN_COMMAND)

    def test_exact_bounds_are_unchanged(self) -> None:
        self.assertEqual(daemon.clamp_command(daemon.OPEN_COMMAND), -0.785)
        self.assertEqual(daemon.clamp_command(daemon.CLOSED_COMMAND), 0.0)

    def test_interior_values_pass_through(self) -> None:
        for value in (-0.78, -0.72, -0.55, -0.2, -0.006917812500000031):
            with self.subTest(value=value):
                self.assertEqual(daemon.clamp_command(value), value)

    def test_values_beyond_the_tolerance_are_rejected(self) -> None:
        for value in (-0.79, -0.7852, 0.01, 1.0, -2.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    daemon.clamp_command(value)

    def test_tolerance_is_far_below_the_control_deadband(self) -> None:
        """The slack must not be able to mask a meaningful command error."""
        self.assertLess(daemon.COMMAND_BOUND_TOLERANCE, 0.01)

    def test_non_finite_targets_are_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    daemon.clamp_command(value)


class SendGripperTest(unittest.TestCase):
    def test_saturated_open_reaches_gdk_at_the_actuator_bound(self) -> None:
        robot = RecordingRobot()
        outcome = daemon.send_gripper(robot, float(np.float32(-0.785)))
        self.assertEqual(robot.commanded, [daemon.OPEN_COMMAND])
        self.assertEqual(robot.groups, [("right_tool", "omnipicker", 1)])
        self.assertEqual(outcome["commanded"], daemon.OPEN_COMMAND)
        self.assertEqual(outcome["result"], 0)

    def test_non_zero_result_code_raises(self) -> None:
        robot = RecordingRobot(result_code=2)
        with self.assertRaises(RuntimeError) as caught:
            daemon.send_gripper(robot, -0.78)
        self.assertIn("move_ee_pos returned 2", str(caught.exception))

    def test_out_of_range_target_never_reaches_gdk(self) -> None:
        robot = RecordingRobot()
        with self.assertRaises(ValueError):
            daemon.send_gripper(robot, -0.9)
        self.assertEqual(robot.commanded, [])


def client_connection(request: dict, *, read_error=None, send_error=None):
    connection = mock.MagicMock()
    connection.__enter__.return_value = connection
    stream = io.BytesIO((json.dumps(request) + "\n").encode())
    if read_error is not None:
        stream = mock.MagicMock()
        stream.readline.side_effect = read_error
    connection.makefile.return_value.__enter__.return_value = stream
    connection.sendall.side_effect = send_error
    return connection


class ClientDisconnectTest(unittest.TestCase):
    def run_clients(self, clients):
        robot = RecordingRobot()
        listener = mock.MagicMock()
        listener.__enter__.return_value = listener
        listener.accept.side_effect = [(client, ("127.0.0.1", 12345)) for client in clients]
        with (
            mock.patch.object(daemon.socket, "socket", return_value=listener),
            mock.patch.object(daemon, "require_loopback"),
            mock.patch.object(daemon.time, "sleep"),
            mock.patch.object(daemon.agibot_gdk, "Robot", return_value=robot),
            mock.patch.object(daemon.agibot_gdk, "gdk_release") as release,
            mock.patch.object(daemon, "read_gripper", return_value={"position": 0.0}),
            mock.patch.object(sys, "argv", ["daemon", "--confirm", daemon.CONFIRMATION]),
        ):
            self.assertEqual(daemon.main(), 0)
        release.assert_called_once_with()
        return robot

    def test_lost_command_response_does_not_restart_or_repeat_command(self):
        for error in (BrokenPipeError(), ConnectionResetError(), socket.timeout()):
            with self.subTest(error=type(error).__name__):
                command = client_connection({"op": "command", "target": 0.0}, send_error=error)
                status = client_connection({"op": "status"})
                shutdown = client_connection({"op": "shutdown"})
                robot = self.run_clients([command, status, shutdown])
                self.assertEqual(robot.commanded, [0.0])
                response = json.loads(status.sendall.call_args.args[0])
                self.assertTrue(response["ok"])
                self.assertEqual(response["position"], 0.0)

    def test_disconnected_or_idle_reader_does_not_block_next_client(self):
        for error in (ConnectionResetError(), socket.timeout()):
            with self.subTest(error=type(error).__name__):
                disconnected = client_connection({}, read_error=error)
                command = client_connection({"op": "command", "target": 0.0})
                shutdown = client_connection({"op": "shutdown"})
                robot = self.run_clients([disconnected, command, shutdown])
                self.assertEqual(robot.commanded, [0.0])
                disconnected.sendall.assert_not_called()
                for connection in (disconnected, command, shutdown):
                    timeout = connection.settimeout.call_args.args[0]
                    self.assertGreater(timeout, 0.0)
                    self.assertLessEqual(timeout, daemon.CLIENT_TIMEOUT_S)


if __name__ == "__main__":
    unittest.main(verbosity=2)
