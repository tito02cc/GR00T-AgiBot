#!/usr/bin/env python3
"""Service-level regression tests without GDK or physical robot commands."""

import importlib
import io
import json
import socket
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np


sys.modules.setdefault("agibot_gdk", types.ModuleType("agibot_gdk"))

bridge = importlib.import_module("g2_groot_persistent_h1_action_bridge")


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


class RecordingWorker:
    def __init__(self):
        self.gripper = None
        self.stop_event = threading.Event()
        self.fatal_error = None
        self.commands = []

    def submit(self, command):
        self.commands.append(command)

    def status(self):
        return {"submitted_ids": [command.command_id for command in self.commands]}

    def initiate_shutdown(self):
        self.stop_event.set()


class ClientDisconnectTest(unittest.TestCase):
    def run_clients(self, clients):
        worker = RecordingWorker()
        listener = mock.MagicMock()
        listener.__enter__.return_value = listener
        listener.accept.side_effect = [(client, ("127.0.0.1", 12345)) for client in clients]
        with (
            mock.patch.object(bridge.socket, "socket", return_value=listener),
            mock.patch.object(bridge, "emit"),
            mock.patch.object(bridge, "parse_target", return_value=(np.zeros(7), 0.0)),
        ):
            bridge.serve("127.0.0.1", 9201, worker, {}, session_limit_s=30.0)
        return worker

    def test_lost_acceptance_response_preserves_worker_and_submits_once(self):
        for error in (BrokenPipeError(), ConnectionResetError(), socket.timeout()):
            with self.subTest(error=type(error).__name__):
                command = client_connection(
                    {"op": "execute_h1", "command_id": "once"}, send_error=error
                )
                status = client_connection({"op": "status"})
                shutdown = client_connection({"op": "shutdown"})
                worker = self.run_clients([command, status, shutdown])
                self.assertEqual([command.command_id for command in worker.commands], ["once"])
                self.assertEqual(
                    json.loads(status.sendall.call_args.args[0]),
                    {"ok": True, "submitted_ids": ["once"]},
                )

    def test_disconnected_or_idle_reader_allows_next_client(self):
        for error in (ConnectionResetError(), socket.timeout()):
            with self.subTest(error=type(error).__name__):
                disconnected = client_connection({}, read_error=error)
                command = client_connection({"op": "execute_h1", "command_id": "next"})
                shutdown = client_connection({"op": "shutdown"})
                worker = self.run_clients([disconnected, command, shutdown])
                self.assertEqual([command.command_id for command in worker.commands], ["next"])
                disconnected.sendall.assert_not_called()
                for connection in (disconnected, command, shutdown):
                    timeout = connection.settimeout.call_args.args[0]
                    self.assertGreater(timeout, 0.0)
                    self.assertLessEqual(timeout, 30.0)


class LegacyStartupTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.robot, self.tf = mock.Mock(), mock.Mock()
        self.pose = np.asarray([0.5, -0.2, 1.05, 0, 0, 0, 1.0])
        self.state = {"mode": 1, "whole": {}, "right": self.pose}
        self.controller = mock.Mock()
        self.controller.calibrate.side_effect = self.calibrate
        self.worker = mock.Mock()
        self.worker.start.side_effect = lambda: self.events.append("start")
        self.worker.stop.side_effect = lambda: self.events.append("stop")
        self.worker.status.return_value = {"right_gripper": None}
        patchers = {
            "init": mock.patch.object(bridge.agibot_gdk, "gdk_init", return_value=0, create=True),
            "release": mock.patch.object(
                bridge.agibot_gdk, "gdk_release", return_value=0, create=True
            ),
            "res": mock.patch.object(
                bridge.agibot_gdk, "GDKRes", types.SimpleNamespace(kSuccess=0), create=True
            ),
            "robot": mock.patch.object(
                bridge.agibot_gdk, "Robot", return_value=self.robot, create=True
            ),
            "tf": mock.patch.object(bridge.agibot_gdk, "TF", return_value=self.tf, create=True),
            "snapshot": mock.patch.object(bridge, "snapshot", return_value=self.state),
            "safe": mock.patch.object(bridge, "require_safe_state"),
            "competitors": mock.patch.object(bridge, "competing_controllers", return_value=[]),
            "controller": mock.patch.object(
                bridge, "PersistentRightArmController", side_effect=self.make_controller
            ),
            "worker": mock.patch.object(bridge, "ControlWorker", side_effect=self.make_worker),
            "serve": mock.patch.object(
                bridge, "serve", side_effect=lambda *a, **k: self.events.append("serve")
            ),
            "sleep": mock.patch.object(bridge.time, "sleep"),
            "emit": mock.patch.object(bridge, "emit"),
        }
        self.patches = {}
        for name, patcher in patchers.items():
            self.patches[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def make_controller(self, *args, **kwargs):
        self.events.append("controller")
        return self.controller

    def calibrate(self, duration):
        self.events.append("calibrate")
        return {"duration": duration}

    def make_worker(self, *args, **kwargs):
        self.events.append("worker")
        return self.worker

    def run_main(self, *, execute=False, extra=()):
        argv = ["bridge", "--required-motion-mode", "1", "--startup-countdown-s", "0"]
        if execute:
            argv += ["--enable-control", "--confirm", bridge.CONFIRMATION]
        argv += list(extra)
        with mock.patch.object(sys, "argv", argv):
            return bridge.main()

    def test_readonly_main_never_constructs_controller_or_sends_commands(self):
        self.assertEqual(self.run_main(), 0)
        self.patches["controller"].assert_not_called()
        self.patches["worker"].assert_not_called()
        self.patches["serve"].assert_not_called()
        self.robot.end_effector_pose_control.assert_not_called()
        self.robot.move_ee_pos.assert_not_called()
        self.robot.move_end_effector_joint.assert_not_called()
        self.patches["release"].assert_called_once_with()
        preflight = self.patches["emit"].call_args.kwargs
        self.assertEqual(preflight["shutdown_gripper_action"], "hold")

    def test_legacy_main_calibrates_before_worker_and_serve(self):
        self.assertEqual(self.run_main(execute=True), 0)
        self.assertEqual(
            self.events, ["controller", "calibrate", "worker", "start", "serve", "stop"]
        )
        self.controller.calibrate.assert_called_once_with(2.0)
        options = self.patches["controller"].call_args.kwargs
        self.assertEqual(options["required_motion_mode"], 1)
        self.assertNotIn("command_sender", options)
        self.assertNotIn("compensation_policy", options)
        self.assertNotIn("joint_hold_monitor", options)
        self.assertNotIn("adapt_during_hold", options)
        self.assertEqual(self.patches["worker"].call_args.kwargs["shutdown_gripper_action"], "hold")
        self.patches["release"].assert_called_once_with()

    def test_failed_calibration_preserves_error_and_never_starts_worker(self):
        self.controller.calibrate.side_effect = RuntimeError("legacy calibration failed")
        with self.assertRaisesRegex(RuntimeError, "legacy calibration failed"):
            self.run_main(execute=True)
        self.patches["worker"].assert_not_called()
        self.patches["serve"].assert_not_called()
        self.patches["release"].assert_called_once_with()

    def test_failed_worker_start_is_cleaned_up(self):
        self.worker.start.side_effect = RuntimeError("thread startup failed")
        with self.assertRaisesRegex(RuntimeError, "thread startup failed"):
            self.run_main(execute=True)
        self.worker.stop.assert_called_once_with()
        self.patches["serve"].assert_not_called()
        self.patches["release"].assert_called_once_with()

    def test_cleanup_error_does_not_mask_original_gdk_failure(self):
        self.patches["serve"].side_effect = RuntimeError("GDK collision imminent")
        self.worker.stop.side_effect = RuntimeError("cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "^GDK collision imminent$"):
            self.run_main(execute=True)
        self.patches["release"].assert_called_once_with()
        self.assertIn(
            mock.call("worker_cleanup_failed", message="cleanup failed"),
            self.patches["emit"].call_args_list,
        )

    def test_explicit_open_shutdown_option_is_not_silently_changed(self):
        self.assertEqual(
            self.run_main(execute=True, extra=["--shutdown-gripper-action", "open"]), 0
        )
        self.assertEqual(self.patches["worker"].call_args.kwargs["shutdown_gripper_action"], "open")

    def test_control_confirmation_still_required_before_gdk(self):
        with (
            mock.patch.object(
                sys, "argv", ["bridge", "--enable-control", "--required-motion-mode", "1"]
            ),
            mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            bridge.main()
        self.patches["init"].assert_not_called()

    def test_native_backend_is_not_part_of_restored_legacy_cli(self):
        with (
            mock.patch.object(sys, "argv", ["bridge", "--control-backend", "native_trajectory"]),
            mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            bridge.main()
        self.patches["init"].assert_not_called()


class LegacyWorkerTest(unittest.TestCase):
    def setUp(self):
        self.controller_module = importlib.import_module("g2_groot_persistent_right_arm_controller")
        self.pose = np.asarray([0.5, -0.2, 1.05, 0, 0, 0, 1.0])
        self.live = self.pose.copy()
        self.robot, self.tf = object(), object()
        self.now = 10.0
        self.sent = []
        self.stop_after = 5
        self.worker = None
        for patcher in (
            mock.patch.object(bridge, "pose_values", side_effect=lambda *a: self.live.copy()),
            mock.patch.object(
                self.controller_module, "pose_values", side_effect=lambda *a: self.live.copy()
            ),
            mock.patch.object(self.controller_module, "snapshot", return_value={}),
            mock.patch.object(self.controller_module, "require_safe_state"),
            mock.patch.object(
                self.controller_module.time, "monotonic", side_effect=lambda: self.now
            ),
            mock.patch.object(self.controller_module.time, "sleep", side_effect=self.sleep),
            mock.patch.object(self.controller_module, "send", side_effect=self.send),
            mock.patch.object(bridge, "emit"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.controller = self.controller_module.PersistentRightArmController(
            self.robot,
            self.tf,
            initial_translation_compensation=[0, 0, 0],
            initial_rotation_compensation=[0, 0, 0, 1],
            workspace_min=[0.4, -0.3, 0.9],
            workspace_max=[0.9, -0.1, 1.3],
            required_motion_mode=1,
        )
        self.controller.calibrated = True
        adapt = mock.patch.object(self.controller, "_adapt")
        adapt.start()
        self.addCleanup(adapt.stop)
        self.worker = bridge.ControlWorker(self.controller, self.tf, self.robot)

    def sleep(self, duration):
        self.now += duration

    def send(self, robot, target):
        self.sent.append((self.now, target.copy()))
        self.live = target.copy()
        if len(self.sent) >= self.stop_after:
            self.worker.stop_event.set()

    def test_idle_worker_uses_unchanged_fifty_hz_eef_hold(self):
        self.worker._run()
        self.assertEqual(len(self.sent), 5)
        for index, (stamp, target) in enumerate(self.sent):
            self.assertAlmostEqual(stamp, 10.0 + index * 0.02)
            np.testing.assert_allclose(target, self.pose)
        self.assertEqual(self.worker.tick_count, 5)
        self.assertIsNone(self.worker.fatal_error)

    def test_one_h1_row_runs_five_ticks_with_original_target_and_receipt(self):
        target = self.pose.copy()
        target[0] += 0.01
        command = bridge.H1Command("legacy-h1", target, self.now)
        self.worker.submit(command)
        self.worker._run()
        self.assertEqual(len(self.sent), 5)
        self.assertTrue(command.completed.is_set())
        self.assertEqual(command.result["ticks"], 5)
        np.testing.assert_allclose(command.result["target_pose"], target)
        np.testing.assert_allclose(self.sent[-1][1], target)
        self.assertEqual(self.worker.status()["recent_results"][0]["command_id"], "legacy-h1")
        self.assertIsNone(command.error)

    def test_stop_of_unstarted_worker_is_safe_and_does_not_open_tool(self):
        self.worker.gripper = mock.Mock()
        self.assertEqual(self.worker.shutdown_gripper_action, "hold")
        self.worker.stop()
        self.assertTrue(self.worker.stop_event.is_set())
        self.worker.gripper.request_safe_open.assert_not_called()

    def test_hold_shutdown_for_running_worker_never_requests_gripper_open(self):
        self.worker.gripper = mock.Mock()
        with mock.patch.object(self.worker.thread, "is_alive", return_value=True):
            self.worker.initiate_shutdown()
        self.assertTrue(self.worker.stop_event.is_set())
        self.worker.gripper.request_safe_open.assert_not_called()

    def test_original_gdk_error_is_retained_without_diagnostic_dependency(self):
        with mock.patch.object(
            self.controller, "tick", side_effect=RuntimeError("motion control error=2")
        ):
            self.worker._run()
        self.assertEqual(self.worker.fatal_error, "RuntimeError: motion control error=2")
        self.assertFalse(self.worker.status()["ready"])
        self.assertEqual(self.sent, [])

    def test_failed_gripper_handoff_controller_none_keeps_readable_error_status(self):
        self.worker.gripper = mock.Mock()
        self.worker.gripper.status.return_value = {"held": True}
        command = bridge.H1Command("handoff", self.pose.copy(), self.now, 0.0)
        self.worker.submit(command)

        def failed_handoff(_target):
            self.worker.controller = None
            raise RuntimeError("gdk_init failed after gripper handoff")

        with mock.patch.object(self.worker, "_apply_gripper_target", side_effect=failed_handoff):
            self.worker._run()
        status = self.worker.status()
        self.assertFalse(status["ready"])
        self.assertEqual(
            status["fatal_error"], "RuntimeError: gdk_init failed after gripper handoff"
        )
        self.assertTrue(command.completed.is_set())
        self.assertEqual(command.error, status["fatal_error"])
        self.assertEqual(self.sent, [])

    def test_command_ids_remain_unique(self):
        command = bridge.H1Command("once", self.pose.copy(), self.now)
        self.worker.submit(command)
        with self.assertRaisesRegex(ValueError, "duplicate command_id"):
            self.worker.submit(command)
        self.assertEqual(self.worker.commands.qsize(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
