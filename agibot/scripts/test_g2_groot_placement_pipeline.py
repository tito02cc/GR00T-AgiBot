"""SOFTWARE integration of the production runner, TCP client, and process mux.

Policy rows, child arm execution, gripper feedback, and time are synthetic.
This test exercises no model, GDK, robot, or physical task-success claim.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
import socket
import threading
import unittest
from unittest import mock

from agibot.robot import g2_groot_place_action_mux as mux
from agibot.scripts import run_g2_groot_full_protected_inference as runner
from agibot.scripts.run_g2_groot_full_place_inference import (
    ACTIVATION_CONFIRMATION,
    PlacementBridgeSession,
    inspect_standby_bridge,
    validate_activation_state,
)
import numpy as np


class SyntheticClock:
    def __init__(self):
        self.now = 0.0
        self.lock = threading.Lock()

    def monotonic(self):
        with self.lock:
            return self.now

    def time_ns(self):
        return 1_800_000_000_000_000_000 + round(self.monotonic() * 1_000_000_000)

    def sleep(self, duration):
        with self.lock:
            self.now += max(0.0, duration)


class SyntheticProcess:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise AssertionError("synthetic child was not shut down")
        return self.returncode


class SyntheticChildren:
    def __init__(self, owner, clock, initial_pose):
        self.owner = owner
        self.clock = clock
        self.pose = list(initial_pose)
        self.generation = 0
        self.pending = deque()
        self.recent_results = deque(maxlen=32)
        self.executed = []
        self.arm_submissions = []
        self.gripper_position = runner.OPEN
        self.gripper_target = None
        self.gripper_ready_at = 0.0
        self.gripper_transitions = []
        self.events = []

    def start_arm(self, *, restart=False):
        if self.pending:
            raise AssertionError("arm restarted before queued actions completed")
        self.generation += 1
        self.recent_results.clear()
        self.owner.arm = SyntheticProcess()
        self.events.append({"event": "arm_started", "generation": self.generation})

    def start_daemon(self):
        self.owner.daemon = SyntheticProcess()

    def finish_due_actions(self):
        while self.pending and self.pending[0][0] <= self.clock.monotonic():
            _, payload = self.pending.popleft()
            self.pose = list(payload["target_pose"])
            receipt = {
                "command_id": payload["command_id"],
                "target_pose": list(self.pose),
                "actual_pose": list(self.pose),
                "accepted": True,
                "ok": True,
                "synthetic_arm_generation": self.generation,
            }
            self.recent_results.append(receipt)
            self.executed.append(receipt)
            self.events.append({"event": "arm_completed", "command_id": payload["command_id"]})

    def request(self, port, payload, timeout=30.0):
        self.clock.sleep(0.001)
        self.finish_due_actions()
        operation = payload["op"]
        if port == self.owner.args.backend_port:
            if not self.owner.arm_alive():
                raise AssertionError("request reached a stopped synthetic arm")
            if operation == "execute_h1":
                self.arm_submissions.append(dict(payload))
                self.events.append({"event": "arm_submitted", "command_id": payload["command_id"]})
                self.pending.append((self.clock.monotonic() + 0.1, dict(payload)))
                return {"ok": True, "accepted": True}
            if operation in ("status", "info"):
                return {
                    "ok": True,
                    "ready": True,
                    "fatal_error": None,
                    "live_pose": list(self.pose),
                    "desired_pose": list(self.pose),
                    "queue_depth": len(self.pending),
                    "recent_results": list(self.recent_results),
                    "server_time_ns": self.clock.time_ns(),
                }
            if operation == "shutdown":
                if self.pending:
                    raise AssertionError("handoff dropped pending arm actions")
                self.owner.arm.returncode = 0
                self.events.append({"event": "arm_stopped", "generation": self.generation})
                return {"ok": True, "shutdown": True}
        elif port == self.owner.args.gripper_port:
            if operation == "command":
                if self.owner.arm_alive():
                    raise AssertionError("tool command overlaps Cartesian ownership")
                target = float(payload["target"])
                if target != self.gripper_target:
                    self.gripper_target = target
                    self.gripper_ready_at = self.clock.monotonic() + 0.35
                    self.gripper_transitions.append(target)
                    self.events.append(
                        {"event": "tool_command", "target": target, "pose": list(self.pose)}
                    )
                return {"ok": True}
            if operation == "status":
                if (
                    self.gripper_target is not None
                    and self.clock.monotonic() >= self.gripper_ready_at
                ):
                    gap = self.gripper_target - self.gripper_position
                    self.gripper_position += max(-0.25, min(0.25, gap))
                moving = (
                    self.gripper_target is not None and self.gripper_position != self.gripper_target
                )
                return {
                    "ok": True,
                    "position": self.gripper_position,
                    "status": 1 if moving else 0,
                    "err_code": 0,
                    "whole_end_error": 0,
                    "effort": 1.0,
                }
            if operation == "shutdown":
                self.owner.daemon.returncode = 0
                return {"ok": True, "shutdown": True}
        raise AssertionError(f"unexpected synthetic child request: {port}, {payload}")


class LoopbackMuxServer:
    def __init__(self, owner):
        self.owner = owner
        self.error = None
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(5.0)
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self.serve, daemon=True)

    def serve(self):
        try:
            stopping = False
            deadline = mux.time.monotonic() + 1800.0
            while not stopping:
                connection, _ = self.listener.accept()
                # Exercise the production connection ownership, activation,
                # protocol dispatch, and EOF/shutdown cleanup, not a copy.
                stopping = mux.serve_connection(self.owner, connection, deadline)
        except BaseException as error:
            self.error = error
        finally:
            try:
                self.owner.close()
            except BaseException as error:
                self.error = self.error or error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc_info):
        self.listener.close()
        self.thread.join(timeout=6.0)
        if self.thread.is_alive():
            raise AssertionError("synthetic mux server did not exit")
        if self.error is not None:
            raise self.error


class PlacementPipelineSoftwareTest(unittest.TestCase):
    def test_four_h16_chunks_survive_handoffs_with_exact_actions_and_receipts(self):
        clock = SyntheticClock()
        owner = mux.ProcessMux(argparse.Namespace(backend_port=9201, gripper_port=9300))
        initial_pose = [0.5, -0.17, 1.05, 0.0, 0.0, 0.0, 1.0]
        children = SyntheticChildren(owner, clock, initial_pose)
        targets = np.tile(np.append(initial_pose, runner.OPEN), (4 * runner.HORIZON, 1))
        targets[:, 0] += np.arange(len(targets)) * 0.001
        targets[16:24, 7] = np.linspace(runner.OPEN, -0.10, 8)
        targets[24:32, 7] = 0.0
        targets[32:40, 7] = np.linspace(0.0, -0.69, 8)
        targets[40:, 7] = runner.OPEN
        targets[48:, 0] = np.linspace(targets[47, 0], initial_pose[0] - 0.13, 16)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "time", clock))
            stack.enter_context(mock.patch.object(mux, "time", clock))
            stack.enter_context(mock.patch.object(mux, "GRIPPER_POLL_PERIOD_S", 0.01))
            stack.enter_context(mock.patch.object(mux, "GDK_OWNER_HANDOFF_SETTLE_S", 0.05))
            stack.enter_context(mock.patch.object(mux, "request", side_effect=children.request))
            stack.enter_context(
                mock.patch.object(owner, "start_arm", side_effect=children.start_arm)
            )
            stack.enter_context(
                mock.patch.object(owner, "start_daemon", side_effect=children.start_daemon)
            )
            owner.start_daemon()
            rows = []
            with LoopbackMuxServer(owner) as server:
                standby = inspect_standby_bridge("127.0.0.1", server.port)
                self.assertEqual(standby["activation_state"], "standby")
                self.assertFalse(standby["arm_owner_active"])
                self.assertEqual(children.generation, 0)
                self.assertIsNone(owner.arm)
                self.assertIsNone(owner.activation_owner)
                with PlacementBridgeSession("127.0.0.1", server.port) as client:
                    activated = client.request(
                        {
                            "op": "activate",
                            "confirm": ACTIVATION_CONFIRMATION,
                        }
                    )
                    self.assertTrue(activated["ok"])
                    info = client.request({"op": "info"})
                    status = client.request({"op": "status"})
                    validate_activation_state(info, status, "active")
                    self.assertEqual(children.generation, 1)
                    self.assertIsNotNone(owner.activation_owner)
                    self.assertEqual(info["gripper_policy_mode"], "endpoint_process_handoff")
                    for cycle in range(4):
                        chunk = targets[cycle * runner.HORIZON : (cycle + 1) * runner.HORIZON]
                        final_target, chunk_rows = runner.execute_action_chunk(
                            client, chunk, f"synthetic-c{cycle}", 0
                        )
                        rows.extend(chunk_rows)
                        np.testing.assert_array_equal(final_target, chunk[-1, :7])
                    final_status = client.request({"op": "status"})

            self.assertEqual(children.generation, 3)
            self.assertEqual(children.gripper_transitions, [0.0, runner.OPEN])
            self.assertEqual(len(rows), 64)
            self.assertTrue(all(row["status"] == "COMPLETED" for row in rows))
            expected_ids = [row["command_id"] for row in rows]
            self.assertEqual(len(set(expected_ids)), 64)
            self.assertEqual([row["command_id"] for row in children.executed], expected_ids)
            self.assertEqual(
                [row["command_id"] for row in final_status["recent_results"]], expected_ids
            )
            self.assertEqual([row["completion"]["command_id"] for row in rows], expected_ids)
            np.testing.assert_array_equal([row["pose"] for row in rows], targets[:, :7])
            np.testing.assert_array_equal([row["gripper"] for row in rows], targets[:, 7])
            np.testing.assert_array_equal(
                [row["target_pose"] for row in children.arm_submissions], targets[:, :7]
            )
            np.testing.assert_array_equal(
                [row["target_pose"] for row in children.executed], targets[:, :7]
            )
            submissions = np.asarray([row["submitted_monotonic_s"] for row in rows])
            self.assertTrue(np.all(np.diff(submissions) >= runner.MODEL_PERIOD_S - 1e-9))
            handoff_rows = [index for index, row in enumerate(rows) if "gripper_handoff" in row]
            self.assertEqual(handoff_rows, [24, 40])
            for index in handoff_rows:
                self.assertTrue(rows[index]["cadence_rebased"])
                self.assertGreater(rows[index]["acknowledgement_s"], 0.35)
                self.assertGreaterEqual(
                    rows[index + 1]["submitted_monotonic_s"]
                    - rows[index]["acknowledged_monotonic_s"],
                    runner.MODEL_PERIOD_S - 1e-9,
                )
                handoff = rows[index]["gripper_handoff"]
                self.assertEqual(handoff["status"], "COMPLETED")
                self.assertEqual(handoff["arm_drain"]["command_id"], expected_ids[index])
                self.assertTrue(handoff["arm_row_completed_before_tool"])
                self.assertNotIn("replayed_timestamp_ns", handoff)
                np.testing.assert_array_equal(handoff["arm_pose_before_tool"], targets[index, :7])
                submitted_index = next(
                    event_index for event_index, event in enumerate(children.events)
                    if event["event"] == "arm_submitted"
                    and event["command_id"] == expected_ids[index]
                )
                completed_index = next(
                    event_index for event_index, event in enumerate(children.events)
                    if event["event"] == "arm_completed"
                    and event["command_id"] == expected_ids[index]
                )
                self.assertEqual(children.events[completed_index + 1]["event"], "arm_stopped")
                tool_event = children.events[completed_index + 2]
                self.assertEqual(tool_event["event"], "tool_command")
                np.testing.assert_array_equal(tool_event["pose"], targets[index, :7])
                self.assertEqual(children.events[completed_index + 3]["event"], "arm_started")
                self.assertEqual(children.events[completed_index + 4], {
                    "event": "arm_submitted", "command_id": expected_ids[index + 1],
                })
                self.assertLess(submitted_index, completed_index)
            self.assertEqual(final_status["queue_depth"], 0)
            self.assertIsNone(final_status["fatal_error"])
            self.assertEqual(final_status["right_gripper"]["last_completed"], runner.OPEN)
            np.testing.assert_array_equal(final_status["live_pose"], targets[-1, :7])
            self.assertIsNone(owner.arm)
            self.assertIsNone(owner.daemon)
            self.assertEqual(owner.activation_state, "stopped")


if __name__ == "__main__":
    unittest.main()
