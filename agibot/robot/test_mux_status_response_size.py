#!/usr/bin/env python3
"""Regression: production mux.request must read an entire 32-receipt status.

The historical shared 16 KiB limit truncated normal status history. That old
limit is only a fixture-size reference here; success means the real mux reader
receives all receipts below its current MAX_CHILD_RESPONSE_BYTES ceiling.
Only synthetic JSON over a loopback socket is used: no GDK, subprocess, robot,
model, or persisted simulation state.
"""

import json
import socket
import threading
import unittest

from agibot.robot.g2_groot_place_action_mux import MAX_CHILD_RESPONSE_BYTES, request


HISTORICAL_READ_LIMIT = 16 * 1024
START_POSE = [
    0.5013904571533203,
    -0.1765020340681076,
    1.0539140701293945,
    0.5225321066771904,
    -0.001878945069661977,
    0.8526121988491865,
    0.0030175205845201216,
]


def full_status(*, diagnostics: bool) -> dict:
    results = []
    for index in range(32):
        pose = list(START_POSE)
        pose[0] += (index + 1) * 0.0005
        result = {
            "command_id": f"size-probe-h{index}",
            "accepted": True,
            "queue_delay_s": 0.001234567890123,
            "duration_s": 0.100123456789123,
            "ticks": 5,
            "target_pose": pose,
            "live_pose_at_100ms": list(pose),
            "position_error_at_100ms_m": 0.000012345678901,
            "rotation_error_at_100ms_rad": 0.000123456789012,
            "gripper_target": None,
            "gripper_status": None,
            "simulated": True,
        }
        if diagnostics:
            positions = {f"arm_r_joint{joint}": joint * 0.1234567890123 for joint in range(1, 8)}
            result["controller_diagnostics"] = {
                "compensation_policy": "adaptive",
                "adapt_during_hold": False,
                "last_dispatched_pose": list(pose),
                "joint_hold": {
                    "phase": "hold",
                    "requested_phase": "hold",
                    "source_timestamp_ns": 1_800_000_000_000_000_000 + index * 100_000_000,
                    "last_progress_timestamp_ns": 1_800_000_000_000_000_000 + index * 100_000_000,
                    "latest_positions": positions,
                    "latest_error_codes": dict.fromkeys(positions, 0),
                    "hold_anchor_positions": dict(positions),
                    "current_hold_drift_deg": 0.001234567890123,
                    "max_hold_drift_deg": 0.012345678901234,
                    "worst_joint": "arm_r_joint2",
                    "source_progress_age_s": 0.001234567890123,
                    "settling": False,
                    "settling_remaining_s": 0.0,
                    "sample_count": (index + 1) * 5,
                    "latest_sample_issue": None,
                    "fault": None,
                },
            }
        results.append(result)
    return {
        "ok": True,
        "simulated_arm": True,
        "ready": True,
        "fatal_error": None,
        "queue_depth": 0,
        "control_hz": 50.0,
        "model_waypoint_hz": 10.0,
        "live_pose": list(results[-1]["live_pose_at_100ms"]),
        "desired_pose": list(results[-1]["target_pose"]),
        "recent_results": results,
        "last_result": results[-1],
        "controller_diagnostics": results[-1].get("controller_diagnostics"),
    }


class StatusResponseSizeTest(unittest.TestCase):
    def test_production_mux_reader_preserves_full_status_history(self) -> None:
        for diagnostics in (False, True):
            with self.subTest(joint_diagnostics=diagnostics):
                expected = full_status(diagnostics=diagnostics)
                encoded = (json.dumps(expected) + "\n").encode()
                self.assertGreater(len(encoded), HISTORICAL_READ_LIMIT)
                self.assertLess(len(encoded), MAX_CHILD_RESPONSE_BYTES)
                received_requests = []
                server_errors = []
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                    listener.bind(("127.0.0.1", 0))
                    listener.listen(1)
                    listener.settimeout(3.0)
                    port = listener.getsockname()[1]

                    def serve():
                        try:
                            connection, _ = listener.accept()
                            with connection, connection.makefile("rb") as stream:
                                connection.settimeout(3.0)
                                received_requests.append(json.loads(stream.readline(4096)))
                                # Split the old boundary; the production reader
                                # must still consume the entire status line.
                                connection.sendall(encoded[:HISTORICAL_READ_LIMIT])
                                connection.sendall(encoded[HISTORICAL_READ_LIMIT:])
                        except BaseException as error:
                            server_errors.append(error)

                    thread = threading.Thread(target=serve, daemon=True)
                    thread.start()
                    try:
                        actual = request(port, {"op": "status"}, timeout=3.0)
                    finally:
                        thread.join(timeout=4.0)
                    self.assertFalse(thread.is_alive(), "loopback status server did not exit")
                    self.assertEqual(server_errors, [])
                self.assertEqual(received_requests, [{"op": "status"}])
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual["recent_results"]), 32)
                self.assertEqual(
                    [row["command_id"] for row in actual["recent_results"]],
                    [f"size-probe-h{index}" for index in range(32)],
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
