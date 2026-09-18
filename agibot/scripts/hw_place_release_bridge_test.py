#!/usr/bin/env python3
"""Real-hardware test of the placement release ownership handoff.

Runs on the robot over loopback against the real mux, the real Cartesian arm
child and the real omnipicker daemon.  Nothing is simulated.

Every EEF target is the arm child's own reported ``desired_pose``, so the
commanded Cartesian displacement is exactly zero for every waypoint: this
exercises the GDK ownership serialization, not reach.  The arm's travel is
additionally confined by the tight workspace box the mux is started with.

The gripper track is what drives the test.  It stays closed, ramps through the
partial band that the previous mux tore the arm down for, then issues the
float32 representation of the training open bound that the pre-fix daemon
rejected.  That row must perform the whole handoff and still execute its arm
action.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys
import time
from typing import Any
import uuid

CHUNK_COMPLETION_TIMEOUT_S = 25.0
FULLY_OPEN = -0.72
FLOAT32_OPEN = -0.7850000262260437
MAX_LINE_BYTES = 4 * 1024 * 1024

GRIPPER_TRACKS = [
    (
        "closed_transport",
        "recorded 2026-09-03 policy output",
        [-0.0073, -0.0042, -0.0027, -0.0057, -0.0012, -0.0027, -0.0012,
         -0.0027, -0.0012, -0.0042, -0.0073, -0.0042, -0.0073, -0.0088,
         -0.0073, -0.0103],
    ),
    (
        "release_ramp",
        "derived: no recorded run ever reached a release",
        [-0.0103, -0.0402, -0.1587, -0.3129, -0.5218, -0.6733, FLOAT32_OPEN,
         FLOAT32_OPEN, -0.7841, FLOAT32_OPEN, -0.7833, FLOAT32_OPEN, -0.7846,
         FLOAT32_OPEN, -0.7839, FLOAT32_OPEN],
    ),
    (
        "post_release",
        "recorded 2026-09-03 policy output",
        [-0.776, -0.779, -0.776, -0.777, -0.776, -0.780, -0.777, -0.780,
         -0.782, -0.779, -0.785, -0.780, -0.782, -0.785, -0.782, -0.782],
    ),
]


class MuxSession:
    def __init__(self, host: str, port: int, timeout_s: float = 60.0):
        self.connection = socket.create_connection((host, port), timeout=timeout_s)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection.settimeout(timeout_s)
        self.stream = self.connection.makefile("rb")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.connection.sendall(
            (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        )
        line = self.stream.readline(MAX_LINE_BYTES + 1)
        if not line or len(line) > MAX_LINE_BYTES:
            raise RuntimeError("invalid or missing mux response")
        return json.loads(line)

    def close(self) -> None:
        self.stream.close()
        self.connection.close()

    def __enter__(self) -> MuxSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def results_by_id(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["command_id"]): item
        for item in status.get("recent_results", [])
        if isinstance(item, dict) and "command_id" in item
    }


def wait_result(session: MuxSession, command_id: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    status_failures = 0
    while time.monotonic() < deadline:
        last = session.request({"op": "status"})
        if not last.get("ok"):
            status_failures += 1
        if last.get("fatal_error"):
            raise RuntimeError(str(last["fatal_error"]))
        if results_by_id(last).get(command_id) and last.get("queue_depth") == 0:
            last["status_read_failures_during_wait"] = status_failures
            return last
        time.sleep(0.05)
    raise TimeoutError(
        f"command did not settle: {command_id} "
        f"(status read failures during wait: {status_failures})"
    )


def run_chunk(
    session: MuxSession, grippers: list[float], prefix: str
) -> list[dict[str, Any]]:
    """Submit one H16 chunk at 10 Hz with zero-displacement EEF targets."""
    rows: list[dict[str, Any]] = []
    last_command_id = ""
    deadline = time.monotonic()
    for index, gripper in enumerate(grippers):
        # Re-read the arm's own desired pose each row so the commanded
        # displacement stays exactly zero even across the child restart.
        status = session.request({"op": "status"})
        if not status.get("ok"):
            raise RuntimeError(f"status read failed mid-chunk: {status}")
        pose = list(status["desired_pose"])
        command_id = f"hwbridge-{prefix}-h{index}-{uuid.uuid4()}"
        submitted = time.monotonic()
        acknowledgement = session.request(
            {
                "op": "execute_h1_gripper",
                "command_id": command_id,
                "timestamp_ns": time.time_ns(),
                "target_pose": pose,
                "target_gripper": float(gripper),
            }
        )
        if not acknowledgement.get("ok"):
            raise RuntimeError(f"command rejected: {acknowledgement}")
        row = {
            "row": index,
            "command_id": command_id,
            "gripper": float(gripper),
            "target_pose": pose,
            "submit_elapsed_s": time.monotonic() - submitted,
            "arm_executed": bool(acknowledgement.get("accepted")),
        }
        if acknowledgement.get("release") is not None:
            row["release"] = acknowledgement["release"]
        rows.append(row)
        last_command_id = command_id
        deadline += 0.1
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    settled = wait_result(session, last_command_id, CHUNK_COMPLETION_TIMEOUT_S)
    rows[-1]["settled_recent_results"] = len(settled.get("recent_results", []))
    rows[-1]["status_read_failures_during_wait"] = settled.get(
        "status_read_failures_during_wait", 0
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mux-port", type=int, default=9200)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    expected = "EXECUTE_G2_GROOT_HW_PLACE_RELEASE_BRIDGE_TEST"
    if args.confirm != expected:
        parser.error(f"physical execution requires --confirm {expected}")

    report: dict[str, Any] = {
        "schema": "g2_groot_hw_place_release_bridge_test_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "scope": (
            "real mux, real Cartesian arm child, real omnipicker daemon and "
            "real hardware; zero-displacement EEF targets"
        ),
        "status": "STARTED",
        "chunks": [],
    }
    try:
        with MuxSession("127.0.0.1", args.mux_port) as session:
            info = session.request({"op": "info"})
            if not info.get("ok"):
                raise RuntimeError(f"mux info failed: {info}")
            if info.get("simulated_arm"):
                raise RuntimeError("a simulated arm child is attached; aborting")
            opening = session.request({"op": "status"})
            if not opening.get("ok"):
                raise RuntimeError(f"mux status failed: {opening}")
            if not opening.get("ready"):
                raise RuntimeError(f"mux is not ready: {opening.get('fatal_error')}")
            gripper = opening.get("right_gripper") or {}
            report["preflight"] = {
                "schema": info.get("schema"),
                "gripper_command_mode": info.get("gripper_command_mode"),
                "maximum_command_age_s": info.get("maximum_command_age_s"),
                "simulated_arm": info.get("simulated_arm", False),
                "ready": opening.get("ready"),
                "arm_owner_active": opening.get("arm_owner_active"),
                "release_phase": opening.get("release_phase"),
                "desired_pose": opening.get("desired_pose"),
                "live_pose": opening.get("live_pose"),
                "translation_compensation_m": opening.get(
                    "translation_compensation_m"
                ),
                "gripper_position": (gripper.get("last_observation") or {}).get(
                    "raw_position"
                ),
                "gripper_fault": gripper.get("fault"),
            }
            if opening.get("release_phase") != "closed":
                raise RuntimeError(
                    "release_phase must start closed for the handoff to be "
                    f"exercised; got {opening.get('release_phase')}"
                )

            start_pose = list(opening["live_pose"])
            all_rows: list[dict[str, Any]] = []
            for index, (name, source, track) in enumerate(GRIPPER_TRACKS):
                started = time.monotonic()
                rows = run_chunk(session, track, f"c{index}")
                elapsed = time.monotonic() - started
                status = session.request({"op": "status"})
                all_rows.extend(rows)
                handoffs = [row["row"] for row in rows if "release" in row]
                after_gripper = (status.get("right_gripper") or {}).get(
                    "last_observation", {}
                )
                report["chunks"].append(
                    {
                        "chunk": index,
                        "name": name,
                        "gripper_source": source,
                        "gripper_track": track,
                        "elapsed_s": elapsed,
                        "rows": rows,
                        "handoff_rows": handoffs,
                        "release_phase_after": status.get("release_phase"),
                        "arm_owner_active_after": status.get("arm_owner_active"),
                        "ready_after": status.get("ready"),
                        "gripper_after": after_gripper.get("raw_position"),
                        "gripper_fault_after": (
                            status.get("right_gripper") or {}
                        ).get("fault"),
                        "recent_results_after": len(
                            status.get("recent_results", [])
                        ),
                        "live_pose_after": status.get("live_pose"),
                        "translation_compensation_after": status.get(
                            "translation_compensation_m"
                        ),
                        "motion_drift_from_start_m": (
                            sum(
                                (status["live_pose"][i] - start_pose[i]) ** 2
                                for i in range(3)
                            )
                            ** 0.5
                            if status.get("live_pose")
                            else None
                        ),
                    }
                )
                print(
                    json.dumps(
                        {
                            "chunk": index,
                            "name": name,
                            "elapsed_s": round(elapsed, 3),
                            "handoff_rows": handoffs,
                            "release_phase": status.get("release_phase"),
                            "gripper": after_gripper.get("raw_position"),
                            "drift_m": round(
                                report["chunks"][-1]["motion_drift_from_start_m"]
                                or 0.0,
                                6,
                            ),
                        }
                    ),
                    flush=True,
                )

            final = session.request({"op": "status"})
            final_gripper = final.get("right_gripper") or {}
            actual = float(
                (final_gripper.get("last_observation") or {}).get(
                    "raw_position", 0.0
                )
            )
            handoff_rows = [row for row in all_rows if "release" in row]
            report["evaluation"] = {
                "rows_submitted": len(all_rows),
                "rows_without_arm_execution": [
                    row["row"] for row in all_rows if not row["arm_executed"]
                ],
                "handoff_count": len(handoff_rows),
                "gripper_actual_position": actual,
                "gripper_reached_full_open": actual <= FULLY_OPEN,
                "release_phase": final.get("release_phase"),
                "arm_owner_active": final.get("arm_owner_active"),
                "arm_ready": final.get("ready"),
                "gripper_fault": final_gripper.get("fault"),
                "fatal_error": final.get("fatal_error"),
                "max_status_read_failures": max(
                    (row.get("status_read_failures_during_wait", 0)
                     for row in all_rows),
                    default=0,
                ),
                "total_eef_drift_m": (
                    sum(
                        (final["live_pose"][i] - start_pose[i]) ** 2
                        for i in range(3)
                    )
                    ** 0.5
                    if final.get("live_pose")
                    else None
                ),
            }
            report["final_status"] = final
            evaluation = report["evaluation"]
            report["status"] = (
                "PASS_HW_RELEASE_HANDOFF"
                if (
                    evaluation["handoff_count"] == 1
                    and evaluation["gripper_reached_full_open"]
                    and not evaluation["rows_without_arm_execution"]
                    and evaluation["release_phase"] == "open"
                    and evaluation["arm_owner_active"]
                    and evaluation["arm_ready"]
                    and evaluation["gripper_fault"] is None
                    and evaluation["fatal_error"] is None
                    and evaluation["max_status_read_failures"] == 0
                )
                else "FAILED"
            )
            session.request({"op": "shutdown"})
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"status": report["status"], "report": str(args.report)}, indent=2))
    return 0 if report["status"] == "PASS_HW_RELEASE_HANDOFF" else 1


if __name__ == "__main__":
    sys.exit(main())
