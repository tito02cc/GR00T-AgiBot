#!/usr/bin/env python3
"""Bench rehearsal of the placement release handoff against real tool hardware.

Runs on the robot over loopback.  Everything below the driver is the real
deployed code: the real placement action mux, the real omnipicker command
daemon, and the real omnipicker.  Only two things are substituted, because both
require an arm positioned for the task:

* the GR00T policy, replaced by recorded/derived H16 chunks;
* the Cartesian arm child, replaced by ``sim_g2_groot_arm_child_stub.py``,
  which publishes nothing and therefore requests no arm motion.

The submission cadence and the completion check below mirror
``execute_action_chunk`` and ``wait_result`` in
``run_g2_groot_full_protected_inference.py``: one row every 100 ms, then a
single settle wait on the chunk's last command id with the 25 s handoff budget.
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

HORIZON = 16
CHUNK_COMPLETION_TIMEOUT_S = 25.0
RELEASE_INTENT = -0.60
FULLY_OPEN = -0.72
MIN_RETRACTION_M = 0.13
MAX_LINE_BYTES = 4 * 1024 * 1024


class MuxSession:
    """Persistent JSON-line client, same shape as the runner's BridgeSession."""

    def __init__(self, host: str, port: int, timeout_s: float = 60.0):
        self.connection = socket.create_connection((host, port), timeout=timeout_s)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection.settimeout(timeout_s)
        self.stream = self.connection.makefile("rb")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        self.connection.sendall(encoded)
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


def wait_result(
    session: MuxSession, command_id: str, timeout_s: float
) -> dict[str, Any]:
    """Mirror of the runner's settle wait."""
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = session.request({"op": "status"})
        if last.get("fatal_error"):
            raise RuntimeError(str(last["fatal_error"]))
        if results_by_id(last).get(command_id) and last.get("queue_depth") == 0:
            return last
        time.sleep(0.05)
    raise TimeoutError(f"command did not settle: {command_id}")


def execute_action_chunk(
    session: MuxSession, targets: list[list[float]], prefix: str
) -> list[dict[str, Any]]:
    """Mirror of the runner's chunk execution, including the 10 Hz cadence."""
    rows: list[dict[str, Any]] = []
    last_command_id = ""
    deadline = time.monotonic()
    for index, target in enumerate(targets):
        command_id = f"{prefix}-h{index}-{uuid.uuid4()}"
        submitted = time.monotonic()
        acknowledgement = session.request(
            {
                "op": "execute_h1_gripper",
                "command_id": command_id,
                "timestamp_ns": time.time_ns(),
                "target_pose": target[:7],
                "target_gripper": float(target[7]),
            }
        )
        elapsed = time.monotonic() - submitted
        if not acknowledgement.get("ok"):
            raise RuntimeError(f"command rejected: {acknowledgement}")
        row = {
            "command_id": command_id,
            "row": index,
            "gripper": float(target[7]),
            "submit_elapsed_s": elapsed,
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
    wait_result(session, last_command_id, CHUNK_COMPLETION_TIMEOUT_S)
    return rows


def build_chunks(start_pose: list[float]) -> list[dict[str, Any]]:
    """Three H16 chunks covering closed transport, release ramp and retract.

    Chunk 0 and chunk 2 gripper tracks are the values recorded from the real
    policy on 2026-09-03.  Chunk 1 is derived, because no recorded run ever got
    far enough to emit a release ramp: every attempt failed at or before the
    first fully-open command.  The ramp deliberately includes partial values in
    the band that the previous mux tore the arm down for.
    """
    float32_open = -0.7850000262260437
    recorded_closed = [
        -0.0073, -0.0042, -0.0027, -0.0057, -0.0012, -0.0027, -0.0012, -0.0027,
        -0.0012, -0.0042, -0.0073, -0.0042, -0.0073, -0.0088, -0.0073, -0.0103,
    ]
    derived_ramp = [
        -0.0103, -0.0402, -0.1587, -0.3129, -0.5218, -0.6733, float32_open,
        float32_open, -0.7841, float32_open, -0.7833, float32_open, -0.7846,
        float32_open, -0.7839, float32_open,
    ]
    recorded_retract = [
        -0.776, -0.779, -0.776, -0.777, -0.776, -0.780, -0.777, -0.780,
        -0.782, -0.779, -0.785, -0.780, -0.782, -0.785, -0.782, -0.782,
    ]

    cursor = list(start_pose)

    def chunk(grippers: list[float], dx: float, dz: float) -> list[list[float]]:
        """Chain chunks continuously so no inter-chunk step is manufactured."""
        rows = []
        base = list(cursor)
        for index, gripper in enumerate(grippers):
            step = (index + 1) / len(grippers)
            pose = list(base)
            pose[0] += dx * step
            pose[2] += dz * step
            rows.append(pose + [gripper])
        cursor[0] = base[0] + dx
        cursor[2] = base[2] + dz
        return rows

    # Small per-chunk excursions keep every step inside the child's 0.05 m
    # per-waypoint limit; this rehearsal validates the handoff, not reach.
    return [
        {"name": "closed_transport", "targets": chunk(recorded_closed, 0.048, 0.016),
         "gripper_source": "recorded 2026-09-03"},
        {"name": "release_ramp", "targets": chunk(derived_ramp, 0.016, -0.008),
         "gripper_source": "derived (no recorded run reached a release)"},
        {"name": "model_retract", "targets": chunk(recorded_retract, -0.048, 0.024),
         "gripper_source": "recorded 2026-09-03"},
    ]


def evaluate(rows: list[dict[str, Any]], final_status: dict[str, Any],
             release_pose: list[float] | None) -> dict[str, Any]:
    """Apply the runner's placement completion logic to the rehearsal."""
    gripper = final_status.get("right_gripper") or {}
    observation = gripper.get("last_observation") or {}
    actual = float(observation.get("raw_position", 0.0))
    live = final_status.get("live_pose")
    retraction = (
        None
        if release_pose is None or live is None
        else sum((live[i] - release_pose[i]) ** 2 for i in range(3)) ** 0.5
    )
    dropped = [row["row"] for row in rows if not row["arm_executed"]]
    return {
        "rows_submitted": len(rows),
        "rows_without_arm_execution": dropped,
        "gripper_actual_position": actual,
        "gripper_reached_full_open": actual <= FULLY_OPEN,
        "retraction_m": retraction,
        "retraction_sufficient": (
            retraction is not None and retraction >= MIN_RETRACTION_M
        ),
        "release_phase": final_status.get("release_phase"),
        "arm_owner_active": final_status.get("arm_owner_active"),
        "gripper_fault": gripper.get("fault"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mux-port", type=int, default=9200)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--confirm", required=True,
        help="REHEARSE_G2_GROOT_PLACE_RELEASE_HANDOFF",
    )
    args = parser.parse_args()
    if args.confirm != "REHEARSE_G2_GROOT_PLACE_RELEASE_HANDOFF":
        parser.error("physical tool motion requires the explicit confirmation")

    report: dict[str, Any] = {
        "schema": "g2_groot_place_release_rehearsal_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "scope": (
            "real mux, real omnipicker daemon, real omnipicker; simulated arm "
            "child and recorded/derived policy chunks"
        ),
        "status": "STARTED",
        "chunks": [],
    }
    try:
        with MuxSession("127.0.0.1", args.mux_port) as session:
            info = session.request({"op": "info"})
            if not info.get("ok"):
                raise RuntimeError(f"mux info failed: {info}")
            opening = session.request({"op": "status"})
            if not opening.get("ok"):
                raise RuntimeError(f"mux status failed: {opening}")
            report["preflight"] = {
                "schema": info.get("schema"),
                "gripper_command_mode": info.get("gripper_command_mode"),
                "simulated_arm": info.get("simulated_arm"),
                "ready": opening.get("ready"),
                "arm_owner_active": opening.get("arm_owner_active"),
                "release_phase": opening.get("release_phase"),
                "start_pose": opening.get("live_pose"),
                "gripper_position": (
                    (opening.get("right_gripper") or {})
                    .get("last_observation", {})
                    .get("raw_position")
                ),
                "gripper_fault": (opening.get("right_gripper") or {}).get("fault"),
            }
            if not opening.get("ready"):
                raise RuntimeError("mux is not ready")

            start_pose = list(opening["live_pose"])
            release_pose: list[float] | None = None
            all_rows: list[dict[str, Any]] = []
            for index, chunk in enumerate(build_chunks(start_pose)):
                targets = chunk["targets"]
                intents = [
                    row for row in targets if float(row[7]) <= RELEASE_INTENT
                ]
                started = time.monotonic()
                rows = execute_action_chunk(session, targets, f"rehearse-c{index}")
                elapsed = time.monotonic() - started
                status = session.request({"op": "status"})
                if release_pose is None and intents:
                    release_pose = list(intents[0][:7])
                all_rows.extend(rows)
                handoffs = [row for row in rows if "release" in row]
                report["chunks"].append(
                    {
                        "chunk": index,
                        "name": chunk["name"],
                        "gripper_source": chunk["gripper_source"],
                        "gripper_track": [float(row[7]) for row in targets],
                        "elapsed_s": elapsed,
                        "rows": rows,
                        "handoff_rows": [row["row"] for row in handoffs],
                        "release_phase_after": status.get("release_phase"),
                        "arm_owner_active_after": status.get("arm_owner_active"),
                        "gripper_after": (
                            (status.get("right_gripper") or {})
                            .get("last_observation", {})
                            .get("raw_position")
                        ),
                    }
                )
                print(
                    json.dumps(
                        {
                            "chunk": index,
                            "name": chunk["name"],
                            "elapsed_s": round(elapsed, 3),
                            "handoff_rows": [row["row"] for row in handoffs],
                            "release_phase": status.get("release_phase"),
                        }
                    ),
                    flush=True,
                )

            final_status = session.request({"op": "status"})
            report["final_status"] = final_status
            report["evaluation"] = evaluate(all_rows, final_status, release_pose)
            evaluation = report["evaluation"]
            report["status"] = (
                "REHEARSAL_PASS_RELEASE_HANDOFF_COMPLETED"
                if (
                    evaluation["gripper_reached_full_open"]
                    and not evaluation["rows_without_arm_execution"]
                    and evaluation["release_phase"] == "open"
                    and evaluation["arm_owner_active"]
                    and evaluation["gripper_fault"] is None
                )
                else "REHEARSAL_FAILED"
            )
            session.request({"op": "shutdown"})
    except Exception as error:
        report["status"] = "REHEARSAL_FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"status": report["status"], "report": str(args.report)}, indent=2))
    return 0 if report["status"].startswith("REHEARSAL_PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
