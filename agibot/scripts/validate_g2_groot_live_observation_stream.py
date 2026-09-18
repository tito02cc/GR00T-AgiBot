#!/usr/bin/env python3
"""Measure stationary live observations from the read-only G2 GR00T bridge.

The bridge is request/response, not a continuous 10 Hz transport.  This test
reports both whether 10 Hz was actually achieved and whether one observation
can be fetched inside the H=8 (0.8 s) replanning budget.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agibot.tools.g2_groot_live_observation_client import (  # noqa: E402
    G2LiveObservationClient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19100)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--period-s", type=float, default=0.1)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "agibot/reports/g2_groot_live_observation_stream_10hz.json",
    )
    return parser.parse_args()


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    if args.samples < 2 or args.period_s <= 0:
        raise ValueError("samples must be >=2 and period must be positive")
    rows = []
    errors = []
    request_start_times: list[float] = []
    previous_hashes: dict[str, str] = {}
    duplicates = {"head_color": 0, "hand_right": 0}
    with G2LiveObservationClient(args.host, args.port) as client:
        bridge = client.get_info()
        started = time.monotonic()
        for index in range(args.samples):
            deadline = started + index * args.period_s
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            request_started = time.monotonic()
            request_start_times.append(request_started)
            try:
                snapshot = client.get_snapshot()
                metadata = snapshot.metadata
                for name, digest in snapshot.source_payload_sha256.items():
                    if digest == previous_hashes.get(name):
                        duplicates[name] += 1
                    previous_hashes[name] = digest
                rows.append(
                    {
                        "index": index,
                        "schedule_lateness_ms": max(
                            0.0, (request_started - deadline) * 1000.0
                        ),
                        "request_latency_ms": (time.monotonic() - request_started) * 1000.0,
                        "capture_duration_ms": float(metadata["capture_duration_ms"]),
                        "camera_skew_ms": float(metadata["camera_skew_ms"]),
                        "camera_to_tf_skew_ms": float(metadata["camera_to_tf_skew_ms"]),
                        "camera_to_joint_skew_ms": float(metadata["camera_to_joint_skew_ms"]),
                        "maximum_state_camera_skew_ms": float(
                            metadata["maximum_state_camera_skew_ms"]
                        ),
                        "right_eef_xyz": metadata["right_eef_xyz_quaternion_xyzw"][:3],
                        "right_gripper_training": float(
                            metadata["right_gripper"]["training_position"]
                        ),
                        "head_sha256": snapshot.source_payload_sha256["head_color"],
                        "hand_right_sha256": snapshot.source_payload_sha256["hand_right"],
                    }
                )
            except Exception as error:
                errors.append(f"sample {index}: {type(error).__name__}: {error}")

    finished = time.monotonic()
    if rows:
        xyz = np.asarray([row["right_eef_xyz"] for row in rows], dtype=np.float64)
        xyz_median = np.median(xyz, axis=0)
        xyz_radius = np.linalg.norm(xyz - xyz_median, axis=1)
        gripper = np.asarray(
            [row["right_gripper_training"] for row in rows], dtype=np.float64
        )
        metrics = {
            "request_latency_ms": summary([row["request_latency_ms"] for row in rows]),
            "schedule_lateness_ms": summary(
                [row["schedule_lateness_ms"] for row in rows]
            ),
            "capture_duration_ms": summary([row["capture_duration_ms"] for row in rows]),
            "camera_skew_ms": summary([row["camera_skew_ms"] for row in rows]),
            "maximum_state_camera_skew_ms": summary(
                [row["maximum_state_camera_skew_ms"] for row in rows]
            ),
            "stationary_eef_max_radius_m": float(xyz_radius.max()),
            "right_gripper_min": float(gripper.min()),
            "right_gripper_max": float(gripper.max()),
            "consecutive_duplicate_frames": duplicates,
        }
        elapsed_s = finished - request_start_times[0]
        start_span_s = request_start_times[-1] - request_start_times[0]
        metrics["wall_elapsed_s"] = elapsed_s
        metrics["completed_request_rate_hz"] = len(rows) / elapsed_s
        metrics["request_start_rate_hz"] = (
            (len(request_start_times) - 1) / start_span_s
            if len(request_start_times) > 1 and start_span_s > 0
            else 0.0
        )
        metrics["period_deadline_misses"] = sum(
            row["schedule_lateness_ms"] > args.period_s * 1000.0 * 0.1
            for row in rows[1:]
        )
    else:
        metrics = {}
    duplicate_limit = max(1, int(0.1 * (args.samples - 1)))
    synchronized_and_complete = (
        len(rows) == args.samples
        and not errors
        and all(count <= duplicate_limit for count in duplicates.values())
        and metrics["maximum_state_camera_skew_ms"]["max"] <= 50.0
    )
    ten_hz_transport_pass = (
        synchronized_and_complete
        and metrics["request_latency_ms"]["max"] <= args.period_s * 1000.0
        and metrics["period_deadline_misses"] == 0
    )
    h8_on_demand_candidate = (
        synchronized_and_complete
        and metrics["request_latency_ms"]["max"] <= 800.0
    )
    if ten_hz_transport_pass:
        status = "PASS_10HZ_TRANSPORT"
    elif h8_on_demand_candidate:
        status = "PASS_SYNCHRONIZED_ON_DEMAND_NOT_10HZ"
    else:
        status = "FAIL"
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "scope": "stationary G2 right-only observation stream, no model and no actuation",
        "target_period_s": args.period_s,
        "ten_hz_transport_pass": ten_hz_transport_pass,
        "h8_on_demand_transport_candidate": h8_on_demand_candidate,
        "interpretation": (
            "The observation bridge serves the latest synchronized snapshot on demand. "
            "A 10 Hz low-level action loop, if later authorized, must run separately on "
            "the robot; this bridge is used once per model replanning cycle."
        ),
        "requested_samples": args.samples,
        "successful_samples": len(rows),
        "bridge": bridge,
        "metrics": metrics,
        "rows": rows,
        "errors": errors,
        "live_execution_enabled": False,
        "motor_commands_sent": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
