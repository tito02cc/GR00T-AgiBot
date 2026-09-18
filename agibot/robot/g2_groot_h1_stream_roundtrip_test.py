#!/usr/bin/env python3
"""Guarded 10 Hz streaming round-trip test for an already armed H1 bridge."""

from __future__ import annotations

import argparse
import json
import time

from g2_groot_h1_bridge_client import BridgeSession


CONFIRMATION = "TEST_G2_GROOT_H1_STREAM_0P5MM"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if not args.execute or args.confirm != CONFIRMATION:
        parser.error(f"physical test requires --execute --confirm {CONFIRMATION}")

    with BridgeSession(args.host, args.port) as session:
        initial = session.request({"op": "status"})
        if not initial.get("ok") or not initial.get("ready"):
            raise RuntimeError(f"bridge is not ready: {initial}")
        origin = list(initial["desired_pose"])
        offsets_m = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
                     0.0004, 0.0003, 0.0002, 0.0001, 0.0]
        results = []
        command_ids = []
        send_times = []
        started = time.monotonic()
        next_send = started
        for index, offset_m in enumerate(offsets_m):
            target = origin.copy()
            target[0] += offset_m
            command_id = f"stream-roundtrip-{time.time_ns()}-{index:02d}"
            response = session.request({
                "op": "execute_h1",
                "command_id": command_id,
                "timestamp_ns": time.time_ns(),
                "target_pose": target,
            })
            if not response.get("ok"):
                raise RuntimeError(f"H1 step {index} failed: {response}")
            command_ids.append(command_id)
            send_times.append(time.monotonic())
            next_send += 0.1
            remaining = next_send - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        stream_duration_s = time.monotonic() - started
        time.sleep(0.5)
        final = session.request({"op": "status"})
        results_by_id = {
            item["command_id"]: item for item in final.get("recent_results", [])
        }
        results = [results_by_id[item] for item in command_ids if item in results_by_id]
    if len(results) != len(command_ids):
        raise RuntimeError(
            f"only {len(results)}/{len(command_ids)} H1 results completed"
        )
    max_position_error_m = max(
        item["position_error_at_100ms_m"] for item in results
    )
    max_rotation_error_rad = max(
        item["rotation_error_at_100ms_rad"] for item in results
    )
    passed = (
        final.get("ok")
        and final.get("ready")
        and final["fatal_error"] is None
        and max_position_error_m <= 0.0015
        and max_rotation_error_rad <= 0.02
        and final["live_target_position_error_m"] <= 0.0002
        and final["live_target_rotation_error_rad"] <= 0.002
    )
    report = {
        "schema": "g2_groot_h1_stream_roundtrip_test_v1",
        "passed": passed,
        "origin_pose": origin,
        "offsets_m": offsets_m,
        "stream_duration_s": stream_duration_s,
        "effective_waypoint_hz": (
            (len(send_times) - 1) / (send_times[-1] - send_times[0])
        ),
        "max_position_error_at_100ms_m": max_position_error_m,
        "max_rotation_error_at_100ms_rad": max_rotation_error_rad,
        "final_position_error_m": final["live_target_position_error_m"],
        "final_rotation_error_rad": final["live_target_rotation_error_rad"],
        "steps": results,
        "final_status": final,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
