#!/usr/bin/env python3
"""Complete placement with asynchronous RTC and the established GDK joint owner."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agibot.rtc.action_queue import ActionQueue
from agibot.rtc.contract import validate_capability
from agibot.rtc.live import ExecutionTimeline, SnapshotWorker
from agibot.rtc.planner import AsyncRtcPlanner
from agibot.scripts.run_g2_groot_full_place_inference import (
    ACTIVATION_CONFIRMATION,
    TASK_PROMPT,
    TRAINING_GRIPPER_REFERENCE,
    TRAINING_REFERENCE,
    PlacementBridgeSession,
    PlacementProgress,
    inspect_standby_bridge,
    prepare_model,
    validate_activation_state,
    validate_preflight,
    validate_snapshot,
)
from agibot.scripts.run_g2_groot_full_protected_inference import (
    calibrate_bridge_clock,
    validate_complete_chunk,
)
from agibot.tools.g2_gr00t_shadow_adapter import build_policy_observation, decode_action_chunk
from agibot.tools.g2_groot_live_observation_client import G2LiveObservationClient
from gr00t.policy.server_client import PolicyClient


CONFIRMATION = "EXECUTE_G2_GROOT_RTC_RIGHT_ARM_PLACE_INFERENCE"


def model_observation(snapshot, prompt):
    return build_policy_observation(
        snapshot.head_color_rgb,
        snapshot.hand_right_rgb,
        np.asarray(snapshot.metadata["right_eef_xyz_quaternion_xyzw"]),
        float(snapshot.metadata["right_gripper"]["training_position"]),
        prompt,
    )


def run_stream(
    action_client,
    actions,
    planner,
    observer,
    progress,
    report,
    clock_offset_ns,
    *,
    prompt,
    frozen_steps=8,
    request_remaining=10,
    ramp_rate=3.0,
):
    """Single socket owner dispatches at 10 Hz; observation/model calls never block it."""
    timeline = ExecutionTimeline()
    rows = report["executions"]
    row_by_id = {}
    last_snapshot_index = None
    last_request_index = None
    next_send = time.monotonic()
    next_status = next_send
    status = None
    finalizing = False
    aligned_snapshot = None
    receipt_cursor = None
    last_skip = None
    exhaustion_since = None
    while True:
        now = time.monotonic()
        if now >= next_status:
            # Latch BEFORE the status RPC: a new frame arriving during this
            # RPC must not overwrite the older, time-alignable frame.
            aligned_snapshot = observer.latest()
            status_requested_at = now
            status = action_client.request(
                {"op": "status", "compact": True, "after_command_id": receipt_cursor}
            )
            report.setdefault("status_requests", []).append(
                {
                    "monotonic_s": now,
                    "roundtrip_s": time.monotonic() - now,
                    "new_receipts": len(status.get("recent_results", [])),
                }
            )
            if not status.get("ready") or status.get("fatal_error"):
                raise RuntimeError(f"bridge is not healthy: {status.get('fatal_error')}")
            timeline.update(status)
            newly_completed = []
            for receipt in status.get("recent_results", []):
                row = row_by_id.get(receipt.get("command_id"))
                if row is not None and row["status"] != "COMPLETED":
                    row.update(status="COMPLETED", completion=receipt)
                    newly_completed.append(row)
                    receipt_cursor = receipt["command_id"]
            progress.observe_executions(newly_completed, len(report["predictions"]))
            task = progress.status(status)
            if task["passed"]:
                finalizing = True
            if finalizing and status.get("queue_depth") == 0:
                # Check again after all already committed commands complete.
                if task["passed"]:
                    report.update(
                        status="PASS_PLACED_RELEASED_AND_RETRACTED",
                        task_progress=task,
                        final_bridge_status=status,
                    )
                    return
                finalizing = False
            next_status = time.monotonic() + 0.04

        result = planner.poll()
        if result is not None:
            report["predictions"].append(result)
            print(
                json.dumps(
                    {
                        "event": "rtc_prediction",
                        "inference_s": result["inference_s"],
                        "merge": result["merge"],
                    }
                ),
                flush=True,
            )
            # A late result leaves the existing queue untouched. A fresh camera
            # sample may retry while remaining old actions continue; no replay
            # or silent ordinary-inference fallback.

        if not finalizing and not planner.pending and actions.remaining <= request_remaining:
            latest = aligned_snapshot
            if latest is not None:
                snapshot, captured_request, received = latest
                snapshot_index = snapshot.metadata["snapshot_index"]
                if (
                    snapshot_index != last_snapshot_index
                    and received <= status_requested_at
                    and time.monotonic() - received < 0.5
                ):
                    origin, alignment = timeline.observation_origin(snapshot.metadata)
                    overlap = actions.next_index + actions.remaining - origin
                    committed = actions.next_index - origin
                    # The configured reserve is an upper bound, not a gate
                    # that permanently excludes O<8 after a missed sample.
                    actual_frozen = min(frozen_steps, overlap)
                    if (
                        0 <= committed <= actual_frozen <= overlap < 16
                        and actual_frozen > 0
                        and origin != last_request_index
                    ):
                        validate_snapshot(snapshot, status)
                        request = planner.request(
                            model_observation(snapshot, prompt),
                            observation_step_index=origin,
                            frozen_steps=actual_frozen,
                            ramp_rate=ramp_rate,
                        )
                        report["requests"].append(
                            {
                                **request,
                                **alignment,
                                "snapshot_index": snapshot_index,
                                "already_committed_steps": committed,
                                "observation_roundtrip_s": received - captured_request,
                                "snapshot_received_age_s": time.monotonic() - received,
                                "bridge_queue_depth": status.get("queue_depth"),
                            }
                        )
                        last_snapshot_index, last_request_index = snapshot_index, origin
                    else:
                        key = (snapshot_index, origin, actions.next_index, overlap)
                        if key != last_skip:
                            report.setdefault("request_skips", []).append(
                                {
                                    "snapshot_index": snapshot_index,
                                    "origin": origin,
                                    "next_index": actions.next_index,
                                    "overlap": overlap,
                                    "committed": committed,
                                    "frozen": actual_frozen,
                                    "reason": "same_origin"
                                    if origin == last_request_index
                                    else "prefix_window",
                                }
                            )
                            last_skip = key

        now = time.monotonic()
        if not finalizing and now >= next_send and status.get("queue_depth", 0) < 2:
            item = actions.pop()
            if item is None:
                if exhaustion_since is None:
                    exhaustion_since = now
                # Give sensor/receipt arrival a bounded chance and let already
                # submitted commands finish; do not drop the final receipt.
                if not planner.pending and now - exhaustion_since >= 2.0:
                    raise RuntimeError("RTC queue exhausted without an applicable prediction")
                if (
                    not report["underruns"]
                    or report["underruns"][-1]["index"] != actions.next_index
                ):
                    report["underruns"].append({"index": actions.next_index, "monotonic_s": now})
            else:
                exhaustion_since = None
                index, target = item
                command_id = f"rtc-{index}-{uuid.uuid4()}"
                row = {
                    "index": index,
                    "command_id": command_id,
                    "pose": target[:7].tolist(),
                    "gripper": float(target[7]),
                    "submitted_monotonic_s": now,
                    "status": "SUBMITTED",
                }
                rows.append(row)
                row_by_id[command_id] = row
                timeline.submitted(index, command_id)
                ack = action_client.request(
                    {
                        "op": "execute_h1_gripper",
                        "command_id": command_id,
                        "timestamp_ns": time.time_ns() + clock_offset_ns,
                        "target_pose": target[:7].tolist(),
                        "target_gripper": float(target[7]),
                    }
                )
                row.update(acknowledgement=ack, acknowledged_monotonic_s=time.monotonic())
                if ack.get("ok") is not True or ack.get("accepted") is not True:
                    row["status"] = "FAILED"
                    raise RuntimeError(f"joint arm/gripper command rejected: {ack}")
                row["status"] = "ACCEPTED"
                progress.observe_targets(target[None], len(report["predictions"]))
                status["queue_depth"] = int(status.get("queue_depth", 0)) + 1
                next_send = max(now + 0.1, time.monotonic())
        time.sleep(0.002)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--observation-port", type=int, default=19100)
    parser.add_argument("--action-port", type=int, default=19200)
    parser.add_argument("--model-port", type=int, default=5565)
    parser.add_argument("--prompt", default=TASK_PROMPT)
    parser.add_argument("--frozen-steps", type=int, default=8)
    parser.add_argument("--request-remaining", type=int, default=10)
    parser.add_argument("--ramp-rate", type=float, default=3.0)
    parser.add_argument("--recording-countdown-s", type=float, default=5.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute or args.confirm != CONFIRMATION:
        parser.error(f"requires --execute --confirm {CONFIRMATION}")
    if not 0 < args.frozen_steps <= args.request_remaining < 16:
        parser.error("require 0 < frozen <= request-remaining < 16")
    if not np.isfinite(args.ramp_rate) or args.ramp_rate <= 0:
        parser.error("ramp-rate must be finite and positive")
    if not np.isfinite(args.recording_countdown_s) or not 0 <= args.recording_countdown_s <= 30:
        parser.error("recording-countdown-s must be in [0,30]")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "g2_groot_rtc_place_live_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "PREPARING",
        "executions": [],
        "requests": [],
        "predictions": [],
        "underruns": [],
    }
    with args.report.open("x") as output:
        try:
            with ExitStack() as stack:

                def obs_factory():
                    return G2LiveObservationClient(
                        "127.0.0.1",
                        args.observation_port,
                        timeout_s=2.0,
                        compute_payload_hashes=False,
                    )

                def model_factory():
                    return PolicyClient(host="127.0.0.1", port=args.model_port, timeout_ms=15000)

                observer_client = stack.enter_context(obs_factory())
                model = stack.enter_context(model_factory())
                capability = model.call_endpoint("get_rtc_config", requires_input=False)
                validate_capability(capability)
                report["rtc_capability"] = capability
                report["standby"] = inspect_standby_bridge("127.0.0.1", args.action_port)
                report["model_preparation"] = {}
                prepare_model(model, observer_client, args.prompt, report["model_preparation"])
                print(
                    json.dumps(
                        {
                            "event": "RECORD_VIDEO_NOW",
                            "seconds_until_activation": args.recording_countdown_s,
                        }
                    ),
                    flush=True,
                )
                time.sleep(args.recording_countdown_s)
                # Outer cleanup closes the action owner BEFORE joining workers.
                workers = stack.enter_context(ExitStack())
                client = stack.enter_context(PlacementBridgeSession("127.0.0.1", args.action_port))
                activation = client.request({"op": "activate", "confirm": ACTIVATION_CONFIRMATION})
                if activation.get("ok") is not True:
                    raise RuntimeError(f"activation failed: {activation}")
                info, status = client.request({"op": "info"}), client.request({"op": "status"})
                validate_activation_state(info, status, "active")
                report["preflight"] = validate_preflight(
                    info, status, TRAINING_REFERENCE, TRAINING_GRIPPER_REFERENCE
                )
                if "active_command_started_monotonic_ns" not in status:
                    raise RuntimeError("RTC requires the bridge timing telemetry extension")
                clock_offset, rtt = calibrate_bridge_clock(client)
                report.update(clock_offset_ns=clock_offset, clock_rtt_ns=rtt)
                snapshot = observer_client.get_snapshot()
                report["seed_snapshot_metadata"] = snapshot.metadata
                predicted, _ = model.get_action(model_observation(snapshot, args.prompt))
                targets = decode_action_chunk(predicted)
                validate_complete_chunk(
                    np.asarray(snapshot.metadata["right_eef_xyz_quaternion_xyzw"]), targets
                )
                actions = ActionQueue()
                actions.initialize(targets)
                report["initial_targets"] = targets.tolist()
                observer_client.__exit__(None, None, None)
                observer = SnapshotWorker(obs_factory)
                workers.callback(observer.close)
                planner = AsyncRtcPlanner(actions, model_factory)
                workers.callback(planner.close)
                report["status"] = "RUNNING"
                print(json.dumps({"event": "FULL_RTC_INFERENCE_STARTED"}), flush=True)
                run_stream(
                    client,
                    actions,
                    planner,
                    observer,
                    PlacementProgress(),
                    report,
                    clock_offset,
                    prompt=args.prompt,
                    frozen_steps=args.frozen_steps,
                    request_remaining=args.request_remaining,
                    ramp_rate=args.ramp_rate,
                )
        except BaseException as error:
            report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "executed_rows": len(report["executions"]),
                "rtc_predictions": len(report["predictions"]),
                "report": str(args.report),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
