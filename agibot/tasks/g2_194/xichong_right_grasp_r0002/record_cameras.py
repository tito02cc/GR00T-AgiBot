#!/usr/bin/env python3
"""Record the G2 head/right-wrist source JPEG streams without robot commands.

Run on the robot after sourcing /home/agi/app/env.sh. Start before the action
runner and stop after it exits. The recorder writes on the robot so recording
does not add image traffic to the policy's SSH tunnel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time

import agibot_gdk as gdk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=0.0,
                        help="0 records until SIGINT/SIGTERM; intended for full inference")
    args = parser.parse_args()
    if not 0 < args.hz <= 30 or args.duration_s < 0:
        parser.error("require 0 < hz <= 30 and duration-s >= 0")
    return args


def wait_for_cameras(camera: object, timeout_s: float = 15.0) -> None:
    """Let GDK camera subscriptions initialize before the recording begins."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            head = camera.get_latest_image(gdk.CameraType.kHeadColor, 500)
            camera.get_nearest_image(
                gdk.CameraType.kHandRightColor, int(head.timestamp_ns), 500
            )
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise RuntimeError("head/right-wrist cameras unavailable after warmup")
            time.sleep(0.25)


def capture(camera: object, output: Path, *, hz: float, duration_s: float) -> dict:
    period = 1.0 / hz
    started_wall_ns = time.time_ns()
    started = time.monotonic()
    deadline = started + duration_s if duration_s else None
    next_tick = started
    frame_count = 0
    duplicates = 0
    last_head_timestamp = None
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with (output / "frames.jsonl").open("x", buffering=1) as manifest:
        while not stop and (deadline is None or time.monotonic() < deadline):
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            if stop:
                break
            sample_wall_ns = time.time_ns()
            head = camera.get_latest_image(gdk.CameraType.kHeadColor, 500)
            wrist = camera.get_nearest_image(
                gdk.CameraType.kHandRightColor, int(head.timestamp_ns), 500
            )
            head_ns, wrist_ns = int(head.timestamp_ns), int(wrist.timestamp_ns)
            if head_ns == last_head_timestamp:
                duplicates += 1
            else:
                head_data, wrist_data = bytes(head.data), bytes(wrist.data)
                if not head_data.startswith(b"\xff\xd8") or not wrist_data.startswith(b"\xff\xd8"):
                    raise RuntimeError("camera did not return source JPEG; no silent re-encode")
                name = f"{frame_count:06d}.jpg"
                (output / "head" / name).write_bytes(head_data)
                (output / "hand_right" / name).write_bytes(wrist_data)
                manifest.write(json.dumps({
                    "index": frame_count,
                    "name": name,
                    "sample_wall_ns": sample_wall_ns,
                    "sample_monotonic_ns": time.monotonic_ns(),
                    "head_timestamp_ns": head_ns,
                    "hand_right_timestamp_ns": wrist_ns,
                    "camera_skew_ms": abs(head_ns - wrist_ns) / 1e6,
                    "head_bytes": len(head_data),
                    "hand_right_bytes": len(wrist_data),
                }, separators=(",", ":")) + "\n")
                frame_count += 1
                last_head_timestamp = head_ns
            next_tick += period
            if next_tick < time.monotonic():
                next_tick = time.monotonic()
    return {
        "started_wall_ns": started_wall_ns,
        "finished_wall_ns": time.time_ns(),
        "requested_hz": hz,
        "frames": frame_count,
        "duplicate_head_samples_skipped": duplicates,
        "control_commands_sent": 0,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "head").mkdir()
    (args.output_dir / "hand_right").mkdir()
    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK init failed")
    try:
        time.sleep(2)
        camera = gdk.Camera()
        wait_for_cameras(camera)
        result = capture(camera, args.output_dir, hz=args.hz,
                         duration_s=args.duration_s)
    finally:
        gdk.gdk_release()
    (args.output_dir / "recording.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
