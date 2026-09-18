#!/usr/bin/env python3
"""Guarded deterministic recovery of the G2 right EEF to the training start."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from agibot.robot.g2_groot_h1_bridge_client import BridgeSession  # noqa: E402
except ModuleNotFoundError:
    # Robot deployment keeps the two audited files in one flat directory.
    from g2_groot_h1_bridge_client import BridgeSession  # type: ignore[no-redef]


CONFIRMATION = "RECOVER_G2_GROOT_RIGHT_ARM_TO_TRAINING_START"
EXPANDED_CONFIRMATION = (
    "RECOVER_G2_GROOT_RIGHT_ARM_FROM_APPROACH_TO_TRAINING_START"
)
MAX_RECOVERY_DISTANCE_M = 0.004
MAX_RECOVERY_ROTATION_DEG = 2.0
MAX_SUBSTEP_M = 0.005
MAX_SUBSTEP_ROTATION_DEG = 1.0
SETTLE_POSITION_M = 0.0005
SETTLE_ROTATION_RAD = 0.002
TRAINING_REFERENCE = np.asarray(
    [
        0.5081030780841185,
        -0.25893071977192195,
        1.04842646506165,
        0.6620299522145109,
        -0.020616324660443348,
        0.7490960119476248,
        0.01210266138135472,
    ],
    dtype=np.float64,
)


def results_by_id(status: dict) -> dict[str, dict]:
    return {
        str(item["command_id"]): item
        for item in status.get("recent_results", [])
    }


def rotation_deg(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(first[3:7], second[3:7])))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))))


class SafeShutdownBridgeSession(BridgeSession):
    def __exit__(self, *exc_info):
        try:
            self.request({"op": "shutdown"})
        except Exception:
            pass
        return super().__exit__(*exc_info)


def interpolate(start: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    distance = float(np.linalg.norm(target[:3] - start[:3]))
    angle = rotation_deg(start, target)
    count = max(
        1,
        int(
            np.ceil(
                max(distance / MAX_SUBSTEP_M, angle / MAX_SUBSTEP_ROTATION_DEG)
            )
        ),
    )
    q0 = start[3:7].copy()
    q1 = target[3:7].copy()
    if float(np.dot(q0, q1)) < 0:
        q1 = -q1
    waypoints = []
    for index in range(1, count + 1):
        alpha = index / count
        item = np.empty(7, dtype=np.float64)
        item[:3] = start[:3] * (1 - alpha) + target[:3] * alpha
        quaternion = q0 * (1 - alpha) + q1 * alpha
        item[3:7] = quaternion / np.linalg.norm(quaternion)
        waypoints.append(item)
    return waypoints


def wait_settled(client: BridgeSession, command_id: str) -> tuple[dict, dict]:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        status = client.request({"op": "status"})
        result = results_by_id(status).get(command_id)
        if status.get("fatal_error"):
            raise RuntimeError(status["fatal_error"])
        if (
            result
            and status.get("queue_depth") == 0
            and status.get("live_target_position_error_m") <= SETTLE_POSITION_M
            and status.get("live_target_rotation_error_rad") <= SETTLE_ROTATION_RAD
        ):
            return status, result
        time.sleep(0.1)
    raise TimeoutError(f"recovery waypoint did not settle: {command_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expanded-session", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--port", type=int, default=19200)
    parser.add_argument(
        "--target-pose",
        type=float,
        nargs=7,
        default=TRAINING_REFERENCE.tolist(),
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            REPO_ROOT
            / "agibot/reports"
            / f"g2_groot_training_start_recovery_{datetime.now().astimezone():%Y%m%d}.json"
        ),
    )
    args = parser.parse_args()
    required_confirmation = (
        EXPANDED_CONFIRMATION if args.expanded_session else CONFIRMATION
    )
    if not args.execute or args.confirm != required_confirmation:
        parser.error(
            "physical recovery requires --execute --confirm "
            f"{required_confirmation}"
        )
    target = np.asarray(args.target_pose, dtype=np.float64)
    quaternion_norm = float(np.linalg.norm(target[3:7]))
    if not np.isfinite(target).all() or not 0.999 <= quaternion_norm <= 1.001:
        raise ValueError("--target-pose must be finite XYZ plus normalized XYZW")

    report = {
        "schema": "g2_groot_training_start_recovery_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "target": target.tolist(),
        "status": "STARTED",
        "waypoints": [],
    }
    with SafeShutdownBridgeSession("127.0.0.1", args.port) as client:
        info = client.request({"op": "info"})
        status = client.request({"op": "status"})
        if not status.get("ready") or status.get("fatal_error"):
            raise RuntimeError("recovery bridge is not ready")
        session_limit = float(
            info.get("right_session_max_translation_m", np.inf)
        )
        if args.expanded_session:
            if not 0.239 <= session_limit <= 0.241:
                raise RuntimeError("expanded recovery requires the 0.24 m session")
        elif session_limit > 0.0041:
            raise RuntimeError("recovery requires the standard 4 mm session")
        start = np.asarray(status["live_pose"], dtype=np.float64)
        distance = float(np.linalg.norm(target[:3] - start[:3]))
        angle = rotation_deg(start, target)
        report["start"] = start.tolist()
        report["start_error_m"] = distance
        report["start_error_deg"] = angle
        maximum_distance = 0.24 if args.expanded_session else MAX_RECOVERY_DISTANCE_M
        maximum_rotation = 20.0 if args.expanded_session else MAX_RECOVERY_ROTATION_DEG
        if distance > maximum_distance or angle > maximum_rotation:
            raise RuntimeError("current pose is outside the guarded recovery envelope")
        waypoints = interpolate(start, target)
        last_command_id = ""
        deadline = time.monotonic()
        for index, waypoint in enumerate(waypoints):
            acknowledgement = None
            command_id = ""
            for stale_attempt in range(3):
                command_id = (
                    f"training-start-recovery-{index}-retry{stale_attempt}-"
                    f"{uuid.uuid4()}"
                )
                acknowledgement = client.request(
                    {
                        "op": "execute_h1_gripper",
                        "command_id": command_id,
                        # Robot and workstation wall clocks can differ by a few
                        # seconds; the live socket and per-waypoint settling
                        # checks remain authoritative for this deterministic
                        # recovery path.
                        "timestamp_ns": time.time_ns() + 5_000_000_000,
                        "target_pose": waypoint.tolist(),
                        "target_gripper": -0.785,
                    }
                )
                if acknowledgement.get("ok"):
                    break
                message = str(acknowledgement.get("message", ""))
                if "command age" not in message or "exceeds" not in message:
                    break
                time.sleep(0.1)
            assert acknowledgement is not None
            if not acknowledgement.get("ok"):
                raise RuntimeError(f"recovery command rejected: {acknowledgement}")
            report["waypoints"].append(
                {
                    "target": waypoint.tolist(),
                    "acknowledgement": acknowledgement,
                }
            )
            last_command_id = command_id
            deadline += 0.1
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        settled, result = wait_settled(client, last_command_id)
        report["final_command_result"] = result
        report["settled_position_error_m"] = settled["live_target_position_error_m"]
        report["settled_rotation_error_rad"] = settled["live_target_rotation_error_rad"]
        final = client.request({"op": "status"})
        report["final"] = final
        final_pose = np.asarray(final["live_pose"], dtype=np.float64)
        report["final_error_m"] = float(
            np.linalg.norm(final_pose[:3] - target[:3])
        )
        report["final_error_deg"] = rotation_deg(final_pose, target)
        if abs(float((final.get("right_gripper") or {}).get("last_completed", 0.0)) + 0.785) > 0.02:
            raise RuntimeError("right gripper did not return open during recovery")
        report["status"] = "PASS_RECOVERED_TO_TRAINING_START"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
