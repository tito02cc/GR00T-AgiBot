#!/usr/bin/env python3
"""Guarded model-driven right-arm approach with the gripper forced open.

The runner executes at most two 10 Hz EEF waypoints per fresh observation and
stops before the first model closure intent.  It never sends a closing gripper
target.  On any rejection after motion, it reverses the accepted path.  The
separately confirmed continuous mode retains 10 Hz intermediate tracking gates
but waits for full settling only at a submitted path boundary.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agibot.robot.g2_groot_h1_bridge_client import BridgeSession  # noqa: E402
from agibot.scripts.run_g2_groot_live_model_h2 import (  # noqa: E402
    MAX_OBSERVATION_TO_COMMAND_S,
    MAX_POSE_MISMATCH_DEG,
    MAX_ROTATION_DEG,
    MAX_STEP_M,
    PROMPT,
    SETTLE_TIMEOUT_S,
    TINY_GRIPPER_ACTION_SCHEMA,
    TRAINING_REFERENCE,
    results_by_id,
    rotation_deg,
)
from agibot.tools.g2_gr00t_shadow_adapter import (  # noqa: E402
    build_policy_observation,
    check_shadow_target,
    decode_action_chunk,
)
from agibot.tools.g2_groot_live_observation_client import (  # noqa: E402
    G2LiveObservationClient,
)
from gr00t.policy.server_client import PolicyClient  # noqa: E402


CONFIRMATION = "EXECUTE_G2_GROOT_RECEDING_APPROACH_RIGHT_ARM_OPEN_GRIPPER"
CONTINUOUS_CONFIRMATION = (
    "EXECUTE_G2_GROOT_RECEDING_APPROACH_RIGHT_ARM_OPEN_GRIPPER_CONTINUOUS"
)
LOCAL_PROXY_SCHEMA = "g2_groot_robot_local_slow_path_proxy_v1"
HORIZON = 2
MAX_CYCLES = 80
MAX_FRESHNESS_ATTEMPTS = 10
GRIPPER_OPEN_COMMAND = -0.785
CLOSE_INTENT_THRESHOLD = -0.70
MAX_MODEL_TARGET_STEP_M = 0.035
MAX_MODEL_TARGET_ROTATION_DEG = 3.2
# The G2 controller can briefly lag a stream of consecutive 1 mm waypoints by
# more than its 1.5 mm live-target guard.  The 0.5 mm domain is the fastest
# continuous domain already demonstrated without accumulating that lag.
MAX_PHYSICAL_SUBSTEP_M = 0.0005
MAX_PHYSICAL_SUBSTEP_ROTATION_DEG = 0.25
CONTINUOUS_MAX_PHYSICAL_SUBSTEP_M = 0.0018
CONTINUOUS_MAX_PHYSICAL_SUBSTEP_ROTATION_DEG = 0.5
APPROACH_INITIAL_REFERENCE_M = 0.0002
APPROACH_INITIAL_REFERENCE_DEG = 0.05
APPROACH_POSE_MISMATCH_M = 0.0003
PRE_CLOSE_WORKSPACE_MIN = np.asarray([0.503, -0.311, 1.005])
PRE_CLOSE_WORKSPACE_MAX = np.asarray([0.746, -0.227, 1.081])
RETURN_SETTLE_TIMEOUT_S = 3.0
APPROACH_SETTLE_POSITION_M = 0.00015
APPROACH_SETTLE_ROTATION_RAD = 0.0015


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--observation-port", type=int, default=19100)
    parser.add_argument("--action-port", type=int, default=19200)
    parser.add_argument("--model-port", type=int, default=5564)
    parser.add_argument("--max-waypoints", type=int, default=MAX_CYCLES * HORIZON)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    required_confirmation = CONTINUOUS_CONFIRMATION if args.continuous else CONFIRMATION
    if not args.execute or args.confirm != required_confirmation:
        parser.error(
            f"physical execution requires --execute --confirm {required_confirmation}"
        )
    if args.max_waypoints < 1 or args.max_waypoints > MAX_CYCLES * HORIZON:
        parser.error(f"--max-waypoints must be in 1..{MAX_CYCLES * HORIZON}")
    if args.report is None:
        date = datetime.now().astimezone().strftime("%Y%m%d")
        args.report = REPO_ROOT / f"agibot/reports/g2_groot_receding_approach_{date}.json"
    return args


def save_snapshot_images(snapshot: object, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(directory / "head_color_rgb.png"),
        cv2.cvtColor(snapshot.head_color_rgb, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(directory / "hand_right_rgb.png"),
        cv2.cvtColor(snapshot.hand_right_rgb, cv2.COLOR_RGB2BGR),
    )


def wait_for_ids(
    client: BridgeSession, command_ids: list[str], timeout_s: float
) -> tuple[dict, list[dict | None], bool]:
    started = time.monotonic()
    while True:
        status = client.request({"op": "status"})
        available = results_by_id(status)
        results = [available.get(item) for item in command_ids]
        complete = bool(
            all(results)
            and status.get("ready")
            and status.get("fatal_error") is None
            and status.get("queue_depth") == 0
        )
        settled = bool(
            complete
            and status["live_target_position_error_m"]
            <= APPROACH_SETTLE_POSITION_M
            and status["live_target_rotation_error_rad"]
            <= APPROACH_SETTLE_ROTATION_RAD
        )
        if settled or status.get("fatal_error") or time.monotonic() - started >= timeout_s:
            return status, results, settled
        time.sleep(0.1)


def send_targets(
    client: BridgeSession, targets: list[np.ndarray], prefix: str
) -> tuple[list[str], list[dict]]:
    ids = []
    acknowledgements = []
    deadline = time.monotonic()
    for index, target in enumerate(targets):
        command_id = f"{prefix}-{index}-{uuid.uuid4()}"
        acknowledgement = client.request(
            {
                "op": "execute_h1_gripper",
                "command_id": command_id,
                "timestamp_ns": time.time_ns(),
                "target_pose": target.tolist(),
                "target_gripper": GRIPPER_OPEN_COMMAND,
            }
        )
        if not acknowledgement.get("ok"):
            raise RuntimeError(f"command rejected: {acknowledgement}")
        ids.append(command_id)
        acknowledgements.append(acknowledgement)
        deadline += 0.1
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    return ids, acknowledgements


def interpolate_physical_substeps(
    start: np.ndarray, target: np.ndarray, *, continuous: bool = False
) -> list[np.ndarray]:
    """Slow one training-rate model target into the already validated H1 domain."""

    translation_m = float(np.linalg.norm(target[:3] - start[:3]))
    rotation = rotation_deg(start, target)
    maximum_translation = (
        CONTINUOUS_MAX_PHYSICAL_SUBSTEP_M if continuous else MAX_PHYSICAL_SUBSTEP_M
    )
    maximum_rotation_deg = (
        CONTINUOUS_MAX_PHYSICAL_SUBSTEP_ROTATION_DEG
        if continuous
        else MAX_PHYSICAL_SUBSTEP_ROTATION_DEG
    )
    count = max(
        1,
        int(
            np.ceil(
                max(
                    translation_m / maximum_translation,
                    rotation / maximum_rotation_deg,
                )
            )
        ),
    )
    first_q = start[3:7].copy()
    final_q = target[3:7].copy()
    if float(np.dot(first_q, final_q)) < 0.0:
        final_q = -final_q
    result = []
    for index in range(1, count + 1):
        alpha = index / count
        item = np.empty(7, dtype=np.float64)
        item[:3] = start[:3] * (1.0 - alpha) + target[:3] * alpha
        quaternion = first_q * (1.0 - alpha) + final_q * alpha
        item[3:7] = quaternion / np.linalg.norm(quaternion)
        result.append(item)
    return result


def execute_model_targets_slowly(
    client: BridgeSession,
    start: np.ndarray,
    model_targets: list[np.ndarray],
    prefix: str,
    *,
    continuous: bool = False,
) -> tuple[dict, list[np.ndarray], bool]:
    physical_targets: list[np.ndarray] = []
    previous = start.copy()
    model_mapping = []
    for model_index, target in enumerate(model_targets):
        segments = interpolate_physical_substeps(
            previous, target, continuous=continuous
        )
        physical_targets.extend(segments)
        model_mapping.append(
            {
                "model_target": model_index,
                "physical_substeps": len(segments),
                "target_pose": target.tolist(),
            }
        )
        previous = target

    command_id = f"{prefix}-{uuid.uuid4()}"
    acknowledgement = client.request(
        {
            "op": "execute_path",
            "command_id": command_id,
            "timestamp_ns": time.time_ns(),
            "path": [item.tolist() for item in physical_targets],
            "target_gripper": GRIPPER_OPEN_COMMAND,
        }
    )
    if not acknowledgement.get("ok"):
        return (
            {
                "model_targets": model_mapping,
                "physical_substeps_planned": len(physical_targets),
                "physical_substeps_completed": 0,
                "acknowledgement": acknowledgement,
                "settled": False,
                "fatal_error": f"proxy rejected path: {acknowledgement}",
            },
            [],
            False,
        )
    deadline = time.monotonic() + max(10.0, len(physical_targets) * 1.5)
    final_status = {}
    while time.monotonic() < deadline:
        final_status = client.request({"op": "status"})
        if final_status.get("fatal_error"):
            break
        result = final_status.get("last_result") or {}
        if result.get("command_id") == command_id and not final_status.get("busy"):
            settled = result.get("status") == "PASS"
            completed = (
                [item.copy() for item in physical_targets] if settled else []
            )
            underlying = final_status.get("underlying_status") or {}
            return (
                {
                    "model_targets": model_mapping,
                    "physical_substeps_planned": len(physical_targets),
                    "physical_substeps_completed": len(completed),
                    "acknowledgement": acknowledgement,
                    "proxy_result": result,
                    "settled": settled,
                    "settled_position_error_m": underlying.get(
                        "live_target_position_error_m"
                    ),
                    "settled_rotation_error_rad": underlying.get(
                        "live_target_rotation_error_rad"
                    ),
                    "fatal_error": final_status.get("fatal_error"),
                },
                completed,
                settled,
            )
        time.sleep(0.1)
    settled = False
    completed = []
    return (
        {
            "model_targets": model_mapping,
            "physical_substeps_planned": len(physical_targets),
            "physical_substeps_completed": 0,
            "acknowledgement": acknowledgement,
            "settled": settled,
            "fatal_error": final_status.get("fatal_error") or "proxy path timeout",
        },
        completed,
        settled,
    )


def reverse_path(
    client: BridgeSession, origin: np.ndarray, accepted: list[np.ndarray]
) -> dict:
    command_id = f"approach-return-{uuid.uuid4()}"
    acknowledgement = client.request(
        {"op": "return_to_origin", "command_id": command_id}
    )
    if not acknowledgement.get("ok"):
        return {"waypoints": len(accepted), "settled": False, "fatal_error": str(acknowledgement)}
    deadline = time.monotonic() + max(15.0, len(accepted) * 1.5)
    status = {}
    settled = False
    result = None
    while time.monotonic() < deadline:
        status = client.request({"op": "status"})
        result = status.get("last_result") or {}
        if status.get("fatal_error"):
            break
        if result.get("command_id") == command_id and not status.get("busy"):
            settled = result.get("status") == "PASS"
            break
        time.sleep(0.1)
    underlying = status.get("underlying_status") or {}
    return {
        "waypoints": len(accepted),
        "acknowledgement": acknowledgement,
        "proxy_result": result,
        "settled": settled,
        "final_position_error_m": underlying.get("live_target_position_error_m"),
        "final_rotation_error_rad": underlying.get("live_target_rotation_error_rad"),
        "fatal_error": status.get("fatal_error"),
    }


def main() -> int:
    args = parse_args()
    report = {
        "schema": "g2_groot_receding_approach_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "STARTED",
        "scope": "right arm H2 receding approach; right gripper forced open",
        "cycles": [],
        "accepted_waypoints": 0,
        "accepted_model_targets": 0,
        "gripper_close_commands": 0,
        "maximum_waypoints": args.max_waypoints,
    }
    exit_code = 1
    accepted: list[np.ndarray] = []
    accepted_model_targets = 0
    origin: np.ndarray | None = None
    with (
        G2LiveObservationClient("127.0.0.1", args.observation_port) as obs_client,
        PolicyClient(host="127.0.0.1", port=args.model_port, timeout_ms=60000) as model_client,
        BridgeSession("127.0.0.1", args.action_port) as action_client,
    ):
        obs_info = obs_client.get_info()
        action_info = action_client.request({"op": "info"})
        initial_status = action_client.request({"op": "status"})
        if obs_info.get("control_api_exposed") is not False:
            raise RuntimeError("observation bridge is not read-only")
        if action_info.get("schema") != LOCAL_PROXY_SCHEMA:
            raise RuntimeError("unexpected action schema")
        if action_info.get("right_gripper_forced_open") is not True:
            raise RuntimeError("local proxy does not force the gripper open")
        if action_info.get("continuous_10hz") is not args.continuous:
            raise RuntimeError("local proxy continuous mode does not match runner")
        if action_info.get("left_arm_enabled") is not False or action_info.get(
            "head_waist_chassis_enabled"
        ) is not False:
            raise RuntimeError("approach bridge exposes a forbidden body group")
        maximum_proxy_substep = 0.00181 if args.continuous else 0.00101
        if float(action_info.get("maximum_substep_m", np.inf)) > maximum_proxy_substep:
            raise RuntimeError("physical translation substep limit is too large")
        if not initial_status.get("ready") or initial_status.get("fatal_error"):
            raise RuntimeError("action bridge is not ready")
        if not model_client.ping():
            raise RuntimeError("model server ping failed")
        initial_underlying = initial_status["underlying_status"]
        origin = np.asarray(initial_underlying["desired_pose"], dtype=np.float64)

        warm = obs_client.get_snapshot()
        warm_pose = np.asarray(
            warm.metadata["right_eef_xyz_quaternion_xyzw"], dtype=np.float64
        )
        warm_gripper = float(warm.metadata["right_gripper"]["training_position"])
        warm_started = time.monotonic()
        model_client.get_action(
            build_policy_observation(
                warm.head_color_rgb,
                warm.hand_right_rgb,
                warm_pose,
                warm_gripper,
                PROMPT,
            )
        )
        report["warmup_inference_s"] = time.monotonic() - warm_started

        stop_snapshot = None
        stop_targets = None
        for cycle in range(MAX_CYCLES):
            if accepted_model_targets >= args.max_waypoints:
                report["status"] = "PASS_BOUNDED_APPROACH_STAGE"
                exit_code = 0
                break
            acquisition_attempts = []
            for acquisition_attempt in range(MAX_FRESHNESS_ATTEMPTS):
                cycle_started = time.monotonic()
                snapshot = obs_client.get_snapshot()
                pose = np.asarray(
                    snapshot.metadata["right_eef_xyz_quaternion_xyzw"],
                    dtype=np.float64,
                )
                gripper = float(
                    snapshot.metadata["right_gripper"]["training_position"]
                )
                action, _ = model_client.get_action(
                    build_policy_observation(
                        snapshot.head_color_rgb,
                        snapshot.hand_right_rgb,
                        pose,
                        gripper,
                        PROMPT,
                    )
                )
                remaining = args.max_waypoints - accepted_model_targets
                action_chunk = decode_action_chunk(action).astype(np.float64)
                targets = action_chunk[: min(HORIZON, remaining)]
                elapsed_s = time.monotonic() - cycle_started
                status = action_client.request({"op": "status"})
                live = np.asarray(
                    status["underlying_status"]["live_pose"], dtype=np.float64
                )
                mismatch_m = float(np.linalg.norm(live[:3] - pose[:3]))
                mismatch_deg = rotation_deg(live, pose)
                input_accepted = bool(
                    mismatch_m <= APPROACH_POSE_MISMATCH_M
                    and mismatch_deg <= MAX_POSE_MISMATCH_DEG
                    and status.get("ready")
                    and not status.get("fatal_error")
                )
                acquisition_attempts.append(
                    {
                        "attempt": acquisition_attempt,
                        "snapshot_index": int(snapshot.metadata["snapshot_index"]),
                        "observation_to_command_s": elapsed_s,
                        "pose_mismatch_m": mismatch_m,
                        "pose_mismatch_deg": mismatch_deg,
                        "accepted": input_accepted,
                    }
                )
                if input_accepted:
                    break
                # Discard stale or cross-bridge-misaligned actions and acquire
                # a new synchronized input while the robot remains stationary.
                # The status request above also refreshes the local watchdog.
                if status.get("fatal_error") or not status.get("ready"):
                    break
            violations = []
            metrics = []
            mismatch_m = float(np.linalg.norm(live[:3] - pose[:3]))
            mismatch_deg = rotation_deg(live, pose)
            if mismatch_m > APPROACH_POSE_MISMATCH_M:
                violations.append("observation_translation_mismatch")
            if mismatch_deg > MAX_POSE_MISMATCH_DEG:
                violations.append("observation_rotation_mismatch")
            previous = pose
            for index, target in enumerate(targets):
                check = check_shadow_target(previous, target)
                if check.translation_step_m > MAX_MODEL_TARGET_STEP_M:
                    violations.append(f"step_{index}_translation")
                if check.rotation_step_deg > MAX_MODEL_TARGET_ROTATION_DEG:
                    violations.append(f"step_{index}_rotation")
                if np.any(target[:3] < PRE_CLOSE_WORKSPACE_MIN) or np.any(
                    target[:3] > PRE_CLOSE_WORKSPACE_MAX
                ):
                    violations.append(f"step_{index}_preclose_workspace")
                metrics.append(
                    {
                        "step": index,
                        "translation_m": check.translation_step_m,
                        "rotation_deg": check.rotation_step_deg,
                        "gripper": float(target[7]),
                        "xyz": target[:3].tolist(),
                    }
                )
                previous = target[:7]
            close_intent = bool(np.max(targets[:, 7]) > CLOSE_INTENT_THRESHOLD)
            full_close_indices = np.flatnonzero(
                action_chunk[:, 7] > CLOSE_INTENT_THRESHOLD
            )
            row = {
                "cycle": cycle,
                "snapshot_index": int(snapshot.metadata["snapshot_index"]),
                "observation_to_command_s": elapsed_s,
                "acquisition_attempts": acquisition_attempts,
                "pose_mismatch_m": mismatch_m,
                "pose_mismatch_deg": mismatch_deg,
                "metrics": metrics,
                "close_intent": close_intent,
                "full_horizon_gripper": action_chunk[:, 7].tolist(),
                "full_horizon_targets": action_chunk.tolist(),
                "full_horizon_close_intent": bool(full_close_indices.size),
                "full_horizon_first_close_index": (
                    int(full_close_indices[0]) if full_close_indices.size else None
                ),
                "violations": violations,
            }
            report["cycles"].append(row)
            if violations:
                report["status"] = "REJECTED_DURING_APPROACH"
                break
            if close_intent:
                stop_snapshot = snapshot
                stop_targets = targets
                report["status"] = "PASS_STOPPED_BEFORE_CLOSE_INTENT"
                exit_code = 0
                break
            outbound = [target[:7].copy() for target in targets]
            # The model action is checked relative to the measured observation,
            # but the physical path must be continuous with the controller's
            # last commanded target.  A small settled tracking residual must not
            # be added to an otherwise 1 mm first substep.
            controller_start = np.asarray(
                status["underlying_status"]["desired_pose"], dtype=np.float64
            )
            execution, completed, settled = execute_model_targets_slowly(
                action_client,
                controller_start,
                outbound,
                f"approach-c{cycle}",
                continuous=args.continuous,
            )
            row["execution"] = execution
            accepted.extend(completed)
            report["accepted_waypoints"] = len(accepted)
            if not settled or execution.get("fatal_error"):
                report["status"] = "EXECUTION_FAILED_DURING_APPROACH"
                break
            accepted_model_targets += len(outbound)
            report["accepted_model_targets"] = accepted_model_targets
        else:
            report["status"] = "MAX_CYCLES_WITHOUT_CLOSE_INTENT"

        if report["status"] in {
            "PASS_STOPPED_BEFORE_CLOSE_INTENT",
            "PASS_STOPPED_BEFORE_FUTURE_CLOSE_INTENT",
        }:
            assert stop_snapshot is not None and stop_targets is not None
            image_dir = args.report.with_suffix("")
            save_snapshot_images(stop_snapshot, image_dir)
            report["close_gate"] = {
                "current_pose": stop_snapshot.metadata[
                    "right_eef_xyz_quaternion_xyzw"
                ],
                "current_gripper": stop_snapshot.metadata["right_gripper"],
                "predicted_targets": stop_targets.tolist(),
                "images": {
                    "head": str(image_dir / "head_color_rgb.png"),
                    "wrist": str(image_dir / "hand_right_rgb.png"),
                },
            }
        elif accepted and origin is not None:
            report["automatic_return"] = reverse_path(action_client, origin, accepted)

        final_status = action_client.request({"op": "status"})
        report["final_bridge_status"] = final_status
        final_underlying = final_status.get("underlying_status") or {}
        gripper_status = final_underlying.get("right_gripper") or {}
        if int(gripper_status.get("command_count", -1)) != 0:
            report["status"] = "FAIL_GRIPPER_HARDWARE_COMMAND_OBSERVED"
            exit_code = 1
        report["gripper_hardware_commands"] = int(
            gripper_status.get("command_count", -1)
        )
        shutdown_id = f"approach-shutdown-{uuid.uuid4()}"
        shutdown = action_client.request(
            {"op": "shutdown_hold", "command_id": shutdown_id}
        )
        report["shutdown_hold"] = shutdown
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
