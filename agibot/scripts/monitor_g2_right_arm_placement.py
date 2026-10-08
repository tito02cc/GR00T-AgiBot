#!/usr/bin/env python3
"""Read-only placement monitor for the G2 (10.20.15.60) right arm.

Purpose
-------
Field staff reposition the right arm with Agibot's own teach/homing tools.
This program does NOT move the robot.  It only reads the read-only
observation bridge and prints, at ~1 Hz, how far the live end-effector is
from the grasp training start and whether the two hard gates that block
`recover_g2_groot_training_start.py` would now pass.

It commands no motion.  It opens no action bridge.  It sends no motor
command.  It refuses to run at all if the bridge advertises any control
capability.

Gates checked (constants read from the robot's own controller,
/home/agi/vla_ct/groot_right_arm_shadow/g2_groot_persistent_right_arm_controller.py:49-70)
  * WORKSPACE_MIN / WORKSPACE_MAX -- the live pose must be inside the cell
    box, in particular z >= 0.987 m.
  * APPROACH_SESSION_MAX_TRANSLATION_M = 0.240 m -- distance from the live
    pose to the training-start median must not exceed this.
  * APPROACH_SESSION_MAX_ROTATION_RAD = 20 deg -- rotation error likewise.

Exit status
  0  all gates pass -> the sanctioned fine-approach script may be run next
  1  at least one gate still fails
  2  refused: bridge is not read-only, or a joint/body fault is present

Usage (on 10.20.15.170, with a tunnel 19100 -> robot:9100)
  cd ~/ct/Isaac-GR00T
  .venv/bin/python3.12 agibot/scripts/monitor_g2_right_arm_placement.py \
      --host 127.0.0.1 --port 19100 \
      --training-dataset agibot/gr00t_data/xichong_right_single_grasp_300
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agibot.tools.g2_gr00t_shadow_adapter import (  # noqa: E402
    make_right_eef_state,
    rot6d_to_quaternion_xyzw,
)
from agibot.tools.g2_groot_live_observation_client import (  # noqa: E402
    G2LiveObservationClient,
)

# --- guards, transcribed from the robot controller; do not "tune" these -----
WORKSPACE_MIN = np.asarray([0.448, -0.333, 0.987])
WORKSPACE_MAX = np.asarray([0.754, -0.197, 1.217])
APPROACH_MAX_TRANSLATION_M = 0.240
APPROACH_MAX_ROTATION_DEG = 20.0

# Margin so the operator stops clear of the limit instead of exactly on it.
RECOMMENDED_TRANSLATION_M = 0.200
RECOMMENDED_Z_CLEARANCE_M = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19100)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--training-dataset",
        type=Path,
        required=True,
        help="converted grasp dataset, used for the training-start reference",
    )
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument(
        "--once",
        action="store_true",
        help="print a single reading and exit instead of monitoring",
    )
    parser.add_argument(
        "--hold-passes",
        type=int,
        default=3,
        help="consecutive passing readings required before declaring success",
    )
    return parser.parse_args()


def load_training_reference(dataset: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (xyz_median, quaternions_xyzw, gripper_median) over episode starts."""
    starts = []
    for path in sorted(dataset.resolve().glob("data/chunk-*/*.parquet")):
        frame = pd.read_parquet(path, columns=["observation.state"])
        starts.append(np.asarray(frame["observation.state"].iloc[0], dtype=np.float64))
    if not starts:
        raise FileNotFoundError(f"no training episodes under {dataset}")
    array = np.stack(starts)
    if array.shape[1] != 10 or not np.isfinite(array).all():
        raise ValueError(f"unexpected training state shape {array.shape}")
    quaternions = np.stack([rot6d_to_quaternion_xyzw(row[3:9]) for row in array])
    return (
        np.median(array[:, :3], axis=0),
        quaternions,
        float(np.median(array[:, 9])),
    )


def assert_read_only(info: dict) -> None:
    """Refuse to proceed unless the bridge proves it cannot actuate."""
    violations = []
    if info.get("control_api_exposed") is not False:
        violations.append(f"control_api_exposed={info.get('control_api_exposed')}")
    if info.get("live_execution_enabled") is not False:
        violations.append(f"live_execution_enabled={info.get('live_execution_enabled')}")
    sent = int(info.get("motor_commands_sent", -1))
    if sent != 0:
        violations.append(f"motor_commands_sent={sent}")
    if violations:
        print("REFUSED: bridge is not in read-only mode -> " + ", ".join(violations), file=sys.stderr)
        print("This monitor only attaches to the read-only observation bridge.", file=sys.stderr)
        raise SystemExit(2)


def evaluate(metadata: dict, xyz_median: np.ndarray, quaternions: np.ndarray) -> dict:
    pose = np.asarray(metadata["right_eef_xyz_quaternion_xyzw"], dtype=np.float64)
    gripper = float(metadata["right_gripper"]["training_position"])
    state = make_right_eef_state(pose)
    xyz = state[:3]

    distance_m = float(np.linalg.norm(xyz - xyz_median))
    live_rotation = Rotation.from_quat(pose[3:7])
    rotation_deg = float(
        np.median(np.degrees((Rotation.from_quat(quaternions).inv() * live_rotation).magnitude()))
    )

    inside_box = bool(np.all(xyz >= WORKSPACE_MIN) and np.all(xyz <= WORKSPACE_MAX))
    # Signed per-axis slack: negative means the axis is outside the box.
    low_slack = xyz - WORKSPACE_MIN
    high_slack = WORKSPACE_MAX - xyz

    gates = {
        "workspace_box": inside_box,
        "approach_translation": distance_m <= APPROACH_MAX_TRANSLATION_M,
        "approach_rotation": rotation_deg <= APPROACH_MAX_ROTATION_DEG,
    }

    faults = [
        item
        for item in metadata.get("right_joint_health", {}).get("joints", [])
        if int(item.get("error_code", 0)) != 0
    ]
    body = metadata.get("right_body_health", {})

    return {
        "xyz": xyz,
        "gripper": gripper,
        "distance_m": distance_m,
        "rotation_deg": rotation_deg,
        "low_slack": low_slack,
        "high_slack": high_slack,
        "gates": gates,
        "all_pass": all(gates.values()) and not faults,
        "faults": faults,
        "body": body,
    }


def mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def render(result: dict, xyz_median: np.ndarray) -> str:
    xyz = result["xyz"]
    delta = xyz - xyz_median
    lines = [
        f"[{datetime.now():%H:%M:%S}] live xyz = "
        f"[{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]  gripper = {result['gripper']:+.4f}",
        f"  target (training median) = [{xyz_median[0]:.4f}, {xyz_median[1]:.4f}, {xyz_median[2]:.4f}]",
        f"  remaining dx/dy/dz (mm)  = "
        f"[{-delta[0]*1000:+8.1f}, {-delta[1]*1000:+8.1f}, {-delta[2]*1000:+8.1f}]"
        "   <- move the EEF by these amounts",
        f"  {mark(result['gates']['approach_translation'])} distance      "
        f"{result['distance_m']*1000:8.1f} mm  (limit {APPROACH_MAX_TRANSLATION_M*1000:.0f}, "
        f"aim below {RECOMMENDED_TRANSLATION_M*1000:.0f})",
        f"  {mark(result['gates']['approach_rotation'])} rotation      "
        f"{result['rotation_deg']:8.2f} deg (limit {APPROACH_MAX_ROTATION_DEG:.0f})",
        f"  {mark(result['gates']['workspace_box'])} workspace box "
        f"z = {xyz[2]:.4f} m  (floor {WORKSPACE_MIN[2]:.3f}, "
        f"clearance {result['low_slack'][2]*1000:+.1f} mm)",
    ]
    if not result["gates"]["workspace_box"]:
        axes = "xyz"
        for index, axis in enumerate(axes):
            if result["low_slack"][index] < 0:
                lines.append(
                    f"      {axis} is {abs(result['low_slack'][index])*1000:.1f} mm BELOW "
                    f"min {WORKSPACE_MIN[index]:.3f}"
                )
            if result["high_slack"][index] < 0:
                lines.append(
                    f"      {axis} is {abs(result['high_slack'][index])*1000:.1f} mm ABOVE "
                    f"max {WORKSPACE_MAX[index]:.3f}"
                )
    if result["faults"]:
        names = ", ".join(
            f"{item.get('name', '?')}={item.get('error_code')}" for item in result["faults"]
        )
        lines.append(f"  FAULT joint error codes present: {names}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    xyz_median, quaternions, gripper_median = load_training_reference(args.training_dataset)

    print("G2 right-arm placement monitor -- READ ONLY, commands no motion")
    print(f"bridge {args.host}:{args.port}")
    print(
        f"training-start median xyz = [{xyz_median[0]:.4f}, {xyz_median[1]:.4f}, "
        f"{xyz_median[2]:.4f}], gripper {gripper_median:+.4f}"
    )
    print(
        "Move the arm with Agibot's teach/homing tools only. This program will not "
        "move it.\nPress Ctrl-C to stop.\n"
    )

    consecutive = 0
    try:
        with G2LiveObservationClient(
            args.host, args.port, args.timeout_s, compute_payload_hashes=False
        ) as client:
            assert_read_only(client.get_info())
            while True:
                result = evaluate(client.get_snapshot().metadata, xyz_median, quaternions)
                print(render(result, xyz_median), flush=True)

                if result["all_pass"]:
                    consecutive += 1
                    margin_ok = (
                        result["distance_m"] <= RECOMMENDED_TRANSLATION_M
                        and result["low_slack"][2] >= RECOMMENDED_Z_CLEARANCE_M
                    )
                    print(
                        f"  -> all gates PASS ({consecutive}/{args.hold_passes})"
                        + ("" if margin_ok else "  [on the limit; prefer more margin]"),
                        flush=True,
                    )
                    if args.once or consecutive >= args.hold_passes:
                        print(
                            "\nPLACEMENT_OK -- gates satisfied.\n"
                            "Next step is still a separate, explicitly approved operation:\n"
                            "  recover_g2_groot_training_start.py --execute --expanded-session\n"
                            "Also compare head_color against a training frame before any "
                            "inference: the pose gates cannot detect a moved chassis."
                        )
                        return 0
                else:
                    consecutive = 0
                    if args.once:
                        print("\nPLACEMENT_NOT_READY -- see failing gates above.")
                        return 1
                print("", flush=True)
                time.sleep(max(0.2, args.interval_s))
    except KeyboardInterrupt:
        print("\nstopped by operator", flush=True)
        return 1
    except (ConnectionRefusedError, ConnectionError, TimeoutError, OSError) as error:
        print(
            f"\nCANNOT REACH the observation bridge at {args.host}:{args.port} -- {error}\n"
            "\nCheck, in this order:\n"
            "  1. the read-only observation bridge is running on the robot (.60),\n"
            "     listening on 127.0.0.1:9100\n"
            f"  2. an SSH tunnel maps this host's port {args.port} -> robot 127.0.0.1:9100\n"
            "     ssh -N -L 19100:127.0.0.1:9100 agi@10.20.15.60\n"
            "  3. --host/--port match that tunnel\n"
            "\nNothing was sent to the robot.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
