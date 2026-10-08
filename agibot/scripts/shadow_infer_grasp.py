#!/usr/bin/env python3
"""SHADOW inference for the G2 grasp policy -- decodes actions, sends nothing.

What this does
--------------
1. Reads ONE live observation from the read-only observation bridge
   (head_color + hand_right + right_eef pose + right_gripper).
2. Builds the strict (B=1,T=1) policy observation with the exact training
   prompt, using the same `agibot.tools.g2_gr00t_shadow_adapter` helpers the
   validated production runner uses.
3. Calls the GR00T model server.
4. Decodes the H16 action chunk to `XYZ, quaternion XYZW, gripper` targets and
   prints them with sanity checks.

What this does NOT do
---------------------
It never opens the action bridge, never tunnels port 9200, and never sends any
target to the robot.  The observation bridge it talks to is physically
incapable of actuation (`control_api_exposed: false`, `action: []`,
`motor_commands_sent: 0`), and this script asserts those flags before running.

This is the zero-risk way to exercise the full perception -> model -> action
chain against live robot data.

Usage (on 10.20.15.170, tunnel 19100 -> robot:9100, model server on 5564)
  cd ~/ct/Isaac-GR00T
  .venv/bin/python3.12 agibot/scripts/shadow_infer_grasp.py --repeat 3
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agibot.scripts.run_g2_groot_live_model_h2 import PROMPT, TRAINING_REFERENCE  # noqa: E402
from agibot.tools.g2_gr00t_shadow_adapter import (  # noqa: E402
    build_policy_observation,
    decode_action_chunk,
)
from agibot.tools.g2_groot_live_observation_client import (  # noqa: E402
    G2LiveObservationClient,
)
from gr00t.policy.server_client import PolicyClient  # noqa: E402

# Cell limits, for reporting only -- this script commands nothing.
WORKSPACE_MIN = np.asarray([0.448, -0.333, 0.987])
WORKSPACE_MAX = np.asarray([0.754, -0.197, 1.217])
GRIPPER_OPEN = -0.785
GRIPPER_CLOSED = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--observation-host", default="127.0.0.1")
    parser.add_argument("--observation-port", type=int, default=19100)
    parser.add_argument("--model-host", default="127.0.0.1")
    parser.add_argument("--model-port", type=int, default=5564)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--model-timeout-ms", type=int, default=30000)
    parser.add_argument("--repeat", type=int, default=1, help="inference calls")
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def assert_read_only(info: dict) -> None:
    problems = []
    if info.get("control_api_exposed") is not False:
        problems.append(f"control_api_exposed={info.get('control_api_exposed')}")
    if info.get("live_execution_enabled") is not False:
        problems.append(f"live_execution_enabled={info.get('live_execution_enabled')}")
    if int(info.get("motor_commands_sent", -1)) != 0:
        problems.append(f"motor_commands_sent={info.get('motor_commands_sent')}")
    if info.get("action") not in ([], None):
        problems.append(f"action={info.get('action')}")
    if problems:
        raise SystemExit("REFUSED: bridge is not read-only -> " + ", ".join(problems))


def summarise_chunk(targets: np.ndarray, live_xyz: np.ndarray) -> dict[str, Any]:
    xyz = targets[:, :3]
    grip = targets[:, 7]
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if len(xyz) > 1 else np.zeros(0)
    inside = np.all((xyz >= WORKSPACE_MIN) & (xyz <= WORKSPACE_MAX), axis=1)
    closing = np.where(grip > GRIPPER_OPEN + 0.05)[0]
    return {
        "horizon": int(targets.shape[0]),
        "all_finite": bool(np.isfinite(targets).all()),
        "first_xyz": xyz[0].round(5).tolist(),
        "last_xyz": xyz[-1].round(5).tolist(),
        "net_displacement_mm": round(float(np.linalg.norm(xyz[-1] - xyz[0])) * 1000, 2),
        "drift_from_live_mm": round(float(np.linalg.norm(xyz[0] - live_xyz)) * 1000, 2),
        "max_step_mm": round(float(step.max()) * 1000, 2) if step.size else 0.0,
        "z_min": round(float(xyz[:, 2].min()), 5),
        "z_max": round(float(xyz[:, 2].max()), 5),
        "net_lift_mm": round(float(xyz[:, 2].max() - xyz[0, 2]) * 1000, 2),
        "inside_workspace_all": bool(inside.all()),
        "inside_workspace_count": int(inside.sum()),
        "gripper_first": round(float(grip[0]), 4),
        "gripper_last": round(float(grip[-1]), 4),
        "gripper_min": round(float(grip.min()), 4),
        "gripper_max": round(float(grip.max()), 4),
        "first_closing_step": int(closing[0]) if closing.size else None,
    }


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "schema": "g2_grasp_shadow_inference_v1",
        "started_at": datetime.now().astimezone().isoformat(),
        "mode": "SHADOW_NO_ACTUATION",
        "prompt": PROMPT,
        "observation": f"{args.observation_host}:{args.observation_port}",
        "model": f"{args.model_host}:{args.model_port}",
        "calls": [],
    }

    print("=" * 78)
    print("SHADOW INFERENCE -- decodes actions, sends NOTHING to the robot")
    print("=" * 78)
    print(f"observation bridge : {args.observation_host}:{args.observation_port}")
    print(f"model server       : {args.model_host}:{args.model_port}")
    print(f"prompt             : {PROMPT[:72]}...")
    print()

    client = PolicyClient(
        host=args.model_host, port=args.model_port, timeout_ms=args.model_timeout_ms
    )
    try:
        if not client.ping():
            raise SystemExit("model server did not answer ping")
        print("model server ping OK")

        with G2LiveObservationClient(
            args.observation_host, args.observation_port, args.timeout_s,
            compute_payload_hashes=False,
        ) as bridge:
            info = bridge.get_info()
            assert_read_only(info)
            print("observation bridge read-only assertions OK"
                  f" (control_api_exposed={info.get('control_api_exposed')},"
                  f" motor_commands_sent={info.get('motor_commands_sent')},"
                  f" action={info.get('action')})")
            print()

            for call in range(1, args.repeat + 1):
                snapshot = bridge.get_snapshot()
                meta = snapshot.metadata
                pose = np.asarray(meta["right_eef_xyz_quaternion_xyzw"], dtype=np.float64)
                gripper = float(meta["right_gripper"]["training_position"])

                observation = build_policy_observation(
                    head_color_rgb=snapshot.head_color_rgb,
                    hand_right_rgb=snapshot.hand_right_rgb,
                    right_pose_xyz_quaternion_xyzw=pose,
                    right_gripper_training=gripper,
                    prompt=PROMPT,
                )

                started = time.monotonic()
                action, _info = client.get_action(observation)
                latency = time.monotonic() - started

                targets = decode_action_chunk(action)
                summary = summarise_chunk(targets, pose[:3])
                summary["latency_s"] = round(latency, 4)
                summary["live_xyz"] = pose[:3].round(5).tolist()
                summary["live_gripper"] = round(gripper, 4)
                summary["live_dist_to_training_start_mm"] = round(
                    float(np.linalg.norm(pose[:3] - TRAINING_REFERENCE[:3])) * 1000, 2
                )
                report["calls"].append(summary)

                print(f"--- call {call}/{args.repeat} ---")
                print(f"  live xyz            {summary['live_xyz']}"
                      f"  gripper {summary['live_gripper']:+.4f}"
                      f"  (dist to training start {summary['live_dist_to_training_start_mm']} mm)")
                print(f"  latency             {summary['latency_s']:.3f} s")
                print(f"  chunk horizon       {summary['horizon']}"
                      f"   all_finite={summary['all_finite']}")
                print(f"  action xyz first    {summary['first_xyz']}")
                print(f"  action xyz last     {summary['last_xyz']}")
                print(f"  drift from live     {summary['drift_from_live_mm']} mm"
                      "   (first target vs current pose)")
                print(f"  net displacement    {summary['net_displacement_mm']} mm"
                      f"   max step {summary['max_step_mm']} mm")
                print(f"  z range             {summary['z_min']} .. {summary['z_max']}"
                      f"   net lift {summary['net_lift_mm']} mm")
                print(f"  inside cell box     {summary['inside_workspace_all']}"
                      f"  ({summary['inside_workspace_count']}/{summary['horizon']} waypoints)")
                print(f"  gripper first/last  {summary['gripper_first']:+.4f}"
                      f" -> {summary['gripper_last']:+.4f}"
                      f"   min {summary['gripper_min']:+.4f} max {summary['gripper_max']:+.4f}")
                print(f"  first closing step  {summary['first_closing_step']}"
                      "   (None = never closes in this chunk)")
                print("  full chunk (idx, x, y, z, gripper):")
                for index, row in enumerate(targets):
                    print(f"    {index:2d}  {row[0]:+.5f} {row[1]:+.5f} {row[2]:+.5f}"
                          f"   grip {row[7]:+.4f}")
                print()
                if call < args.repeat:
                    time.sleep(args.interval_s)

        print("=" * 78)
        print("SHADOW COMPLETE -- no target was sent to the robot")
        print("=" * 78)
        report["result"] = "PASS_SHADOW"
        return 0

    except KeyboardInterrupt:
        print("\nstopped by operator")
        report["result"] = "OPERATOR_INTERRUPT"
        return 1
    except Exception as error:  # noqa: BLE001
        print(f"\nSHADOW FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        report["result"] = f"ERROR_{type(error).__name__}"
        report["error"] = str(error)
        return 1
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
        report["finished_at"] = datetime.now().astimezone().isoformat()
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False))
            print(f"report -> {args.report}")


if __name__ == "__main__":
    raise SystemExit(main())
