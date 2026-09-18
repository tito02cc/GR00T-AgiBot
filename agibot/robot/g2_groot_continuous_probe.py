#!/usr/bin/env python3
"""Measure combined-API feedback control while holding the initial jaw command.

Default is read-only. An executed probe calibrates at the measured pose, moves
up 2 mm, returns along that same short path, then exits. It never opens/closes
the jaw, changes robot settings, or sends a recovery command after failure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import agibot_gdk
from g2_groot_continuous_controller import ContinuousRightArmController
from g2_groot_continuous_sender import NativeTrajectorySender
from g2_groot_trajectory_tracking_probe import quaternion_angle
import numpy as np


CONFIRMATION = "TEST_G2_GROOT_CONTINUOUS_HOLD"


def read_sample(robot, tf):
    stamp = time.monotonic_ns()
    frame = tf.get_tf_from_base_link("arm_r_end_link")
    pose = [frame.translation.x, frame.translation.y, frame.translation.z,
            frame.rotation.x, frame.rotation.y, frame.rotation.z, frame.rotation.w]
    source_ns = int(tf.get_latest_timestamp("arm_r_end_link"))
    motion = robot.get_motion_control_status()
    whole = robot.get_whole_body_status()
    end = robot.get_end_state()["right_end_state"]
    motor = end["end_states"][0]
    joints = robot.get_joint_states()
    right_joints = [item for item in joints["states"]
                    if item["name"].startswith(tuple(f"idx{i}_" for i in range(61, 68)))]
    return {
        "read_monotonic_ns": stamp,
        "read_wall_ns": time.time_ns(),
        "pose": [float(value) for value in pose],
        "tf_timestamp_ns": source_ns,
        "joint_timestamp_ns": int(joints["timestamp"]),
        "right_joints": [{key: item[key] for key in ("name", "position", "error_code")}
                         for item in right_joints],
        "mode": int(motion.mode),
        "motion_error": int(motion.error_code),
        "motion_error_msg": str(motion.error_msg),
        "whole": whole,
        "jaw": float(motor["position"]),
        "jaw_error": int(motor["err_code"]),
        "jaw_effort": float(motor["effort"]),
        "jaw_enabled": bool(motor["enable"]),
        "tool_names": list(end["names"]),
    }


def validate_sample(sample, anchor, initial_gripper, *, calibration=False):
    pose = np.asarray(sample["pose"], dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise RuntimeError("invalid EEF feedback")
    if abs(float(np.linalg.norm(pose[3:])) - 1.0) > 0.001:
        raise RuntimeError("invalid EEF quaternion")
    if sample["mode"] != 1 or sample["motion_error"] != 0:
        raise RuntimeError(f"motion mode/error: {sample['mode']}/{sample['motion_error']}: "
                           f"{sample['motion_error_msg']}")
    if sample["tool_names"] != ["idx71_gripper_r_inner_joint1"]:
        raise RuntimeError("unexpected right tool joint")
    if sample["whole"].get("right_end_model") != "omnipicker":
        raise RuntimeError("unexpected right tool model")
    if not sample["jaw_enabled"] or sample["jaw_error"]:
        raise RuntimeError("right jaw disabled or faulted")
    if not math.isfinite(sample["jaw"]) or abs(sample["jaw"] - initial_gripper) > 0.015:
        raise RuntimeError("jaw moved away from the explicitly retained initial command")
    for key in ("right_arm_error", "left_arm_error", "right_end_error", "left_end_error"):
        if int(sample["whole"].get(key, -1)) != 0:
            raise RuntimeError(f"robot fault: {key}")
    for key in ("right_arm_estop", "left_arm_estop"):
        if sample["whole"].get(key):
            raise RuntimeError(f"robot emergency stop: {key}")
    if len(sample["right_joints"]) != 7 or any(
        item["error_code"] for item in sample["right_joints"]
    ):
        raise RuntimeError("right joint feedback incomplete or faulted")
    for key in ("tf_timestamp_ns", "joint_timestamp_ns"):
        age = (sample["read_wall_ns"] - sample[key]) / 1e9
        if sample[key] <= 0 or not -0.1 <= age <= 0.5:
            raise RuntimeError(f"invalid feedback source time: {key}, age={age}")
    if anchor is not None:
        delta = pose[:3] - np.asarray(anchor["pose"][:3])
        # Startup compensation needs time to settle. This is a probe-specific
        # allowance, not an official GDK fault threshold or a convergence claim.
        downward_limit = 0.003 if calibration else 0.0015
        if delta[2] < -downward_limit or np.linalg.norm(delta) > 0.004:
            raise RuntimeError(f"hold/roundtrip envelope exceeded: delta_m={delta.tolist()}")
        if np.linalg.norm(delta[:2]) > 0.002:
            raise RuntimeError("unexpected XY displacement above 2 mm")
        if quaternion_angle(pose, np.asarray(anchor["pose"])) > 0.02:
            raise RuntimeError("unexpected orientation change above 0.02 rad")


class ProbeController(ContinuousRightArmController):
    def __init__(self, *args, report, sample, **kwargs):
        super().__init__(*args, **kwargs)
        self.report = report
        self.sample = sample
        self.phase = "calibration"

    def tick(self, desired):
        row = {"phase": self.phase, "desired_pose": np.asarray(desired).tolist()}
        self.report["ticks"].append(row)
        row["before"] = self.sample()
        try:
            result = super().tick(desired)
            row["controller"] = self.last_tick
            row["sender"] = self.command_sender.snapshot()
            row["after"] = self.sample()
            row["status"] = "COMPLETED"
            return result
        except Exception as error:
            row["status"] = "FAILED"
            row["error"] = str(error)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--initial-gripper-command", type=float, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.execute and args.confirm != CONFIRMATION:
        parser.error(f"--execute requires --confirm {CONFIRMATION}")
    if args.confirm and not args.execute:
        parser.error("--confirm is only valid with --execute")
    report = {"execute": args.execute, "ticks": [], "fixed_gripper_command":
              args.initial_gripper_command, "automatic_recovery": False,
              "calibration_downward_allowance_m": 0.003,
              "movement_downward_allowance_m": 0.0015}
    initialized = False
    sender = None
    with args.report.open("x", encoding="utf-8") as stream:
        try:
            if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
                raise RuntimeError("gdk_init failed")
            initialized = True
            robot, tf = agibot_gdk.Robot(), agibot_gdk.TF()
            time.sleep(2.0)
            initial = read_sample(robot, tf)
            report["initial"] = initial
            validate_sample(initial, None, args.initial_gripper_command)
            print(json.dumps({"event": "initial", "execute": args.execute,
                              "pose": initial["pose"], "jaw": initial["jaw"]}), flush=True)
            if args.execute:
                for key in ("left_arm_control", "right_arm_control"):
                    if initial["whole"].get(key):
                        raise RuntimeError(f"another arm owner is active: {key}")

                def sample():
                    value = read_sample(robot, tf)
                    report["latest_sample"] = value
                    validate_sample(value, initial, args.initial_gripper_command,
                                    calibration=controller.phase == "calibration")
                    return value

                sender = NativeTrajectorySender(args.initial_gripper_command, before_publish=sample)
                sender.initialize_references(robot, tf)
                anchor = np.asarray(initial["pose"])
                controller = ProbeController(
                    robot, tf, sender=sender, report=report, sample=sample,
                    workspace_min=anchor[:3] - 0.004,
                    workspace_max=anchor[:3] + 0.004,
                )
                report["calibration"] = controller.calibrate(2.0)
                calibrated = sample()
                if np.linalg.norm(np.asarray(calibrated["pose"][:3]) - anchor[:3]) > 0.0005:
                    raise RuntimeError("calibration error above 0.5 mm; no upward target sent")
                print(json.dumps({"event": "calibrated", **report["calibration"]}), flush=True)
                target = anchor.copy()
                target[2] += 0.002
                controller.phase = "up"
                controller.move_to(target, 0.5)
                controller.phase = "top_hold"
                controller.hold_ticks(25)
                top = sample()
                if top["pose"][2] - anchor[2] < 0.001:
                    raise RuntimeError("no measured upward response of at least 1 mm")
                controller.phase = "return"
                controller.move_to(anchor, 0.5)
                controller.phase = "final_hold"
                controller.hold_ticks(25)
                final = sample()
                error = float(np.linalg.norm(np.asarray(final["pose"][:3]) - anchor[:3]))
                if error > 0.0005:
                    raise RuntimeError(f"return error {error} above 0.5 mm")
                report["result"] = {"status": "PASS", "return_error_m": error,
                                    "scope": "fixed-jaw hold and 2 mm arm roundtrip only"}
            else:
                report["result"] = {"status": "READ_ONLY", "commands_sent": 0}
        except Exception as error:
            report["result"] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
        finally:
            if sender is not None:
                report["sender"] = sender.snapshot()
            if initialized:
                try:
                    agibot_gdk.gdk_release()
                except Exception as error:
                    report["release_error"] = str(error)
                    report["result"]["status"] = "FAIL"
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"event": "result", **report["result"]}), flush=True)
    return 0 if report["result"]["status"] in ("PASS", "READ_ONLY") else 2


if __name__ == "__main__":
    raise SystemExit(main())
