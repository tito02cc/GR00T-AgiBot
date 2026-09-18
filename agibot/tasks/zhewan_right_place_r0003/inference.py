#!/usr/bin/env python3
"""Task-specific official server/offline entry; live commands are printed, never executed."""

import argparse
import json
import os
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[3]
PROFILE = Path(__file__).with_name("inference.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("server", "offline", "print-live-command"))
    parser.add_argument("--port", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config = json.loads(PROFILE.read_text())
    model = ROOT / config["model"]
    dataset = ROOT / config["dataset_root"]
    actual_prompt = json.loads((dataset / "train/meta/tasks.jsonl").read_text())["task"]
    if config["prompt"] != actual_prompt:
        raise ValueError("Profile prompt differs from training prompt")
    processor = json.loads((model / "processor_config.json").read_text())
    if processor["processor_kwargs"]["use_percentiles"] != config["use_percentiles"]:
        raise ValueError("Profile normalization differs from saved processor")
    port = args.port or config["model_port"]
    python = str(ROOT / ".venv/bin/python")
    if args.mode == "server":
        command = [
            python,
            "-u",
            str(ROOT / "gr00t/eval/run_gr00t_server.py"),
            "--model-path",
            str(model),
            "--embodiment-tag",
            config["embodiment_tag"],
            "--device",
            "cuda:0",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
    elif args.mode == "offline":
        if args.report is None:
            parser.error("offline requires --report OUTPUT_DIRECTORY; start the server first")
        command = [
            python,
            "-u",
            str(ROOT / "agibot/scripts/evaluate_g2_place_offline.py"),
            "--dataset-root",
            str(dataset),
            "--port",
            str(port),
            "--output-dir",
            str(args.report.resolve()),
        ]
    else:
        guard_path = None
        if config.get("contact_guard_config"):
            sys.path.insert(0, str(ROOT))
            from agibot.robot.g2_groot_contact_guard import ContactGuard

            guard_path = ROOT / config["contact_guard_config"]
            guard = ContactGuard.from_file(guard_path)
            guard.ensure_ready()
            if guard.snapshot()["config"]["robot_ip"] != config["robot_ip"]:
                raise ValueError("Contact guard calibration belongs to a different robot")
        reference = json.loads((ROOT / config["initial_pose_reference"]).read_text())
        if args.report is None:
            parser.error("print-live-command requires --report OUTPUT_JSON")
        command = [
            python,
            "-u",
            str(ROOT / "agibot/scripts/run_g2_groot_full_place_inference.py"),
            "--execute",
            "--confirm",
            "EXECUTE_G2_GROOT_FULL_RIGHT_ARM_PLACE_INFERENCE",
            "--model-port",
            str(port),
            "--observation-port",
            str(config["observation_port"]),
            "--action-port",
            str(config["action_port"]),
            "--prompt",
            config["prompt"],
            "--initial-pose",
            *map(str, reference["xyz_quaternion_xyzw"]),
            "--initial-gripper",
            str(reference["right_gripper_native_radians"]),
            "--minimum-retraction-m",
            str(config["minimum_retraction_m"]),
            "--max-cycles",
            "0",
            "--report",
            str(args.report.resolve()),
        ]
        if config["optimize_transport"]:
            command.append("--optimize-transport")
        if config.get("native_chunk_submission", False):
            command.append("--native-chunk-submission")
        if config.get("require_native_collision_latch", False):
            command.extend(["--require-native-collision-latch", "--required-control-mode", str(config["required_control_mode"])])
        if config.get("freeze_compensation_after_calibration", False):
            command.append("--freeze-compensation-after-calibration")
        if guard_path is not None:
            command.extend(["--contact-guard-config", str(guard_path)])
        print(f"# Robot bridge directory: {config['robot_bridge_directory']}")
        print("# PRINT ONLY: confirm live scene, initial image/pose and new-task workspace first.")
        print(shlex.join(command))
        return
    print(shlex.join(command), flush=True)
    os.chdir(ROOT)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_ALBUMENTATIONS_UPDATE="1")
    os.execv(python, command)


if __name__ == "__main__":
    main()
