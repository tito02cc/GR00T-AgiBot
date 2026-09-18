#!/usr/bin/env python3
"""Print task-matched commands only. Never connect, start services or move a robot."""

import argparse
import json
from pathlib import Path
import shlex


HERE = Path(__file__).resolve().parent
TASKS = ("xichong_right_place_r0002", "zhewan_right_place_r0003")


def build_commands(task, repo, report, robot_root):
    config = json.loads((HERE / task / "profile.json").read_text())
    python = str(repo / ".venv/bin/python")
    runtime = HERE / "workstation_runtime"
    server = [python, "-u", str(repo / "gr00t/eval/run_gr00t_server.py"),
              "--model-path", str(repo / config["model"]), "--embodiment-tag", "NEW_EMBODIMENT",
              "--device", "cuda:0", "--host", "127.0.0.1", "--port", str(config["model_port"])]
    if config["modality_config"]:
        server.extend(["--modality-config-path", str(runtime / config["modality_config"])])
    live = ["env", f"PYTHONPATH={runtime}:{repo}", python, "-u",
            str(runtime / "agibot/scripts/run_g2_groot_full_place_inference.py"),
            "--execute", "--confirm", "EXECUTE_G2_GROOT_FULL_RIGHT_ARM_PLACE_INFERENCE",
            "--model-port", str(config["model_port"]),
            "--observation-port", str(config["observation_port"]),
            "--action-port", str(config["action_port"]),
            "--prompt", config["prompt"], "--initial-pose", *map(str, config["initial_pose"]),
            "--initial-gripper", str(config["initial_gripper"]),
            "--minimum-retraction-m", str(config["minimum_retraction_m"]),
            "--max-cycles", "0", "--report", str(report)]
    for key in ("optimize_transport", "native_chunk_submission",
                "require_native_collision_latch", "freeze_compensation_after_calibration"):
        if config[key]:
            live.append("--" + key.replace("_", "-"))
    if config["require_native_collision_latch"]:
        live.extend(["--required-control-mode", str(config["required_control_mode"])])
    bridge = str(robot_root / task / "robot_bridge.sh")
    return {
        "model_server_on_workstation": server,
        "observation_on_robot_if_not_running": ["bash", bridge, "observation"],
        "action_standby_on_robot": ["bash", bridge, "standby"],
        "action_control_on_robot_alternative_to_standby": ["bash", bridge, "control"],
        "forward_on_workstation_if_not_running": ["ssh", "-N", "-o", "ExitOnForwardFailure=yes",
            "-L", "19100:127.0.0.1:9100", "-L", "19200:127.0.0.1:9200", "agi@10.20.15.194"],
        "live_after_onsite_confirmation": live,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--robot-root", type=Path,
                        default=Path("/home/agi/vla_ct/releases/g2_194_20260914"))
    args = parser.parse_args()
    report = args.report.resolve()
    if report.exists():
        parser.error("report already exists; choose a new run path")
    print("# PRINT ONLY. Separate terminals; standby and control are alternatives.")
    print("# Same ports: never run both tasks together. Confirm actual model and scene before live.")
    for label, command in build_commands(args.task, args.repo.resolve(), report, args.robot_root).items():
        print(f"\n# {label}\n{shlex.join(command)}")


if __name__ == "__main__":
    main()
