#!/usr/bin/env python3
"""Supported team entry point for the Agibot G2 GR00T pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGIBOT_ROOT = REPO_ROOT / "agibot"
CASE_TASK = AGIBOT_ROOT / "configs/task_right_grasp.toml"
EXAMPLE_ARTIFACTS_LOCK = (
    AGIBOT_ROOT / "examples/task_profile_example/artifacts.lock.json"
)


def repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def nested(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted)
        value = value[key]
    return value


def validate_task(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        config = load_toml(path)
    except Exception as exc:  # noqa: BLE001 - aggregate configuration failures
        return [f"cannot load task config {path}: {exc}"]
    if config.get("schema_version") != 1:
        errors.append("task schema_version must be 1")
    for key in ("task.id", "task.prompt"):
        try:
            value = str(nested(config, key)).strip()
            if not value or "CHANGE_ME" in value:
                errors.append(f"task config must replace {key}")
        except KeyError:
            errors.append(f"task config missing {key}")
    arm = str(config.get("task", {}).get("arm", "")).lower()
    if arm not in {"left", "right"}:
        errors.append("task.arm must be 'left' or 'right'")
        arm = "right"
    expected = {
        "task.embodiment": "NEW_EMBODIMENT",
        "task.frequency_hz": 10.0,
        "task.action_horizon": 16,
        "images.keys": ["head_color", f"hand_{arm}"],
        "images.width": 640,
        "images.height": 480,
        f"state.{arm}_eef.dimension": 9,
        f"state.{arm}_eef.representation": "XYZ_ROT6D",
        f"state.{arm}_eef.rotation_6d_order": "r00_r01_r02_r10_r11_r12",
        f"state.{arm}_eef.pose_frame_label": "base_link_tf",
        f"state.{arm}_gripper.open": -0.785,
        f"state.{arm}_gripper.closed": 0.0,
        f"action.{arm}_eef.type": "RELATIVE",
        f"action.{arm}_gripper.type": "ABSOLUTE",
    }
    for key, expected_value in expected.items():
        try:
            actual = nested(config, key)
        except KeyError:
            errors.append(f"task config missing {key}")
            continue
        if actual != expected_value:
            errors.append(f"task config {key}={actual!r}, expected {expected_value!r}")
    try:
        indices = nested(config, f"action.{arm}_eef.delta_indices")
        if indices != list(range(16)):
            errors.append(f"action.{arm}_eef.delta_indices must be exactly 0..15")
    except KeyError:
        errors.append(f"task config missing action.{arm}_eef.delta_indices")
    if config.get("pipeline", {}).get("arm") != arm:
        errors.append(f"pipeline.arm must match task.arm ({arm})")
    for key in (
        "pipeline.modality_config",
        "pipeline.converter",
        "pipeline.selector",
        "pipeline.training_validator",
        "pipeline.shadow_validator",
        "pipeline.training_preflight",
        "pipeline.training_launcher",
        "pipeline.inference_runner",
        "pipeline.recovery_runner",
    ):
        try:
            value = str(nested(config, key))
            if "CHANGE_ME" in value:
                errors.append(f"task config must replace {key}")
            elif not repository_path(value).is_file():
                errors.append(f"task pipeline file does not exist: {value}")
        except KeyError:
            errors.append(f"task config missing {key}")
    return errors


def validate_robot(path: Path, allow_unconfirmed: bool = False) -> list[str]:
    errors: list[str] = []
    try:
        config = load_toml(path)
    except Exception as exc:  # noqa: BLE001 - aggregate configuration failures
        return [f"cannot load robot config {path}: {exc}"]
    arm = str(config.get("identity", {}).get("arm", "")).lower()
    if arm not in {"left", "right"}:
        errors.append("robot identity.arm must be 'left' or 'right'")
        arm = "right"
    arm_letter = "l" if arm == "left" else "r"
    required = {
        "frames.source_pose_frame": "base_link",
        "frames.training_pose_frame_label": "base_link_tf",
        f"frames.{arm}_eef_frame": f"arm_{arm_letter}_end_link",
        "frames.quaternion_order": "xyzw",
        "frames.rot6d_order": "r00_r01_r02_r10_r11_r12",
        "timing.policy_hz": 10.0,
        "timing.controller_hz": 50.0,
        "timing.controller_ticks_per_action": 5,
        "cameras.head_key": "head_color",
        f"cameras.{arm}_wrist_key": f"hand_{arm}",
        "cameras.width": 640,
        "cameras.height": 480,
        "gripper.end_model": "omnipicker",
        "gripper.training_open": -0.785,
        "gripper.training_closed": 0.0,
        "gripper.feedback_direction": "increases_when_closing",
        "gdk.single_persistent_owner_required": True,
        "gdk.motion_plan_for_policy_waypoints": False,
    }
    for key, expected in required.items():
        try:
            actual = nested(config, key)
        except KeyError:
            errors.append(f"robot config missing {key}")
            continue
        if actual != expected:
            errors.append(f"robot config {key}={actual!r}, expected {expected!r}")
    placeholders = (
        "identity.robot_id",
        "identity.gdk_version",
        "identity.calibrated_at",
        "network.robot_host",
    )
    for key in placeholders:
        try:
            if "CHANGE_ME" in str(nested(config, key)):
                errors.append(f"robot config still contains placeholder {key}")
        except KeyError:
            errors.append(f"robot config missing {key}")
    if not allow_unconfirmed and not bool(config.get("identity", {}).get("calibration_confirmed")):
        errors.append("robot calibration_confirmed must be true")
    try:
        pose = nested(config, "initial_pose.xyz_quaternion_xyzw")
        if len(pose) != 7 or sum(float(v) ** 2 for v in pose[3:]) < 0.99:
            errors.append("initial pose must be XYZ plus a normalized XYZW quaternion")
    except (KeyError, TypeError, ValueError):
        errors.append("invalid initial_pose.xyz_quaternion_xyzw")
    try:
        bounds = (
            (
                nested(config, "workspace.hardware_minimum_xyz_m"),
                nested(config, "workspace.hardware_maximum_xyz_m"),
                "hardware",
            ),
            (
                nested(config, "workspace.policy_minimum_xyz_m"),
                nested(config, "workspace.policy_maximum_xyz_m"),
                "policy",
            ),
        )
        for minimum, maximum, label in bounds:
            if (
                len(minimum) != 3
                or len(maximum) != 3
                or any(a >= b for a, b in zip(minimum, maximum))
            ):
                errors.append(
                    f"{label} workspace minimum must be below maximum on every axis"
                )
    except (KeyError, TypeError):
        errors.append("invalid workspace bounds")
    try:
        joint_name = str(nested(config, "gripper.joint_name")).strip()
        feedback_encoding = str(nested(config, "gripper.feedback_encoding"))
        raw_open = float(nested(config, "gripper.raw_open_nominal"))
        raw_closed = float(nested(config, "gripper.raw_closed_nominal"))
        statuses = nested(config, "gripper.accepted_settled_statuses")
        if not joint_name or "CHANGE_ME" in joint_name:
            errors.append("gripper joint_name has not been calibrated")
        if feedback_encoding == "raw_0_120":
            endpoints_valid = raw_open >= 0 and raw_closed > raw_open
        elif feedback_encoding == "native_radians":
            endpoints_valid = (
                abs(raw_open - (-0.785)) <= 0.02
                and abs(raw_closed) <= 0.02
            )
        else:
            endpoints_valid = False
            errors.append("gripper feedback_encoding is unsupported")
        if not endpoints_valid or not statuses:
            errors.append("gripper raw endpoints/statuses have not been calibrated")
    except (KeyError, TypeError, ValueError):
        errors.append("invalid gripper calibration")
    return errors


def validate_artifacts_lock(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"cannot load {path}: {exc}"]
    errors = []
    if payload.get("schema") != "agibot_groot_artifacts_v1":
        errors.append("unexpected artifacts.lock.json schema")
    if "CHANGE_ME" in json.dumps(payload):
        errors.append("artifacts lock still contains CHANGE_ME placeholders")
    dataset = payload.get("training_dataset", {})
    checkpoint = payload.get("finetuned_checkpoint", {})
    for key in ("episodes", "frames", "parquet_files", "video_files"):
        if not isinstance(dataset.get(key), int) or dataset[key] <= 0:
            errors.append(f"training_dataset.{key} must be a positive integer")
    if not dataset.get("sha256_manifest"):
        errors.append("training_dataset.sha256_manifest is required")
    if not isinstance(checkpoint.get("training_steps"), int) or checkpoint["training_steps"] <= 0:
        errors.append("finetuned_checkpoint.training_steps must be a positive integer")
    if not isinstance(checkpoint.get("safetensor_shards"), int) or checkpoint[
        "safetensor_shards"
    ] <= 0:
        errors.append("finetuned_checkpoint.safetensor_shards must be a positive integer")
    if not checkpoint.get("sha256_manifest"):
        errors.append("finetuned_checkpoint.sha256_manifest is required")
    return errors


def sha256_error(manifest: Path, directory: Path) -> str | None:
    result = subprocess.run(
        ["sha256sum", "-c", "--quiet", str(manifest.resolve())],
        cwd=directory,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode == 0:
        return None
    detail = result.stdout.strip() or f"sha256sum exited {result.returncode}"
    return f"SHA256 verification failed for {directory}: {detail}"


def doctor(args: argparse.Namespace) -> int:
    errors = []
    task_config: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    if sys.version_info[:2] != (3, 12):
        errors.append(f"Python 3.12 required, running {sys.version.split()[0]}")
    if args.task_config:
        task_path = args.task_config.resolve()
        errors.extend(validate_task(task_path))
        try:
            task_config = load_toml(task_path)
        except Exception:  # already reported by validate_task
            pass
    if args.robot_config:
        errors.extend(validate_robot(args.robot_config.resolve(), args.allow_unconfirmed))
    if args.artifacts_lock:
        lock_path = args.artifacts_lock.resolve()
        errors.extend(validate_artifacts_lock(lock_path))
        try:
            artifacts = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:  # already reported by validate_artifacts_lock
            pass
    required = [
        REPO_ROOT / "gr00t/eval/run_gr00t_server.py",
        AGIBOT_ROOT / "templates/task_profile/task.toml",
        AGIBOT_ROOT / "configs/g2_right_arm.template.toml",
        AGIBOT_ROOT / "scripts/run_g2_groot_full_protected_inference.py",
        AGIBOT_ROOT / "robot/g2_groot_right_observation_bridge.py",
        AGIBOT_ROOT / "robot/g2_groot_persistent_h1_action_bridge.py",
    ]
    for path in required:
        if not path.is_file():
            errors.append(f"missing required repository file: {path.relative_to(REPO_ROOT)}")
    if args.require_data:
        if not task_config:
            errors.append("--require-data requires --task-config")
        if not args.dataset:
            errors.append("--require-data requires --dataset")
            dataset = Path("/__missing_dataset__")
        else:
            dataset = args.dataset.resolve()
        parquet = list(dataset.glob("data/chunk-*/*.parquet"))
        videos = list(dataset.glob("videos/chunk-*/*/*.mp4"))
        dataset_lock = artifacts.get("training_dataset", {})
        expected_parquet = int(
            dataset_lock.get("parquet_files")
            or task_config.get("dataset", {}).get("target_episode_count", 0)
        )
        expected_videos = int(
            dataset_lock.get("video_files")
            or expected_parquet * task_config.get("dataset", {}).get("videos_per_episode", 0)
        )
        if not parquet or not videos:
            errors.append(f"dataset is empty or incomplete at {dataset}")
        if expected_parquet and len(parquet) != expected_parquet:
            errors.append(
                f"dataset parquet mismatch at {dataset}: "
                f"actual={len(parquet)}, expected={expected_parquet}"
            )
        if expected_videos and len(videos) != expected_videos:
            errors.append(
                f"dataset video mismatch at {dataset}: "
                f"actual={len(videos)}, expected={expected_videos}"
            )
        embedded_manifest = dataset / "SHA256SUMS"
        released_value = dataset_lock.get("sha256_manifest", "")
        released_manifest = repository_path(released_value) if released_value else Path()
        manifest = embedded_manifest if embedded_manifest.is_file() else released_manifest
        if not manifest.is_file():
            errors.append(f"missing dataset SHA256 manifest: {dataset}")
        elif parquet and videos:
            hash_error = sha256_error(manifest, dataset)
            if hash_error:
                errors.append(hash_error)
    if args.require_model:
        if not artifacts:
            errors.append("--require-model requires --artifacts-lock")
        if not args.model:
            errors.append("--require-model requires --model")
            model = Path("/__missing_model__")
        else:
            model = args.model.resolve()
        shards = list(model.glob("model-*-of-*.safetensors"))
        checkpoint_lock = artifacts.get("finetuned_checkpoint", {})
        expected_shards = int(checkpoint_lock.get("safetensor_shards", 0))
        if not shards or (expected_shards and len(shards) != expected_shards):
            errors.append(
                f"incomplete checkpoint at {model}: "
                f"shards={len(shards)}, expected={expected_shards or 'unspecified'}"
            )
        else:
            manifest_value = checkpoint_lock.get("sha256_manifest", "")
            manifest = repository_path(manifest_value) if manifest_value else Path()
            if not manifest.is_file():
                errors.append(f"missing checkpoint SHA256 manifest: {manifest}")
            else:
                hash_error = sha256_error(manifest, model)
                if hash_error:
                    errors.append(hash_error)
    result = {
        "status": "PASS" if not errors else "FAIL",
        "python": sys.version.split()[0],
        "repository": str(REPO_ROOT),
        "task_config": str(args.task_config.resolve()) if args.task_config else None,
        "robot_config": str(args.robot_config.resolve()) if args.robot_config else None,
        "artifacts_lock": str(args.artifacts_lock.resolve()) if args.artifacts_lock else None,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def prepare(args: argparse.Namespace) -> int:
    task_path = args.task_config.resolve()
    errors = validate_task(task_path)
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 1
    task_config = load_toml(task_path)
    pipeline = task_config["pipeline"]
    episode_count = args.count or int(task_config["dataset"]["target_episode_count"])
    if episode_count <= 0:
        raise SystemExit("set --count or dataset.target_episode_count to a positive value")
    python = str(REPO_ROOT / ".venv/bin/python")
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    convert_command = [
        python,
        str(repository_path(pipeline["converter"])),
        "--source",
        str(args.raw_root.resolve()),
        "--output",
        str(args.full_dataset.resolve()),
        "--report",
        str(report_dir / "raw_validation.json"),
        "--workers",
        str(args.workers),
        "--quarantine-invalid",
    ]
    if args.overwrite:
        convert_command.append("--overwrite")
    run(convert_command)
    run(
        [
            python,
            str(repository_path(pipeline["selector"])),
            "--raw-root",
            str(args.raw_root.resolve()),
            "--source-dataset",
            str(args.full_dataset.resolve()),
            "--output-dataset",
            str(args.selected_dataset.resolve()),
            "--report",
            str(report_dir / "selection.json"),
            "--count",
            str(episode_count),
        ]
    )
    run(
        [
            python,
            "gr00t/data/stats.py",
            "--dataset-path",
            str(args.selected_dataset.resolve()),
            "--embodiment-tag",
            str(task_config["task"]["embodiment"]),
            "--modality-config-path",
            str(repository_path(pipeline["modality_config"])),
        ]
    )
    run(
        [
            python,
            str(repository_path(pipeline["training_validator"])),
            "--raw-root",
            str(args.raw_root.resolve()),
            "--dataset",
            str(args.selected_dataset.resolve()),
            "--report",
            str(report_dir / "training_hard_gate.json"),
            "--manifest",
            str(report_dir / "SHA256SUMS"),
            "--workers",
            str(args.workers),
        ]
    )
    shutil.copyfile(report_dir / "SHA256SUMS", args.selected_dataset.resolve() / "SHA256SUMS")
    run(
        [
            python,
            str(repository_path(pipeline["shadow_validator"])),
            "--raw-root",
            str(args.raw_root.resolve()),
            "--dataset",
            str(args.selected_dataset.resolve()),
            "--report",
            str(report_dir / "g2_shadow_gate.json"),
        ]
    )
    print(f"PREPARE PASS: {args.selected_dataset.resolve()}")
    return 0


def train(args: argparse.Namespace) -> int:
    task_path = args.task_config.resolve()
    errors = validate_task(task_path)
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 1
    task_config = load_toml(task_path)
    pipeline = task_config["pipeline"]
    training = task_config["training"]
    dataset = args.dataset.resolve()
    embedded_manifest = dataset / "SHA256SUMS"
    released_manifest = Path()
    if args.artifacts_lock:
        artifacts = json.loads(args.artifacts_lock.resolve().read_text(encoding="utf-8"))
        manifest_value = artifacts.get("training_dataset", {}).get("sha256_manifest", "")
        if manifest_value:
            released_manifest = repository_path(manifest_value)
    manifest = embedded_manifest if embedded_manifest.is_file() else released_manifest
    if not manifest.is_file():
        raise SystemExit(
            "dataset SHA256 manifest is missing; embed SHA256SUMS or pass --artifacts-lock"
        )
    environment = os.environ.copy()
    environment.update(
        {
            "CT_ROOT": str(args.ct_root.resolve()),
            "GROOT_REPO_ROOT": str(REPO_ROOT),
            "GROOT_DATASET": str(dataset),
            "GROOT_DATASET_MANIFEST": str(manifest),
            "GROOT_MODALITY_CONFIG": str(repository_path(pipeline["modality_config"])),
            "RUN_ID": str(training["run_id"]),
            "GROOT_MAX_STEPS": str(training["max_steps"]),
            "GROOT_SAVE_STEPS": str(training["save_steps"]),
            "GROOT_SAVE_TOTAL_LIMIT": str(training["save_total_limit"]),
            "GROOT_EXPECTED_EPISODES": str(
                task_config["dataset"]["target_episode_count"]
            ),
            "GROOT_EXPECTED_VIDEOS": str(
                task_config["dataset"]["target_episode_count"]
                * task_config["dataset"]["videos_per_episode"]
            ),
            "GROOT_EMBODIMENT_TAG": str(task_config["task"]["embodiment"]),
        }
    )
    if args.mode == "preflight":
        script = repository_path(pipeline["training_preflight"])
        command = ["bash", str(script)]
    else:
        script = repository_path(pipeline["training_launcher"])
        command = ["bash", str(script), args.mode]
    run(command, env=environment)
    return 0


def serve(args: argparse.Namespace) -> int:
    errors = validate_task(args.task_config.resolve())
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 1
    task_config = load_toml(args.task_config.resolve())
    run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            "-u",
            "gr00t/eval/run_gr00t_server.py",
            "--embodiment-tag",
            str(task_config["task"]["embodiment"]),
            "--modality-config-path",
            str(repository_path(task_config["pipeline"]["modality_config"])),
            "--model-path",
            str(args.model.resolve()),
            "--device",
            args.device,
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
    )
    return 0


def robot_preflight(args: argparse.Namespace) -> int:
    errors = validate_task(args.task_config.resolve()) + validate_robot(
        args.robot_config.resolve()
    )
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 1
    config = load_toml(args.robot_config)
    port = int(config["network"]["observation_local_port"])
    output = args.output_dir or (
        AGIBOT_ROOT
        / "local_reports"
        / f"robot_preflight_{datetime.now().astimezone():%Y%m%d_%H%M%S}"
    )
    run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            "agibot/scripts/capture_g2_groot_live_observation.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--training-dataset",
            str(args.dataset.resolve()),
            "--output-dir",
            str(output.resolve()),
            "--initial-translation-tolerance-m",
            str(config["initial_pose"]["read_only_translation_tolerance_m"]),
            "--initial-rotation-tolerance-deg",
            str(config["initial_pose"]["read_only_rotation_tolerance_deg"]),
            "--initial-gripper-tolerance",
            str(config["initial_pose"]["gripper_tolerance"]),
        ]
    )
    return 0


def infer(args: argparse.Namespace) -> int:
    errors = validate_task(args.task_config.resolve()) + validate_robot(
        args.robot_config.resolve()
    )
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 1
    if not args.execute:
        raise SystemExit("physical inference requires --execute")
    task_config = load_toml(args.task_config)
    config = load_toml(args.robot_config)
    task = task_config["task"]
    inference = task_config.get("inference", {})
    confirmation = str(
        inference.get(
            "confirmation",
            "EXECUTE_G2_GROOT_FULL_RIGHT_ARM_PROTECTED_INFERENCE",
        )
    )
    report = args.report or (
        AGIBOT_ROOT
        / "local_reports"
        / f"full_inference_{datetime.now().astimezone():%Y%m%d_%H%M%S}.json"
    )
    command = [
        str(REPO_ROOT / ".venv/bin/python"),
        "-u",
        str(repository_path(task_config["pipeline"]["inference_runner"])),
        "--execute",
        "--confirm",
        confirmation,
        "--observation-port",
        str(config["network"]["observation_local_port"]),
        "--action-port",
        str(config["network"]["action_local_port"]),
        "--model-port",
        str(config["network"]["model_local_port"]),
        "--prompt",
        str(task["prompt"]),
        "--initial-pose",
        *[str(value) for value in config["initial_pose"]["xyz_quaternion_xyzw"]],
        "--max-cycles",
        str(args.max_cycles),
        "--report",
        str(report.resolve()),
    ]
    completion = str(inference.get("completion", "grasp_lift"))
    if completion == "place_release_retract":
        command.extend(
            [
                "--initial-gripper",
                str(config["initial_pose"]["right_gripper_position"]),
                "--minimum-retraction-m",
                str(inference["minimum_retraction_m"]),
            ]
        )
    elif completion == "grasp_lift":
        minimum_progress = task.get(
            "minimum_task_progress_m", task.get("minimum_lift_m")
        )
        if minimum_progress is None:
            raise ValueError("grasp task must define minimum_task_progress_m")
        command.extend(["--minimum-task-progress-m", str(minimum_progress)])
    else:
        raise ValueError(f"unsupported inference completion mode: {completion}")
    run(command)
    return 0


def recover(args: argparse.Namespace) -> int:
    errors = validate_task(args.task_config.resolve()) + validate_robot(
        args.robot_config.resolve()
    )
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 1
    if not args.execute:
        raise SystemExit("physical recovery requires --execute")
    task_config = load_toml(args.task_config)
    config = load_toml(args.robot_config)
    report = args.report or (
        AGIBOT_ROOT
        / "local_reports"
        / f"recovery_{datetime.now().astimezone():%Y%m%d_%H%M%S}.json"
    )
    run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            "-u",
            str(repository_path(task_config["pipeline"]["recovery_runner"])),
            "--execute",
            "--expanded-session",
            "--confirm",
            "RECOVER_G2_GROOT_RIGHT_ARM_FROM_APPROACH_TO_TRAINING_START",
            "--port",
            str(config["network"]["action_local_port"]),
            "--target-pose",
            *[str(value) for value in config["initial_pose"]["xyz_quaternion_xyzw"]],
            "--report",
            str(report.resolve()),
        ]
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="validate repository/config/artifacts")
    doctor_parser.add_argument("--task-config", type=Path)
    doctor_parser.add_argument("--robot-config", type=Path)
    doctor_parser.add_argument("--artifacts-lock", type=Path)
    doctor_parser.add_argument("--allow-unconfirmed", action="store_true")
    doctor_parser.add_argument("--require-data", action="store_true")
    doctor_parser.add_argument("--require-model", action="store_true")
    doctor_parser.add_argument("--dataset", type=Path)
    doctor_parser.add_argument("--model", type=Path)
    doctor_parser.set_defaults(function=doctor)

    prepare_parser = subparsers.add_parser("prepare", help="convert, select, stats and gate")
    prepare_parser.add_argument("--task-config", type=Path, required=True)
    prepare_parser.add_argument("--raw-root", type=Path, required=True)
    prepare_parser.add_argument("--full-dataset", type=Path, required=True)
    prepare_parser.add_argument("--selected-dataset", type=Path, required=True)
    prepare_parser.add_argument("--report-dir", type=Path, required=True)
    prepare_parser.add_argument("--count", type=int)
    prepare_parser.add_argument("--workers", type=int, default=8)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(function=prepare)

    train_parser = subparsers.add_parser("train", help="single-A100 preflight/train launcher")
    train_parser.add_argument(
        "mode", choices=("preflight", "audit", "smoke", "baseline", "resume")
    )
    train_parser.add_argument("--ct-root", type=Path, required=True)
    train_parser.add_argument("--dataset", type=Path, required=True)
    train_parser.add_argument("--task-config", type=Path, required=True)
    train_parser.add_argument("--artifacts-lock", type=Path)
    train_parser.set_defaults(function=train)

    serve_parser = subparsers.add_parser("serve", help="start the GR00T inference service")
    serve_parser.add_argument("--task-config", type=Path, required=True)
    serve_parser.add_argument("--model", type=Path, required=True)
    serve_parser.add_argument("--device", default="cuda:0")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=5564)
    serve_parser.set_defaults(function=serve)

    preflight_parser = subparsers.add_parser(
        "robot-preflight", help="read-only live observation and initial-pose gate"
    )
    preflight_parser.add_argument("--task-config", type=Path, required=True)
    preflight_parser.add_argument("--robot-config", type=Path, required=True)
    preflight_parser.add_argument("--dataset", type=Path, required=True)
    preflight_parser.add_argument("--output-dir", type=Path)
    preflight_parser.set_defaults(function=robot_preflight)

    infer_parser = subparsers.add_parser("infer", help="run the verified complete H16 loop")
    infer_parser.add_argument("--task-config", type=Path, required=True)
    infer_parser.add_argument("--robot-config", type=Path, required=True)
    infer_parser.add_argument("--execute", action="store_true")
    infer_parser.add_argument("--max-cycles", type=int, default=0)
    infer_parser.add_argument("--report", type=Path)
    infer_parser.set_defaults(function=infer)

    recover_parser = subparsers.add_parser("recover", help="open and return to training start")
    recover_parser.add_argument("--task-config", type=Path, required=True)
    recover_parser.add_argument("--robot-config", type=Path, required=True)
    recover_parser.add_argument("--execute", action="store_true")
    recover_parser.add_argument("--report", type=Path)
    recover_parser.set_defaults(function=recover)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
