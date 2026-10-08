import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

TASK = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("grasp_cloud_audit", TASK / "check_cloud_ready.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
ROOT = Path("/root/gpufree-data/GR00T")


def resolved():
    return {
        "dry_run": True,
        "dataset_paths": [str(ROOT / "datasets/xichong_right_grasp_r0002_400/train")],
        "effective_output_dir": str(ROOT / "outputs/xichong_rgrasp_r0002_n1d7_e400_f10_h16_1xa10080_s30k_ckpt5k_v1"),
        "base_model_path": str(ROOT / "models/GR00T-N1.7-3B"),
        "backbone_model_path": str(ROOT / "models/Cosmos-Reason2-2B"),
        "training": {
            "num_gpus": 1, "global_batch_size": 16, "gradient_accumulation_steps": 2,
            "accumulated_batch_size": 32, "dataloader_num_workers": 4, "max_steps": 30000,
            "save_steps": 5000, "save_total_limit": 2, "logging_steps": 10,
            "learning_rate": 1e-4, "weight_decay": 1e-5, "warmup_ratio": 0.05,
            "max_grad_norm": 1.0, "eval_strategy": "no",
            "lr_scheduler_type": "cosine", "bf16": True, "tf32": True,
            "optim": "adamw_torch", "save_only_model": False,
            "resume_from_checkpoint": False, "transformers_local_files_only": True,
        },
        "model": {
            "tune_llm": False, "tune_visual": False, "tune_projector": True,
            "tune_diffusion_model": True, "tune_vlln": True,
            "state_dropout_prob": 0.2, "use_percentiles": False, "use_relative_action": True,
        },
        "modality": {
            "video_keys": ["head_color", "hand_right"],
            "video_delta_indices": [0], "state_delta_indices": [0],
            "language_keys": ["annotation.human.task_description"],
            "state_keys": ["right_eef", "right_gripper"],
            "action_keys": ["right_eef", "right_gripper"],
            "action_delta_indices": list(range(16)),
            "action_configs": [
                {"rep": "relative", "type": "eef", "format": "xyz+rot6d", "state_key": "right_eef"},
                {"rep": "absolute", "type": "non_eef", "format": "default", "state_key": "right_gripper"},
            ],
        },
    }


def test_expected_config_passes():
    audit.validate_resolved(resolved(), ROOT)


@pytest.mark.parametrize("mutation", ["heldout", "percentiles", "gpus", "accumulation", "retention", "jaw_relative", "horizon", "old_model"])
def test_incompatible_config_rejected(mutation):
    config = copy.deepcopy(resolved())
    if mutation == "heldout":
        config["dataset_paths"].append(str(ROOT / "datasets/xichong_right_grasp_r0002_400/heldout"))
    elif mutation == "percentiles":
        config["model"]["use_percentiles"] = True
    elif mutation == "gpus":
        config["training"]["num_gpus"] = 2
    elif mutation == "accumulation":
        config["training"]["accumulated_batch_size"] = 16
    elif mutation == "retention":
        config["training"]["save_total_limit"] = 5
    elif mutation == "jaw_relative":
        config["modality"]["action_configs"][1]["rep"] = "relative"
    elif mutation == "horizon":
        config["modality"]["action_delta_indices"] = list(range(8))
    else:
        config["base_model_path"] = str(ROOT / "outputs/old_place/checkpoint-30000")
    with pytest.raises(ValueError):
        audit.validate_resolved(config, ROOT)


def test_real_task_env():
    output = subprocess.check_output([
        "bash", "-c",
        'source "$1"; printf "%s\\n" "$GROOT_DATASET" "$GROOT_GLOBAL_BATCH_SIZE" "$GROOT_GRADIENT_ACCUMULATION_STEPS" "$GROOT_MAX_STEPS" "$GROOT_SAVE_STEPS" "$GROOT_SAVE_TOTAL_LIMIT" "$GROOT_USE_PERCENTILES"',
        "bash", str(TASK / "train_1xa10080.env"),
    ], text=True).splitlines()
    assert output == [str(ROOT / "datasets/xichong_right_grasp_r0002_400/train"), "16", "2", "30000", "5000", "2", "false"]


def test_parse_only_official_dry_run_json():
    data = resolved()
    text = 'warning {not json}\n{"files": 1211}\n' + json.dumps(data)
    assert audit.extract_json(text) == data
    with pytest.raises(ValueError):
        audit.extract_json('{"files": 1211}')


@pytest.mark.parametrize("mode,free_gib,expected_code", [
    ("baseline", 79, 7), ("baseline", 80, 0), ("smoke", 39, 7),
    ("smoke", 40, 0), ("resume", 31, 7), ("resume", 32, 0), ("audit", 0, 0),
])
def test_disk_budget_never_starts_training_when_insufficient(tmp_path, mode, free_gib, expected_code):
    import os
    import shutil
    task = tmp_path / "task"
    task.mkdir()
    repo = tmp_path / "repo"
    scripts = repo / "agibot/training"
    scripts.mkdir(parents=True)
    shutil.copyfile(TASK / "train.sh", task / "train.sh")
    (task / "train_1xa10080.env").write_text(f"export CT_ROOT='{tmp_path}'\nexport GROOT_REPO_ROOT='{repo}'\n")
    marker = tmp_path / "launcher_called"
    (scripts / "launch_1xa10080.sh").write_text(f"#!/bin/bash\ntouch '{marker}'\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    df = binaries / "df"
    df.write_text(f"#!/bin/bash\nprintf 'Avail\\n{free_gib * 1024**3}\\n'\n")
    df.chmod(0o755)
    environment = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"])
    process = subprocess.run(["bash", str(task / "train.sh"), mode], env=environment,
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.returncode == expected_code, process.stdout
    assert marker.exists() == (expected_code == 0)
