#!/usr/bin/env python3
"""CPU audit of the actual cloud configuration; not a GPU/model smoke test."""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[3]
sys.path.insert(0, str(REPO / "agibot/training"))
from check_cloud_inputs import check_inventory, check_model


def require(value, expected, label):
    if value != expected:
        raise ValueError(f"{label}: {value!r} != {expected!r}")


def extract_json(text):
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("dry_run") is True:
            return value
    raise ValueError("No official dry-run JSON found")


def validate_resolved(config, root):
    expected_run = "xichong_rgrasp_r0002_n1d7_e400_f10_h16_1xa10080_s30k_ckpt5k_v1"
    require(config["dataset_paths"], [str(root / "datasets/xichong_right_grasp_r0002_400/train")], "train-only dataset")
    require(config["effective_output_dir"], str(root / "outputs" / expected_run), "output")
    require(config["base_model_path"], str(root / "models/GR00T-N1.7-3B"), "base model")
    require(config["backbone_model_path"], str(root / "models/Cosmos-Reason2-2B"), "backbone")
    for key, expected in {
        "num_gpus": 1, "global_batch_size": 16, "gradient_accumulation_steps": 2,
        "accumulated_batch_size": 32, "dataloader_num_workers": 4, "max_steps": 30000,
        "save_steps": 5000, "save_total_limit": 2, "logging_steps": 10,
        "learning_rate": 1e-4, "weight_decay": 1e-5, "warmup_ratio": 0.05,
        "max_grad_norm": 1.0, "eval_strategy": "no",
        "lr_scheduler_type": "cosine", "bf16": True, "tf32": True,
        "optim": "adamw_torch", "save_only_model": False,
        "resume_from_checkpoint": False, "transformers_local_files_only": True,
    }.items():
        require(config["training"][key], expected, "training." + key)
    for key, expected in {
        "tune_llm": False, "tune_visual": False, "tune_projector": True,
        "tune_diffusion_model": True, "tune_vlln": True,
        "state_dropout_prob": 0.2, "use_percentiles": False, "use_relative_action": True,
    }.items():
        require(config["model"][key], expected, "model." + key)
    modality = config["modality"]
    require(modality["video_keys"], ["head_color", "hand_right"], "cameras")
    require(modality["video_delta_indices"], [0], "video time indices")
    require(modality["state_delta_indices"], [0], "state time indices")
    require(modality["state_keys"], ["right_eef", "right_gripper"], "state keys")
    require(modality["action_keys"], ["right_eef", "right_gripper"], "action keys")
    require(modality["action_delta_indices"], list(range(16)), "H16")
    require(modality["language_keys"], ["annotation.human.task_description"], "language")
    require(modality["action_configs"], [
        {"rep": "relative", "type": "eef", "format": "xyz+rot6d", "state_key": "right_eef"},
        {"rep": "absolute", "type": "non_eef", "format": "default", "state_key": "right_gripper"},
    ], "action semantics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/gpufree-data/GR00T"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "RUNNING", "gpu_smoke_test": "NOT_RUN", "training_started": False}
    try:
        root = args.root
        source_sets, stats, split_result = [], [], {}
        for split, count, frames in (("train", 400, 35108), ("heldout", 91, 7358)):
            dataset = root / "datasets/xichong_right_grasp_r0002_400" / split
            split_result[split] = check_inventory(dataset, root / "manifests" / f"xichong_right_grasp_r0002_{split}.inventory.json")
            info = json.loads((dataset / "meta/info.json").read_text())
            require(info["total_episodes"], count, split + " episodes")
            require(info["total_frames"], frames, split + " frames")
            require(info["fps"], 10, split + " fps")
            mapping = json.loads((dataset / "meta/source_episode_map.json").read_text())["episodes"]
            names = [r["source_episode"] for r in mapping]
            require(names, (TASK / f"{split}.txt").read_text().split(), split + " source manifest")
            source_sets.append(set(names))
            normalization = json.loads((dataset / "meta/training_preprocessing.json").read_text())
            require(normalization["use_percentiles"], False, split + " minmax")
            require(normalization["use_relative_action"], True, split + " relative EEF")
            stats.append([json.loads((dataset / "meta" / n).read_text()) for n in ("stats.json", "relative_stats.json")])
        if source_sets[0] & source_sets[1]:
            raise ValueError("Training/heldout overlap")
        require(stats[0], stats[1], "shared train-only stats")
        result["datasets"] = split_result
        result["base_model"] = check_model(root / "models/GR00T-N1.7-3B")
        result["backbone_model"] = check_model(root / "models/Cosmos-Reason2-2B")
        require(result["base_model"]["model_type"], "Gr00tN1d7", "base model type")
        require(result["backbone_model"]["model_type"], "qwen3_vl", "backbone type")
        process = subprocess.run(["bash", str(TASK / "train.sh"), "audit"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (args.report.parent / "config_dry_run.log").write_text(process.stdout)
        print(process.stdout, flush=True)
        if process.returncode:
            raise RuntimeError(f"Config audit exited {process.returncode}")
        resolved = extract_json(process.stdout)
        validate_resolved(resolved, root)
        (args.report.parent / "resolved_training_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
        # Exercise the installed cloud decoder and loader on both ends of each split.
        # This is a deployment smoke check, not a repeat of local all-frame auditing.
        import numpy as np
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        runpy.run_path(str(TASK / "modality.py"))
        modality = MODALITY_CONFIGS["new_embodiment"]
        result["official_loader_samples"] = {}
        for split, count in (("train", 400), ("heldout", 91)):
            loader = LeRobotEpisodeLoader(root / "datasets/xichong_right_grasp_r0002_400" / split,
                modality, decoder_kwargs={"num_ffmpeg_threads": 1})
            require(len(loader), count, split + " loader length")
            samples = []
            for index in (0, count - 1):
                frame = loader[index]
                for camera in ("head_color", "hand_right"):
                    for image in frame[f"video.{camera}"]:
                        array = np.asarray(image)
                        require(array.shape, (480, 640, 3), "decoded image shape")
                        require(array.dtype, np.dtype("uint8"), "decoded image dtype")
                for key in ("state.right_eef", "state.right_gripper", "action.right_eef", "action.right_gripper"):
                    if not np.isfinite(np.stack(frame[key])).all():
                        raise ValueError(f"Cloud loader nonfinite values: {split}/{index}/{key}")
                samples.append({"index": index, "frames": len(frame), "status": "PASS"})
            result["official_loader_samples"][split] = samples
        result["status"] = "CPU_CONFIG_AND_INVENTORY_PASS_GPU_SMOKE_PENDING"
    except BaseException as exc:
        result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.report.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
