#!/usr/bin/env python3
"""Clean/recheck the frozen new-grasp splits, convert, and verify locally.

Only the format-level convert_split helper is reused from the placement pipeline;
its placement auditor and main/prepare functions are never called. Admission uses
this batch's own inventory and collector-contract auditor. Raw files are read-only.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[3]
sys.path.insert(0, str(REPO / "agibot/scripts"))
from convert_xichong_right_single_grasp import CAMERAS, load_frames, validate_image, write_json
from prepare_g2_right_place_training import check_splits, convert_split, read_manifest, run
from inventory_raw import inspect_episode
from screen_numeric import inspect
from tqdm import tqdm


def check_frame_contract(frames, prompt):
    """Prevent image reordering or silently changing the task language."""
    if not frames:
        raise ValueError("empty episode")
    for i, frame in enumerate(frames):
        if frame["frame_index"] != i or frame["prompt"] != prompt:
            raise ValueError(f"frame index/prompt mismatch at {i}")
        for camera in CAMERAS:
            if frame["images"][camera] != f"images/{camera}_{i:06d}.jpg":
                raise ValueError(f"noncanonical image mapping at {i}: {camera}")


def check_capture_config(capture):
    args, frames, jaw = (capture[k] for k in ("args", "frames", "gripper_mapping"))
    if (float(args["freq"]), args["img_w"], args["img_h"], args["arm_mode"]) != (10., 640, 480, "right"):
        raise ValueError("Unsupported collection frequency/image size/active arm")
    if (frames["pose_frame"], frames["action_mode"]) != ("base_link_tf", "next_delta"):
        raise ValueError("Unsupported pose frame or raw action mode")
    if (jaw["mode"], jaw["official_norm_open"], jaw["official_norm_closed"]) != (
        "official_norm_open_negative_closed0", -0.785, 0.0
    ):
        raise ValueError("Unexpected gripper encoding; do not silently remap")


def prescreen(raw, name, source, expected, prompt):
    inventory = inspect_episode(raw / name)
    row = inspect(raw, inventory, source)
    try:
        check_capture_config(json.loads((raw / name / "parameters/collector_config.json").read_text()))
        for key in ("uuid", "frames", "prompt", "robot_id", "task_id"):
            if inventory.get(key) != expected.get(key):
                raise ValueError(f"source identity changed since inventory: {key}")
        frames = load_frames(raw / name / "frames.jsonl")
        check_frame_contract(frames, prompt)
        for frame in frames:
            for camera in CAMERAS:
                validate_image(raw / name / frame["images"][camera])
        row["identity"] = {
            "episode_uuid": inventory["uuid"], "robot_id": inventory["robot_id"],
            "task_id": inventory["task_id"], "created_at": inventory["created_at"],
        }
        row["decoded_input_images"] = len(frames) * len(CAMERAS)
    except Exception as exc:
        row["invalid_reasons"].append(f"local_recheck: {type(exc).__name__}: {exc}")
        row["status"] = "invalid"
    row.pop("phase_trajectory", None)
    return row


def prepare(config, stage, status):
    source = json.loads((TASK / "source.json").read_text())
    raw, output = (REPO / config[k] for k in ("raw_root", "output_root"))
    reports = REPO / config["report_dir"]
    if raw.resolve() != (REPO / source["local_raw_root"]).resolve():
        raise ValueError("Wrong raw task root")
    if output.resolve().is_relative_to(raw.resolve()) or reports.resolve().is_relative_to(raw.resolve()):
        raise ValueError("Outputs/reports must not modify raw data")
    if config["fps"] != 10 or config["workers"] < 1 or not 0 <= config["crf"] <= 51:
        raise ValueError("Unsupported fps/workers/CRF")
    splits = {s: read_manifest(TASK / f"{s}.txt") for s in ("train", "heldout")}
    check_splits(splits["train"], splits["heldout"])
    for split, names in splits.items():
        if len(names) != config[f"expected_{split}_count"]:
            raise ValueError(f"Unexpected {split} count")
    selected = splits["train"] + splits["heldout"]
    if set(selected) != set(read_manifest(TASK / "transfer.txt")):
        raise ValueError("Split union differs from approved transfer list")
    workers = config["workers"]
    if stage in ("all", "convert"):
        if any((output / s).exists() for s in splits):
            raise FileExistsError("Existing dataset: use verify, or a new output_root; never overwrite")
        status("LOCAL_RAW_RECHECK")
        inventory = json.loads((REPO / config["source_inventory"]).read_text())
        expected = {r["episode"]: r for r in inventory["episodes"]}
        def audit(name):
            return prescreen(raw, name, source, expected[name], config["canonical_prompt"])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(tqdm(pool.map(audit, selected), total=len(selected), desc="Local raw + JPEG"))
        write_json(reports / "local_raw_audit.json", {
            "counts": dict(Counter(r["status"] for r in rows)), "episodes": rows,
            "policy": "Current batch collector contract; no historical motion thresholds",
        })
        rejected = [r["episode"] for r in rows if r["status"] != "candidate"]
        if rejected:
            raise ValueError(f"Selected episodes need review; manifests unchanged: {rejected}")
        uuids = [r["identity"]["episode_uuid"] for r in rows]
        if not all(uuids) or len(set(uuids)) != len(uuids):
            raise ValueError("Missing/duplicate UUID across splits")
        by_name = {r["episode"]: r for r in rows}
        summary = {"source_unchanged": True, "raw_trajectory_edits": False}
        for split, names in splits.items():
            status(f"CONVERT_{split.upper()}")
            summary[split] = convert_split(raw, output / split, names,
                {"canonical_prompt": config["canonical_prompt"], "source_task_id": source["source_task_id"]},
                by_name, workers, config["crf"])
            shutil.copyfile(TASK / f"{split}.txt", reports / f"{split}.txt")
        write_json(reports / "conversion_summary.json", summary)
    if stage == "convert":
        status("CONVERSION_COMPLETE_VALIDATION_PENDING")
        return
    status("TRAIN_ONLY_STATISTICS")
    run([sys.executable, "gr00t/data/stats.py", "--dataset-path", str(output / "train"),
         "--embodiment-tag", "NEW_EMBODIMENT", "--modality-config-path", str(TASK / "modality.py")])
    for name in ("stats.json", "relative_stats.json"):
        shutil.copyfile(output / "train/meta" / name, output / "heldout/meta" / name)
    for split in splits:
        write_json(output / split / "meta/normalization_provenance.json", {
            "normalization_source": "train", "train_episodes": splits["train"],
            "heldout_used_to_fit": False,
        })
        status(f"AUDIT_{split.upper()}_ALL_FRAMES")
        run([sys.executable, "agibot/scripts/validate_g2_right_arm_gr00t_conversion.py",
             "--raw-root", str(raw), "--dataset", str(output / split),
             "--episode-manifest", str(TASK / f"{split}.txt"),
             "--report", str(reports / f"{split}_conversion_audit.json"),
             "--workers", str(workers), "--all-source-frames", "--minimum-psnr-db", "30"])
    status("OFFICIAL_LOADING_AND_MATH")
    run([sys.executable, "agibot/scripts/verify_g2_training_math.py",
         "--train", str(output / "train"), "--heldout", str(output / "heldout"),
         "--modality-config", str(TASK / "modality.py"),
         "--report", str(reports / "official_loading_and_math.json"),
         "--workers", str(workers), "--normalization-bounds", config["normalization_bounds"]])
    status("AUTOMATED_DATA_CHECKS_PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TASK / "prepare.json")
    parser.add_argument("--stage", choices=("all", "convert", "verify"), default="all")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    reports = REPO / config["report_dir"]
    reports.mkdir(parents=True, exist_ok=True)
    def status(value, error=None):
        write_json(reports / "preparation_status.json", {
            "status": value, "updated_at": datetime.now().astimezone().isoformat(),
            "dataset_root": str(REPO / config["output_root"]), "error": error,
            "model_training_started": False, "physical_grasp_visual_review": "not_certified",
        })
        print(value, flush=True)
    try:
        prepare(config, args.stage, status)
    except BaseException as exc:
        status("FAIL", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
