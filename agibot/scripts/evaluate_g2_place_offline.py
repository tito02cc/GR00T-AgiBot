#!/usr/bin/env python3
"""Recorded-observation placement evaluation through the official loopback server.

No robot connections or commands. Predictions do not generate future images;
these are open-loop reference errors, not closed-loop success measurements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agibot.scripts.evaluate_xichong_checkpoints import (  # noqa: E402
    extract_column,
    prepare_observation,
    rot6d_to_matrix,
    rotation_errors_deg,
)
from agibot.scripts.run_g2_groot_full_protected_inference import (  # noqa: E402
    validate_model_modality_config,
)
from agibot.tools.g2_gr00t_shadow_adapter import (  # noqa: E402
    build_policy_observation,
    decode_action_chunk,
    rot6d_to_quaternion_xyzw,
)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy.server_client import PolicyClient  # noqa: E402


def stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return None
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def first_open(values, threshold=-0.72):
    indices = np.flatnonzero(np.asarray(values).reshape(-1) <= threshold)
    return int(indices[0]) if len(indices) else None


def source_rows(dataset):
    rows = json.loads((dataset / "meta/source_episode_map.json").read_text())["episodes"]
    return {row["target_episode_index"]: row for row in rows}


def check_decoded(action):
    eef = np.asarray(action["right_eef"])
    jaw = np.asarray(action["right_gripper"])
    if eef.shape != (1, 16, 9) or jaw.shape != (1, 16, 1):
        raise ValueError(f"Unexpected complete H16: {eef.shape}, {jaw.shape}")
    targets = decode_action_chunk(action)
    np.testing.assert_allclose(targets[:, :3], eef[0, :, :3], atol=1e-7, rtol=0)
    np.testing.assert_allclose(targets[:, 7], np.clip(jaw[0, :, 0], -0.785, 0), atol=1e-7)
    np.testing.assert_allclose(
        Rotation.from_quat(targets[:, 3:7]).as_matrix(),
        rot6d_to_matrix(eef[0, :, 3:]),
        atol=1e-6,
        rtol=0,
    )
    return targets


def check_live_observation_adapter(obs):
    """Compare actual live-adapter conventions against a recorded observation."""
    eef = obs["state"]["right_eef"][0, 0]
    pose = np.r_[eef[:3], rot6d_to_quaternion_xyzw(eef[3:])]
    prompt = obs["language"]["annotation.human.task_description"][0][0]
    live = build_policy_observation(
        obs["video"]["head_color"][0, 0],
        obs["video"]["hand_right"][0, 0],
        pose,
        float(obs["state"]["right_gripper"][0, 0, 0]),
        prompt,
    )
    for group in ("video", "state"):
        for key, value in obs[group].items():
            np.testing.assert_allclose(live[group][key], value, atol=1e-6, rtol=0)
    if live["language"] != obs["language"]:
        raise ValueError("Live prompt differs from recorded prompt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5574)
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--heldout-episode-ids", type=int, nargs="+")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = PolicyClient(host="127.0.0.1", port=args.port, timeout_ms=120000)
    modality = client.get_modality_config()
    validate_model_modality_config(modality)
    maps = {split: source_rows(args.dataset_root / split) for split in ("train", "heldout")}
    identities = {s: {r["identity"]["episode_uuid"] for r in m.values()} for s, m in maps.items()}
    if identities["train"] & identities["heldout"]:
        raise ValueError("Train/heldout episode UUID overlap")
    # Synthetic endpoint/midpoint probe uses the exact real runner decoder.
    probe = {
        "right_eef": np.tile([0.5, -0.2, 0.9, 1, 0, 0, 0, 1, 0], (1, 16, 1)),
        "right_gripper": np.linspace(0, -0.785, 16).reshape(1, 16, 1),
    }
    check_decoded(probe)
    reports, latency = {}, []
    warmup = None
    for split in ("train", "heldout"):
        loader = LeRobotEpisodeLoader(args.dataset_root / split, modality)
        ids = (
            np.linspace(0, len(loader) - 1, args.train_episodes, dtype=int).tolist()
            if split == "train"
            else (
                args.heldout_episode_ids
                if args.heldout_episode_ids is not None
                else list(range(len(loader)))
            )
        )
        rows, errors, angles, jaw_errors = [], [], [], []
        for episode in ids:
            trajectory = loader[episode]
            gt_eef = extract_column(trajectory, "action.right_eef")
            gt_jaw = extract_column(trajectory, "action.right_gripper")[:, 0]
            pred, chunks, chunk_jumps = [], [], []
            client.reset()
            for step in range(0, len(trajectory), 16):
                obs = prepare_observation(trajectory, step, loader, EmbodimentTag.NEW_EMBODIMENT)
                if warmup is None:
                    check_live_observation_adapter(obs)
                    start = time.perf_counter()
                    action, _ = client.get_action(obs)
                    warmup = time.perf_counter() - start
                    check_decoded(action)
                    client.reset()
                start = time.perf_counter()
                action, _ = client.get_action(obs)
                elapsed = time.perf_counter() - start
                latency.append(elapsed)
                targets = check_decoded(action)
                count = min(16, len(trajectory) - step)
                pred.append(
                    np.c_[action["right_eef"][0, :count], action["right_gripper"][0, :count]]
                )
                # Within-chunk consecutive steps only; exclude recorded-observation resets.
                chunk_jumps.extend(
                    np.linalg.norm(np.diff(targets[:count, :3], axis=0), axis=1) * 1000
                )
                chunks.append(
                    {"step": step, "latency_s": elapsed, "full_decoded_targets": targets.tolist()}
                )
            predicted = np.concatenate(pred)
            xyz = np.linalg.norm(predicted[:, :3] - gt_eef[:, :3], axis=1) * 1000
            rotation = rotation_errors_deg(gt_eef[:, 3:], predicted[:, 3:9])
            jaw = np.abs(predicted[:, 9] - gt_jaw)
            gt_open, pred_open = first_open(gt_jaw), first_open(predicted[:, 9])
            recloses = (
                int(
                    np.count_nonzero(
                        (predicted[pred_open + 1 :, 9] > -0.55)
                        & (predicted[pred_open:-1, 9] <= -0.55)
                    )
                )
                if pred_open is not None
                else 0
            )
            row = {
                "episode": episode,
                "source": maps[split][episode]["source_episode"],
                "steps": len(trajectory),
                "calls": len(chunks),
                "xyz_error_mm": stats(xyz),
                "rotation_error_deg": stats(rotation),
                "gripper_mae_rad": float(jaw.mean()),
                "gt_first_open_step": gt_open,
                "pred_first_open_step": pred_open,
                "open_timing_error_steps": pred_open - gt_open
                if gt_open is not None and pred_open is not None
                else None,
                "pred_reclose_crossings_after_open": recloses,
                "pred_gripper_min_max": [
                    float(predicted[:, 9].min()),
                    float(predicted[:, 9].max()),
                ],
                "raw_gripper_outside_native_range_rows": int(
                    np.count_nonzero((predicted[:, 9] < -0.785 - 1e-6) | (predicted[:, 9] > 1e-6))
                ),
                "within_chunk_xyz_step_mm": stats(chunk_jumps),
                "gt_xyz_step_mm": stats(
                    np.linalg.norm(np.diff(gt_eef[:, :3], axis=0), axis=1) * 1000
                ),
                "pred_open_to_final_distance_mm": float(
                    np.linalg.norm(predicted[-1, :3] - predicted[pred_open, :3]) * 1000
                )
                if pred_open is not None
                else None,
            }
            detail = {
                **row,
                "gt_eef": gt_eef.tolist(),
                "gt_gripper": gt_jaw.tolist(),
                "pred_eef_gripper": predicted.tolist(),
                "chunks": chunks,
            }
            (args.output_dir / f"{split}_{episode:04d}.json").write_text(
                json.dumps(detail, indent=2) + "\n"
            )
            rows.append(row)
            errors.extend(xyz)
            angles.extend(rotation)
            jaw_errors.extend(jaw)
            print(
                f"{split} {episode}: H16 calls={len(chunks)}, XYZ mean={xyz.mean():.2f} mm, open GT/pred={gt_open}/{pred_open}",
                flush=True,
            )
        reports[split] = {
            "episodes": len(rows),
            "frames": len(errors),
            "xyz_error_mm": stats(errors),
            "rotation_error_deg": stats(angles),
            "gripper_abs_error_rad": stats(jaw_errors),
            "episode_metrics": rows,
            "missing_predicted_open_episodes": sum(
                r["gt_first_open_step"] is not None and r["pred_first_open_step"] is None
                for r in rows
            ),
            "reclose_after_open_episodes": sum(
                r["pred_reclose_crossings_after_open"] > 0 for r in rows
            ),
        }
    result = {
        "status": "OFFLINE_EXECUTION_PASS_EFFECT_REQUIRES_REVIEW",
        "splits": reports,
        "inference_calls_excluding_warmup": len(latency),
        "warmup_seconds": warmup,
        "loopback_inference_latency_seconds": stats(latency),
        "full_h16_decode": "PASS",
        "live_observation_adapter": "PASS",
        "gripper_endpoints_and_intermediate_mapping": "PASS",
        "seed": "Official server default stochastic RNG; no seed override",
        "scope": "Recorded-observation open-loop, no simulated visual feedback, no robot connection; terminal tails decoded as full H16 but error scored only over real frames",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    client.socket.close(linger=0)
    client.context.term()
    print("OFFLINE_EXECUTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
