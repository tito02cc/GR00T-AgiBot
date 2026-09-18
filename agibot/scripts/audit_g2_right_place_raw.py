#!/usr/bin/env python3
"""Read-only G2 right-place prescreen; candidate does not mean visual task success.

Only JSON/NPZ and file metadata are read: no JPEG decoding, hashes, source edits,
trajectory smoothing, frame removal, or robot control. Python >=3.10, NumPy, SciPy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_THRESHOLDS = {
    "max_dt_s": 0.2,
    "min_dt_s": 0.05,
    "max_camera_span_ms": 100.0,
    "max_head_state_offset_ms": 100.0,
    "max_hand_state_offset_ms": 50.0,
    "max_step_translation_m": 0.08,
    "max_step_rotation_rad": 0.15,
    "max_gripper_backtrack": 0.05,
    "min_withdraw_m": 0.1,
    "max_terminal_stability_m": 0.005,
    "min_pre_release_frames": 15,
    "min_post_release_frames": 10,
    "initial_gripper_held_min": -0.55,
    "terminal_gripper_open_max": -0.7,
    "release_hold_frames": 3,
    "terminal_hold_frames": 5,
    "gripper_min": -0.8,
    "gripper_max": 0.02,
    "consistency_atol": 2e-5,
}
REQUIRED = (
    "arrays.npz",
    "frames.jsonl",
    "meta_info.json",
    "quality_report.json",
    "parameters/collector_config.json",
    "parameters/manifest_row.json",
    "parameters/episode_receipt.json",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _flag(row: dict, level: str, code: str, frames=None, detail=None) -> None:
    reason = {"code": code}
    if frames is not None:
        reason["frames"] = [int(i) for i in frames]
    if detail is not None:
        reason["detail"] = detail
    row[f"{level}_reasons"].append(reason)


def _masked(row: dict, level: str, code: str, mask: np.ndarray) -> None:
    indices = np.flatnonzero(mask)
    if len(indices):
        _flag(row, level, code, indices)


def _array(frames: list, key: str, shape: tuple) -> np.ndarray:
    values = [frame[key] for frame in frames]
    if key in ("left_gripper", "right_gripper"):
        values = [value["position"] if isinstance(value, dict) else value for value in values]
    value = np.asarray(values, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{key}: expected finite shape {shape}, got {value.shape}")
    return value


def _compare(
    row: dict,
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: float,
    quaternion_slices=(),
) -> None:
    error = np.abs(actual - expected)
    for start in quaternion_slices:
        a, b = actual[:, start : start + 4], expected[:, start : start + 4]
        # q and -q represent the same orientation, including in redundant NPZ fields.
        sign = np.where(np.sum(a * b, axis=1) < 0, -1.0, 1.0)
        error[:, start : start + 4] = np.abs(a - sign[:, None] * b)
    per_frame = error if error.ndim == 1 else error.max(axis=1)
    row["metrics"].setdefault("consistency_max_error", {})[name] = float(per_frame.max())
    _masked(row, "invalid", name, per_frame > tolerance)


def _identity(
    row: dict,
    frames: list,
    meta: dict,
    config: dict,
    manifest: dict,
    receipt: dict,
    quality: dict,
    profile: dict,
) -> None:
    text = meta.get("text", {})
    text = json.loads(text) if isinstance(text, str) else text
    sources = {"meta": meta.get("task_id"), "manifest": manifest.get("task_name")}
    batches = {"config": config.get("args", {}).get("operator_batch_id")}
    lineage = receipt.get("task_lineage", {})
    if "operator_batch_id" in lineage:
        batches["receipt"] = lineage["operator_batch_id"]
    if "task_name" in lineage:
        sources["receipt"] = lineage["task_name"]
    row["identity"] = {
        "episode_uuid": meta.get("episode_uuid"),
        "created_at": meta.get("created_at"),
        "operator": meta.get("operator"),
        "robot_id": meta.get("robot_id"),
        "job_id": meta.get("job_id"),
        "source_tasks": sources,
        "source_batches": batches,
        "source_prompt": text.get("description"),
        "canonical_task": profile["task_id"],
        "canonical_prompt": profile["canonical_prompt"],
        "scene_tag": frames[0].get("scene_tag"),
        "position_offset": frames[0].get("position_offset"),
    }
    for name, value in sources.items():
        if value not in profile["accepted_source_tasks"]:
            _flag(row, "review", f"source_task:{name}", detail=value)
    for name, value in batches.items():
        if value != profile["expected_batch"]:
            _flag(row, "review", f"source_batch:{name}", detail=value)
    for name, value in {
        "meta": text.get("description"),
        "manifest": manifest.get("prompt"),
    }.items():
        if value not in profile["accepted_source_prompts"]:
            _flag(row, "review", f"source_prompt:{name}", detail=value)
    for name, obj in (("receipt", receipt), ("manifest", manifest)):
        if obj.get("episode") != row["episode"] or obj.get("episode_uuid") != meta.get(
            "episode_uuid"
        ):
            _flag(row, "review", f"identity:{name}")
    if meta.get("episode_id") != row["episode"] or not meta.get("episode_uuid"):
        _flag(row, "review", "identity:meta")
    try:
        created = datetime.fromisoformat(str(meta.get("created_at")).replace("Z", "+00:00"))
        if created.tzinfo is None:
            _flag(row, "review", "created_at_missing_timezone")
    except ValueError:
        _flag(row, "review", "created_at_missing_or_invalid")
    for key, expected in (("pose_frame", "base_link_tf"), ("action_mode", "next_delta")):
        for name, source in (
            ("meta", meta.get("selected", {})),
            ("contract", quality.get("contract", {})),
        ):
            if source.get(key) != expected:
                _flag(row, "invalid", f"{key}:{name}", detail=source.get(key))
        _masked(row, "invalid", key, np.array([f.get(key) != expected for f in frames]))
    for key, allowed in (
        ("task_name", profile["accepted_source_tasks"]),
        ("prompt", profile["accepted_source_prompts"]),
        ("arm_mode", ["right"]),
    ):
        _masked(row, "review", key, np.array([f.get(key) not in allowed for f in frames]))
    _masked(
        row,
        "review",
        "frame_uuid",
        np.array([f.get("episode_uuid") != meta.get("episode_uuid") for f in frames]),
    )
    if manifest.get("arm_mode") != "right":
        _flag(row, "review", "manifest_arm_mode")


def _collector(row: dict, frames: list, meta: dict, manifest: dict, quality: dict) -> None:
    semantic = quality.get("semantic_terminal") or quality.get("xichong_terminal") or {}
    row["metrics"]["collector"] = {
        "semantic": semantic,
        "timing": quality.get("timing", {}),
        "contract": quality.get("contract", {}),
        "annotations": meta.get("annotations", {}),
        "duplicate_previous_counts": quality.get("images", {}).get("duplicate_previous_counts", {}),
    }
    checks = {
        "collector_contract": quality.get("contract", {}).get("ok") is True,
        "collector_warnings": not quality.get("contract", {}).get("warnings"),
        "collector_errors": not quality.get("contract", {}).get("errors"),
        "collector_late_frames": quality.get("timing", {}).get("late_frame_count", 0) == 0,
        "meta_data_validate": meta.get("quality", {}).get("data_validate") is True,
        "meta_errors": not meta.get("quality", {}).get("errors"),
        "meta_warnings": not meta.get("quality", {}).get("warnings"),
        "manifest_quality": str(manifest.get("quality_ok")).lower() == "true",
        "manifest_success": manifest.get("success") == "y",
        "collector_semantic": semantic.get("applicable") is True
        and semantic.get("phase") == "right_place"
        and semantic.get("ok") is True,
    }
    expected_annotations = {
        "success": "y",
        "collision": "n",
        "slip": "n",
        "manual_correction": "n",
        "person_visible": "n",
        "failure_reason": "none",
    }
    checks.update(
        {
            f"annotation:{k}": meta.get("annotations", {}).get(k) == v
            for k, v in expected_annotations.items()
        }
    )
    for code, ok in checks.items():
        if not ok:
            _flag(row, "review", code)
    faults = []
    for i, frame in enumerate(frames):
        safety = frame.get("safety_status", {})
        whole = safety.get("whole_body_status", {}).get("value", {})
        fault = safety.get("ok") is not True or bool(safety.get("errors"))
        fault |= any(
            v not in (0, False, None)
            for k, v in whole.items()
            if k.endswith("_error") or k.endswith("_estop")
        )
        fault |= any(v != 0 for v in frame.get("joint_error_codes", []))
        fault |= any(
            isinstance(frame.get(key), dict) and frame[key].get("err_code", 0) != 0
            for key in ("left_gripper", "right_gripper")
        )
        fault |= any(isinstance(v, dict) and v.get("ok") is False for v in safety.values())
        motion = safety.get("motion_control_status", {}).get("value", "")
        if isinstance(motion, str):
            fault |= any(int(value) != 0 for value in re.findall(r"error_code=(-?\d+)", motion))
        if fault:
            faults.append(i)
    if faults:
        _flag(row, "review", "robot_fault_or_missing_safety", faults)


def _images(
    row: dict, root: Path, frames: list, timestamps: np.ndarray, profile: dict, t: dict
) -> None:
    cameras = profile["cameras"]
    midpoints, source_stamps, image_stats = [], {}, {}
    for camera in cameras:
        metadata = [f.get("image_meta", {}).get(camera, {}) for f in frames]
        mids = np.array(
            [m.get("software_midpoint_monotonic", np.nan) for m in metadata], dtype=float
        )
        midpoints.append(mids)
        _masked(row, "review", f"software_timestamp_missing:{camera}", ~np.isfinite(mids))
        offsets = np.abs(mids - timestamps) * 1000
        limit = t[
            "max_head_state_offset_ms" if camera == "head_color" else "max_hand_state_offset_ms"
        ]
        _masked(row, "review", f"state_offset:{camera}", offsets > limit)
        _masked(
            row, "review", f"software_time_nonmonotonic:{camera}", np.r_[False, np.diff(mids) <= 0]
        )
        _masked(
            row,
            "review",
            f"duplicate_image:{camera}",
            np.array([bool(m.get("duplicate_previous")) for m in metadata]),
        )
        stamps = [m.get("timestamp_ns") for m in metadata]
        source_stamps[camera] = stamps
        source_delta = [
            int(b) - int(a) if isinstance(a, int) and isinstance(b, int) else None
            for a, b in zip(stamps[:-1], stamps[1:])
        ]
        backwards = [
            i + 1 for i, delta in enumerate(source_delta) if delta is not None and delta < 0
        ]
        if backwards:
            _flag(row, "review", f"source_time_backwards:{camera}", backwards)
        count, size = 0, 0
        for i, frame in enumerate(frames):
            relative = f"images/{camera}_{i:06d}.jpg"
            if frame.get("images", {}).get(camera) != relative:
                _flag(row, "invalid", f"image_mapping:{camera}", [i])
                continue
            path = root / relative
            if not path.is_file() or path.is_symlink():
                _flag(row, "invalid", f"image_missing_or_symlink:{camera}", [i])
                continue
            file_size = path.stat().st_size
            if file_size == 0:
                _flag(row, "invalid", f"image_empty:{camera}", [i])
            count, size = count + 1, size + file_size
        finite_offsets = offsets[np.isfinite(offsets)]
        image_stats[camera] = {
            "files": count,
            "bytes": size,
            "max_state_offset_ms": float(finite_offsets.max()) if len(finite_offsets) else None,
            "source_timestamp_missing_frames": [
                i for i, s in enumerate(stamps) if not isinstance(s, int)
            ],
            "source_timestamp_repeated_frames": [
                i + 1 for i, d in enumerate(source_delta) if d == 0
            ],
            "source_timestamp_delta_ns_min": min(
                (d for d in source_delta if d is not None), default=None
            ),
            "source_timestamp_delta_ns_max": max(
                (d for d in source_delta if d is not None), default=None
            ),
        }
    span = np.ptp(np.stack(midpoints), axis=0) * 1000
    _masked(row, "review", "camera_span", span > t["max_camera_span_ms"])
    pairs = list(zip(*(source_stamps[c] for c in cameras)))
    row["metrics"]["images"] = image_stats
    row["metrics"]["train_image_bytes"] = sum(s["bytes"] for s in image_stats.values())
    row["metrics"]["max_camera_span_ms"] = (
        float(np.nanmax(span)) if np.isfinite(span).any() else None
    )
    row["metrics"]["source_timestamp_pairs"] = {
        "repeated_adjacent_frames": [i for i in range(1, len(pairs)) if pairs[i] == pairs[i - 1]],
        "same_timestamp_across_cameras_frames": [
            i
            for i, pair in enumerate(pairs)
            if all(isinstance(x, int) for x in pair) and len(set(pair)) == 1
        ],
        "cross_camera_clock_domain_verified": False,
        "note": "No source-to-software or cross-camera source timestamp subtraction performed.",
    }


def _numeric(row: dict, root: Path, frames: list, profile: dict, t: dict) -> None:
    n, atol = len(frames), t["consistency_atol"]
    left = _array(frames, "left_ee_pose", (n, 7))
    right = _array(frames, "right_ee_pose", (n, 7))
    target = _array(frames, "next_right_ee_pose", (n, 7))
    grip = _array(frames, "right_gripper", (n,))
    left_grip = _array(frames, "left_gripper", (n,))
    next_grip = _array(frames, "next_right_gripper", (n,))
    action = _array(frames, "action_right_7d", (n, 7))
    left_action = _array(frames, "action_left_7d", (n, 7))
    action14 = _array(frames, "action_14d", (n, 14))
    ts = _array(frames, "timestamp_monotonic", (n,))
    for label, pose in (("current", right), ("next", target), ("left", left)):
        norm_error = np.abs(np.linalg.norm(pose[:, 3:], axis=1) - 1)
        _masked(row, "invalid", f"quaternion_norm:{label}", norm_error > atol)
    if row["invalid_reasons"]:
        return
    expected = {
        "states": np.column_stack((left, left_grip, right, grip)),
        "ee_poses": np.column_stack((left, right)),
        "grippers": np.column_stack((left_grip, grip)),
        "actions": action14,
        "timestamps_monotonic": ts,
    }
    with np.load(root / "arrays.npz", allow_pickle=False) as arrays:
        for name, reference in expected.items():
            value = np.asarray(arrays[name], dtype=float)
            if value.shape != reference.shape or not np.isfinite(value).all():
                raise ValueError(
                    f"NPZ {name}: expected finite shape {reference.shape}, got {value.shape}"
                )
            _compare(
                row,
                f"npz:{name}",
                value,
                reference,
                atol,
                {"states": (3, 11), "ee_poses": (3, 10)}.get(name, ()),
            )
    _compare(row, "action14_composition", action14, np.column_stack((left_action, action)), atol)
    _compare(row, "delta_xyz_reconstruction", right[:, :3] + action[:, :3], target[:, :3], atol)
    rotation_error = (
        (Rotation.from_rotvec(action[:, 3:6]) * Rotation.from_quat(right[:, 3:])).inv()
        * Rotation.from_quat(target[:, 3:])
    ).magnitude()
    _masked(row, "invalid", "delta_rotation_reconstruction", rotation_error > atol)
    row["metrics"]["consistency_max_error"]["delta_rotation_rad"] = float(rotation_error.max())
    _compare(row, "action_next_gripper", action[:, 6], next_grip, atol)
    if n > 1:
        _compare(row, "next_current_pose", target[:-1], right[1:], atol, (3,))
        continuity_angle = (
            Rotation.from_quat(target[:-1, 3:]).inv() * Rotation.from_quat(right[1:, 3:])
        ).magnitude()
        _masked(row, "invalid", "next_current_rotation", continuity_angle > atol)
        row["metrics"]["consistency_max_error"]["next_current_rotation_rad"] = float(
            continuity_angle.max()
        )
        _compare(row, "next_current_gripper", next_grip[:-1], grip[1:], atol)
    _masked(
        row,
        "invalid",
        "frame_index",
        np.array([f.get("frame_index") != i for i, f in enumerate(frames)]),
    )
    dt = np.diff(ts)
    _masked(row, "invalid", "timestamp_nonmonotonic", np.r_[False, dt <= 0])
    _masked(
        row, "review", "sample_interval", np.r_[False, (dt < t["min_dt_s"]) | (dt > t["max_dt_s"])]
    )
    for name, values in (("current", grip), ("next", next_grip)):
        _masked(
            row,
            "invalid",
            f"gripper_range:{name}",
            (values < t["gripper_min"]) | (values > t["gripper_max"]),
        )
    step_xyz = np.linalg.norm(action[:, :3], axis=1)
    step_rotation = np.linalg.norm(action[:, 3:6], axis=1)
    backtrack = grip - np.minimum.accumulate(grip)
    _masked(row, "review", "step_translation", step_xyz > t["max_step_translation_m"])
    _masked(row, "review", "step_rotation", step_rotation > t["max_step_rotation_rad"])
    _masked(row, "review", "gripper_backtrack", backtrack > t["max_gripper_backtrack"])
    opened = grip <= t["terminal_gripper_open_max"]
    hold, terminal = int(t["release_hold_frames"]), int(t["terminal_hold_frames"])
    release = next((i for i in range(n - hold + 1) if opened[i : i + hold].all()), None)
    row["release_index"] = release
    if grip[0] < t["initial_gripper_held_min"]:
        _flag(row, "review", "initial_not_held", [0])
    if n < terminal or not opened[-terminal:].all():
        _flag(row, "review", "terminal_not_open", range(max(0, n - terminal), n))
    xyz, withdraw = right[:, :3], None
    if release is None:
        _flag(row, "review", "release_missing")
    else:
        withdraw = float(np.linalg.norm(xyz[-1] - xyz[release]))
        _masked(row, "review", "reclose_after_release", (np.arange(n) >= release + hold) & ~opened)
        if release < t["min_pre_release_frames"]:
            _flag(row, "review", "short_pre_release", [release])
        if n - 1 - release < t["min_post_release_frames"]:
            _flag(row, "review", "short_post_release", [release])
        if withdraw < t["min_withdraw_m"]:
            _flag(row, "review", "withdraw_short", [release, n - 1])
    stability = float(np.linalg.norm(xyz[-terminal:] - xyz[-terminal:].mean(axis=0), axis=1).max())
    if stability > t["max_terminal_stability_m"]:
        _flag(row, "review", "terminal_unstable", range(max(0, n - terminal), n))
    row["metrics"].update(
        {
            "frames": n,
            "duration_s": float(ts[-1] - ts[0]),
            "dt_min_s": float(dt.min()) if len(dt) else None,
            "dt_max_s": float(dt.max()) if len(dt) else None,
            "dt_median_s": float(np.median(dt)) if len(dt) else None,
            "step_translation_max_m": float(step_xyz.max()),
            "step_rotation_max_rad": float(step_rotation.max()),
            "gripper_backtrack_max": float(backtrack.max()),
            "withdraw_m": withdraw,
            "terminal_stability_m": stability,
            "inactive_left_displacement_m": float(
                np.linalg.norm(left[:, :3] - left[0, :3], axis=1).max()
            ),
            "path_length_m": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
            "last_next_rule": "Last saved next is checked intrinsically; no fabricated next row or zero action.",
        }
    )
    key_frames = {0, n // 2, n - 1}
    if release is not None:
        key_frames.update(max(0, min(n - 1, release + offset)) for offset in (-3, -1, 0, 1, 3, 5))
    row["key_frames"] = sorted(key_frames)
    row["trajectory"] = {
        "timestamps": ts.tolist(),
        "xyz": xyz.tolist(),
        "quaternion": right[:, 3:].tolist(),
        "gripper": grip.tolist(),
    }
    _images(row, root, frames, ts, profile, t)


def audit_episode(episode_dir: Path, profile: dict) -> dict:
    """Audit one episode independently; malformed input becomes an invalid report row."""
    episode_dir = Path(episode_dir)
    row = {
        "episode": episode_dir.name,
        "status": "invalid",
        "invalid_reasons": [],
        "review_reasons": [],
        "metrics": {},
        "identity": {},
        "release_index": None,
        "key_frames": [],
        "trajectory": {},
    }
    try:
        t = {**DEFAULT_THRESHOLDS, **profile.get("thresholds", {})}
        for relative in REQUIRED:
            if not (episode_dir / relative).is_file():
                _flag(row, "invalid", "missing_required_file", detail=relative)
        if row["invalid_reasons"]:
            return row
        files = [p for p in episode_dir.rglob("*") if p.is_file() and not p.is_symlink()]
        row["metrics"].update(
            episode_total_bytes=sum(p.stat().st_size for p in files), total_file_count=len(files)
        )
        frames = [
            json.loads(line)
            for line in (episode_dir / "frames.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if len(frames) < 2:
            raise ValueError("frames.jsonl must contain at least two saved frames")
        meta, quality, config, manifest, receipt = [_json(episode_dir / p) for p in REQUIRED[2:]]
        _identity(row, frames, meta, config, manifest, receipt, quality, profile)
        _collector(row, frames, meta, manifest, quality)
        _numeric(row, episode_dir, frames, profile, t)
    except Exception as exc:  # A corrupt episode must not abort independent episodes.
        _flag(row, "invalid", "read_or_schema_error", detail=f"{type(exc).__name__}: {exc}")
    row["status"] = (
        "invalid" if row["invalid_reasons"] else "review" if row["review_reasons"] else "candidate"
    )
    return row


def run_audit(raw_root: Path, profile: dict) -> dict:
    """Return a report without writing files. Gaps in original episode IDs are valid."""
    raw_root = Path(raw_root).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    for key in (
        "task_id",
        "expected_batch",
        "accepted_source_tasks",
        "accepted_source_prompts",
        "canonical_prompt",
        "cameras",
    ):
        if not profile.get(key):
            raise ValueError(f"profile requires nonempty {key}")
    if profile["cameras"] != ["head_color", "hand_right"]:
        raise ValueError("this right-arm audit requires cameras ['head_color', 'hand_right']")
    t = {**DEFAULT_THRESHOLDS, **profile.get("thresholds", {})}
    if set(t) != set(DEFAULT_THRESHOLDS) or not all(np.isfinite(float(v)) for v in t.values()):
        raise ValueError("unknown or nonfinite threshold")
    if min(t["release_hold_frames"], t["terminal_hold_frames"]) < 1:
        raise ValueError("release/terminal hold frame counts must be positive")
    paths = sorted(
        p for p in raw_root.iterdir() if p.is_dir() and re.fullmatch(r"episode_\d{6}", p.name)
    )
    if not paths:
        raise ValueError(f"no episode directories under {raw_root}")
    rows = [audit_episode(p, profile) for p in paths]
    counts = Counter(row["status"] for row in rows)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "profile": {**profile, "thresholds": t},
        "scope": "Metadata/NPZ consistency, trajectory, software sync, image paths/stat only. No image decode, visual task-success judgment, hashes, source edits, or robot control.",
        "summary": {
            "total": len(rows),
            **{s: counts[s] for s in ("candidate", "review", "invalid")},
        },
        "reason_counts": {
            level: dict(
                Counter(reason["code"] for row in rows for reason in row[f"{level}_reasons"])
            )
            for level in ("invalid", "review")
        },
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.raw_root.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents:
        parser.error("output-dir must be outside raw-root; source data is read-only")
    report = run_audit(source, _json(args.profile))
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    for status, name in (("candidate", "candidates"), ("review", "review"), ("invalid", "invalid")):
        (output / f"{name}.txt").write_text(
            "".join(r["episode"] + "\n" for r in report["episodes"] if r["status"] == status)
        )
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
