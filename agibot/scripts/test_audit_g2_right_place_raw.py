"""Synthetic, metadata-only regression tests for the raw right-place auditor.

The fake JPEGs deliberately cannot be decoded: this stage checks paths/stat only,
and must not claim that the later full-image validation has already happened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agibot.scripts import audit_g2_right_place_raw as audit
import numpy as np
import pytest


CAMERAS = ("head_color", "hand_right", "hand_left")
BATCH = "synthetic_right_place_job"
TASK = "synthetic_right_place_release_withdraw"
PROMPT = "Place the held workpiece, open the right gripper, and withdraw."


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def read_frames(episode: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (episode / "frames.jsonl").read_text().splitlines()]


def gripper_position(value: Any) -> float:
    return float(value["position"] if isinstance(value, dict) else value)


def write_frames(episode: Path, frames: list[dict[str, Any]], *, sync_npz: bool = True) -> None:
    (episode / "frames.jsonl").write_text(
        "".join(json.dumps(frame) + "\n" for frame in frames), encoding="utf-8"
    )
    if sync_npz:
        np.savez(
            episode / "arrays.npz",
            states=np.array(
                [
                    frame["left_ee_pose"]
                    + [gripper_position(frame["left_gripper"])]
                    + frame["right_ee_pose"]
                    + [gripper_position(frame["right_gripper"])]
                    for frame in frames
                ]
            ),
            actions=np.array([frame["action_14d"] for frame in frames]),
            ee_poses=np.array([frame["left_ee_pose"] + frame["right_ee_pose"] for frame in frames]),
            grippers=np.array(
                [
                    [
                        gripper_position(frame["left_gripper"]),
                        gripper_position(frame["right_gripper"]),
                    ]
                    for frame in frames
                ]
            ),
            timestamps_monotonic=np.array([frame["timestamp_monotonic"] for frame in frames]),
        )


def reconcile_actions(frames: list[dict[str, Any]]) -> None:
    """Rebuild zero-rotation action/next fields after fixture state changes."""
    for index, frame in enumerate(frames):
        following = frames[min(index + 1, len(frames) - 1)]
        for arm in ("left", "right"):
            frame[f"next_{arm}_ee_pose"] = list(following[f"{arm}_ee_pose"])
            next_gripper = gripper_position(following[f"{arm}_gripper"])
            frame[f"next_{arm}_gripper"] = next_gripper
            delta = (
                np.array(following[f"{arm}_ee_pose"][:3]) - np.array(frame[f"{arm}_ee_pose"][:3])
            ).tolist()
            frame[f"action_{arm}_7d"] = delta + [0.0, 0.0, 0.0, next_gripper]
        frame["action_14d"] = frame["action_left_7d"] + frame["action_right_7d"]


def make_episode(root: Path, number: int = 0) -> Path:
    episode = root / f"episode_{number:06d}"
    (episode / "images").mkdir(parents=True)
    (episode / "parameters").mkdir()
    frame_count = 48
    uuid = f"fixture-{number:06d}"
    frames = []
    for index in range(frame_count):
        timestamp = 1000.0 + index * 0.1
        wall = 1_780_000_000.0 + index * 0.1
        x = 0.45 + 0.1 * min(index, 19) / 19
        if index >= 23:
            x = 0.55 - 0.25 * min(index - 23, 16) / 16
        gripper = 0.0 if index < 20 else -0.78 * min(index - 19, 4) / 4
        frame: dict[str, Any] = {
            "frame_index": index,
            "episode_uuid": uuid,
            "pose_frame": "base_link_tf",
            "action_mode": "next_delta",
            "sync_mode": "software_monotonic_gdk_latest_image",
            "timestamp_monotonic": timestamp,
            "timestamp_monotonic_ns": int(timestamp * 1e9),
            "timestamp_wall": wall,
            "arm_mode": "right",
            "task_name": TASK,
            "prompt": PROMPT,
            "scene_tag": "synthetic_station",
            "position_offset": "synthetic_station",
            "safety_status": {"ok": True, "errors": []},
            "joint_error_codes": [0],
            "left_ee_pose": [0.3, 0.2, 0.9, 0.0, 0.0, 0.0, 1.0],
            "right_ee_pose": [x, -0.2, 0.9, 0.0, 0.0, 0.0, 1.0],
            "left_gripper": {"position": -0.78, "err_code": 0},
            "right_gripper": {"position": gripper, "err_code": 0},
            "images": {},
            "image_meta": {},
        }
        for camera, offset in zip(CAMERAS, (-0.03, -0.01, -0.02)):
            relative = f"images/{camera}_{index:06d}.jpg"
            frame["images"][camera] = relative
            frame["image_meta"][camera] = {
                "camera": camera,
                "sync_mode": "software_monotonic_gdk_latest_image",
                "software_request_monotonic": timestamp + offset - 0.0005,
                "software_receive_monotonic": timestamp + offset + 0.0005,
                "software_midpoint_monotonic": timestamp + offset,
                "timestamp_ns": int((wall + offset) * 1e9),
                "duplicate_previous": False,
            }
            (episode / relative).write_bytes(b"metadata-only fake image\n")
        frames.append(frame)
    reconcile_actions(frames)
    write_frames(episode, frames)
    annotations = {
        "success": "y",
        "failure_reason": "none",
        "collision": "n",
        "slip": "n",
        "manual_correction": "n",
        "person_visible": "n",
    }
    write_json(
        episode / "meta_info.json",
        {
            "episode_id": episode.name,
            "episode_uuid": uuid,
            "task_id": TASK,
            "created_at": "2026-09-01T09:00:00+08:00",
            "selected": {
                "pose_frame": "base_link_tf",
                "action_mode": "next_delta",
                "target_fps": 10,
            },
            "quality": {"data_validate": True, "errors": [], "warnings": []},
            "annotations": annotations,
            "text": json.dumps({"description": PROMPT}),
        },
    )
    semantic = {
        "applicable": True,
        "ok": True,
        "phase": "right_place",
        "errors": [],
        "metrics": {
            "saved_frames": frame_count,
            "initial_held": {"right": True},
            "terminal_state": "open",
            "release_anchor_index": 23,
            "withdraw_m": {"right": 0.25},
            "terminal_stability_radius_m": {"right": 0.0},
            "inactive_displacement_m": {"left": 0.0},
            "final_next_barrier_confirmed": True,
        },
    }
    write_json(
        episode / "quality_report.json",
        {
            "contract": {
                "ok": True,
                "errors": [],
                "warnings": [],
                "pose_frame": "base_link_tf",
                "action_mode": "next_delta",
            },
            "timing": {"target_freq_hz": 10.0, "late_frame_count": 0},
            "images": {
                "duplicate_previous_counts": dict.fromkeys(CAMERAS, 0),
                "saved_counts": dict.fromkeys(CAMERAS, frame_count),
                "software_retrieval_sync": {"max_span_ms": 20.0},
            },
            "semantic_terminal": semantic,
            "xichong_terminal": semantic,
        },
    )
    write_json(
        episode / "parameters/collector_config.json",
        {
            "args": {
                "operator_batch_id": BATCH,
                "task_name": TASK,
                "arm_mode": "right",
                "prompt": PROMPT,
                "freq": 10.0,
            }
        },
    )
    write_json(
        episode / "parameters/manifest_row.json",
        {
            "episode": episode.name,
            "episode_uuid": uuid,
            "arm_mode": "right",
            "task_name": TASK,
            "prompt": PROMPT,
            "success": "y",
            "train_samples": str(frame_count),
            "quality_ok": "true",
        },
    )
    write_json(
        episode / "parameters/episode_receipt.json",
        {"episode": episode.name, "episode_id": episode.name, "episode_uuid": uuid},
    )
    return episode


@pytest.fixture
def profile() -> dict[str, Any]:
    return {
        "task_id": "synthetic_right_place",
        "expected_batch": BATCH,
        "accepted_source_tasks": [TASK],
        "accepted_source_prompts": [PROMPT],
        "canonical_prompt": PROMPT,
        "cameras": ["head_color", "hand_right"],
        "thresholds": {},
    }


def reasons(row: dict[str, Any], category: str) -> str:
    return json.dumps(row[f"{category}_reasons"]).lower()


def test_clean_episode_is_only_metadata_candidate(tmp_path: Path, profile: dict) -> None:
    episode = make_episode(tmp_path)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "candidate", row
    assert row["invalid_reasons"] == []
    assert row["episode"] == "episode_000000"
    assert row["release_index"] is not None
    assert row["key_frames"]


def test_legacy_scalar_current_grippers_are_supported(tmp_path: Path, profile: dict) -> None:
    episode = make_episode(tmp_path)
    frames = read_frames(episode)
    for frame in frames:
        for arm in ("left", "right"):
            frame[f"{arm}_gripper"] = gripper_position(frame[f"{arm}_gripper"])
    write_frames(episode, frames)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "candidate", row


def test_gripper_feedback_fault_is_review_not_structural_corruption(
    tmp_path: Path, profile: dict
) -> None:
    episode = make_episode(tmp_path)
    frames = read_frames(episode)
    frames[7]["right_gripper"]["err_code"] = 1
    write_frames(episode, frames)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "review", row
    assert row["invalid_reasons"] == [], row


def test_single_saved_frame_is_invalid(tmp_path: Path, profile: dict) -> None:
    episode = make_episode(tmp_path)
    frames = read_frames(episode)[:1]
    write_frames(episode, frames)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "invalid", row


def test_gapped_episode_ids_and_bad_first_episode_do_not_abort(
    tmp_path: Path, profile: dict
) -> None:
    first = make_episode(tmp_path, 3)
    make_episode(tmp_path, 15)
    make_episode(tmp_path, 810)
    (first / "frames.jsonl").write_text("not json\n", encoding="utf-8")
    report = audit.run_audit(tmp_path, profile)
    by_name = {row["episode"]: row["status"] for row in report["episodes"]}
    assert by_name == {
        "episode_000003": "invalid",
        "episode_000015": "candidate",
        "episode_000810": "candidate",
    }
    assert report["summary"]["total"] == 3
    assert report["summary"]["invalid"] == 1


@pytest.mark.parametrize(
    ("array_name", "index"),
    [
        ("states", (5, 8)),
        ("actions", (5, 7)),
        ("ee_poses", (5, 7)),
        ("grippers", (5, 1)),
        ("timestamps_monotonic", (5,)),
    ],
)
def test_npz_jsonl_numeric_disagreement_is_invalid(
    tmp_path: Path, profile: dict, array_name: str, index: tuple
) -> None:
    episode = make_episode(tmp_path)
    with np.load(episode / "arrays.npz") as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    arrays[array_name][index] += 0.001
    np.savez(episode / "arrays.npz", **arrays)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "invalid", row
    assert array_name in reasons(row, "invalid"), row


def test_next_current_chain_checked_even_when_row_delta_is_consistent(
    tmp_path: Path, profile: dict
) -> None:
    episode = make_episode(tmp_path)
    frames = read_frames(episode)
    frames[5]["next_right_ee_pose"][0] += 0.001
    frames[5]["action_right_7d"][0] += 0.001
    frames[5]["action_14d"] = frames[5]["action_left_7d"] + frames[5]["action_right_7d"]
    write_frames(episode, frames)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "invalid", row
    assert row["invalid_reasons"]


def test_opposite_quaternion_sign_represents_same_rotation(tmp_path: Path, profile: dict) -> None:
    episode = make_episode(tmp_path)
    frames = read_frames(episode)
    for frame in frames:
        frame["next_right_ee_pose"][3:] = [-value for value in frame["next_right_ee_pose"][3:]]
    write_frames(episode, frames)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "candidate", row


@pytest.mark.parametrize("global_negation", [True, False])
def test_npz_quaternion_global_sign_equivalence_but_not_individual_components(
    tmp_path: Path, profile: dict, global_negation: bool
) -> None:
    episode = make_episode(tmp_path)
    frames = read_frames(episode)
    quaternion = [0.1, 0.2, 0.3, float(np.sqrt(0.86))]
    for frame in frames:
        frame["right_ee_pose"][3:] = quaternion.copy()
    reconcile_actions(frames)
    write_frames(episode, frames)
    with np.load(episode / "arrays.npz") as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    for name, start in (("states", 11), ("ee_poses", 10)):
        stop = start + 4 if global_negation else start + 1
        arrays[name][:, start:stop] *= -1
    np.savez(episode / "arrays.npz", **arrays)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == ("candidate" if global_negation else "invalid"), row


def test_float32_storage_roundoff_is_not_corruption(tmp_path: Path, profile: dict) -> None:
    episode = make_episode(tmp_path)
    with np.load(episode / "arrays.npz") as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    for name in ("states", "actions", "ee_poses", "grippers"):
        arrays[name] = arrays[name].astype(np.float32)
    np.savez(episode / "arrays.npz", **arrays)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "candidate", row


def test_nonfinite_current_jaw_is_invalid(tmp_path: Path, profile: dict) -> None:
    episode = make_episode(tmp_path)
    frames = read_frames(episode)
    frames[5]["right_gripper"]["position"] = float("nan")
    write_frames(episode, frames)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "invalid", row


@pytest.mark.parametrize("change", ["mapping", "missing", "empty", "outside_episode"])
def test_required_images_are_checked_without_decoding(
    tmp_path: Path, profile: dict, change: str
) -> None:
    episode = make_episode(tmp_path)
    frames = read_frames(episode)
    image_path = episode / frames[5]["images"]["hand_right"]
    if change == "mapping":
        frames[5]["images"]["hand_right"] = frames[4]["images"]["hand_right"]
    elif change == "missing":
        image_path.unlink()
    elif change == "empty":
        image_path.write_bytes(b"")
    else:
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(b"not part of episode")
        frames[5]["images"]["hand_right"] = str(outside)
    write_frames(episode, frames)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "invalid", row
    assert "image" in reasons(row, "invalid"), row


def test_time_gap_is_review_not_structural_corruption(tmp_path: Path, profile: dict) -> None:
    episode = make_episode(tmp_path)
    frames = read_frames(episode)
    for frame in frames[10:]:
        frame["timestamp_monotonic"] += 0.3
        frame["timestamp_monotonic_ns"] += 300_000_000
        frame["timestamp_wall"] += 0.3
        for camera in CAMERAS:
            meta = frame["image_meta"][camera]
            for key in (
                "software_request_monotonic",
                "software_receive_monotonic",
                "software_midpoint_monotonic",
            ):
                meta[key] += 0.3
            meta["timestamp_ns"] += 300_000_000
    write_frames(episode, frames)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "review", row
    assert row["invalid_reasons"] == [], row
    assert any(word in reasons(row, "review") for word in ("dt", "tim", "gap", "interval")), row


def test_left_camera_only_anomalies_do_not_exclude_right_arm_candidate(
    tmp_path: Path, profile: dict
) -> None:
    episode = make_episode(tmp_path)
    frames = read_frames(episode)
    for frame in frames:
        frame["image_meta"]["hand_left"]["duplicate_previous"] = True
        frame["image_meta"]["hand_left"]["timestamp_ns"] = 1
        frame["image_meta"]["hand_left"]["software_midpoint_monotonic"] -= 2
        (episode / frame["images"]["hand_left"]).unlink()
    quality = json.loads((episode / "quality_report.json").read_text())
    quality["images"]["duplicate_previous_counts"]["hand_left"] = len(frames)
    quality["images"]["software_retrieval_sync"]["max_span_ms"] = 2000.0
    write_json(episode / "quality_report.json", quality)
    write_frames(episode, frames)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "candidate", row


def test_used_camera_stagnation_requires_review(tmp_path: Path, profile: dict) -> None:
    episode = make_episode(tmp_path)
    frames = read_frames(episode)
    for frame in frames[10:13]:
        frame["image_meta"]["hand_right"]["timestamp_ns"] = frames[9]["image_meta"]["hand_right"][
            "timestamp_ns"
        ]
        frame["image_meta"]["hand_right"]["duplicate_previous"] = True
    write_frames(episode, frames)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "review", row
    assert row["invalid_reasons"] == [], row


@pytest.mark.parametrize("change", ["reclose", "no_release"])
def test_release_semantics_are_reviewed_not_called_corrupt(
    tmp_path: Path, profile: dict, change: str
) -> None:
    episode = make_episode(tmp_path)
    frames = read_frames(episode)
    affected = frames[31:34] if change == "reclose" else frames
    for frame in affected:
        frame["right_gripper"]["position"] = -0.1
    reconcile_actions(frames)
    write_frames(episode, frames)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "review", row
    assert row["invalid_reasons"] == [], row
    assert any(word in reasons(row, "review") for word in ("grip", "release", "clos")), row


def test_valid_final_unsaved_next_state_is_not_forced_to_zero(
    tmp_path: Path, profile: dict
) -> None:
    episode = make_episode(tmp_path)
    frames = read_frames(episode)
    frames[-1]["next_right_ee_pose"][0] += 0.001
    frames[-1]["next_right_gripper"] = -0.781
    frames[-1]["action_right_7d"][0] = 0.001
    frames[-1]["action_right_7d"][6] = -0.781
    frames[-1]["action_14d"] = frames[-1]["action_left_7d"] + frames[-1]["action_right_7d"]
    write_frames(episode, frames)
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "candidate", row


def test_identity_mismatch_is_review_and_preserves_source_fields(
    tmp_path: Path, profile: dict
) -> None:
    episode = make_episode(tmp_path)
    config_path = episode / "parameters/collector_config.json"
    config = json.loads(config_path.read_text())
    config["args"]["operator_batch_id"] = "unexpected_batch"
    write_json(config_path, config)
    before = config_path.read_bytes()
    row = audit.audit_episode(episode, profile)
    assert row["status"] == "review", row
    assert row["invalid_reasons"] == [], row
    assert config_path.read_bytes() == before


def test_audit_does_not_mutate_raw_or_compute_hashes(
    tmp_path: Path, profile: dict, monkeypatch
) -> None:
    import hashlib

    make_episode(tmp_path, 10)
    before = {
        str(path.relative_to(tmp_path)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    def forbidden(*args, **kwargs):
        pytest.fail("metadata prescreen must not compute content hashes")

    for name in ("sha256", "sha512", "sha1", "md5", "new", "file_digest"):
        monkeypatch.setattr(hashlib, name, forbidden)
    report = audit.run_audit(tmp_path, profile)
    assert report["summary"]["candidate"] == 1
    after = {
        str(path.relative_to(tmp_path)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
