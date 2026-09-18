"""Small synthetic checks for task mapping, strict manifests and source preservation."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation


sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_g2_right_place_training import (
    check_splits,
    convert_split,
    prescreen_one,
    read_manifest,
)
from prepare_g2_task import build_command
from test_audit_g2_right_place_raw import BATCH, PROMPT, TASK, make_episode, write_frames
from validate_g2_right_arm_gr00t_conversion import validate_video
from verify_g2_training_math import relative_windows


@pytest.mark.parametrize(
    "text", ["", "episode_000001\nepisode_000001", "../../episode_000001", "episode_abcdef"]
)
def test_bad_manifest(tmp_path, text):
    path = tmp_path / "list.txt"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_manifest(path)


def test_overlap():
    with pytest.raises(ValueError):
        check_splits(["episode_000000"], ["episode_000000"])


def test_task_type_is_explicit():
    with pytest.raises(ValueError):
        build_command({"pipeline": "grasp"}, "all")


def test_noncommuting_rotation_relative_math():
    from gr00t.data.state_action.action_chunking import EndEffectorActionChunk
    from gr00t.data.state_action.pose import EndEffectorPose
    from gr00t.data.types import ActionFormat

    rng = np.random.default_rng(3)
    state = np.c_[
        rng.normal(size=(20, 3)),
        Rotation.random(20, random_state=rng).as_matrix()[:, :2].reshape(20, 6),
        np.zeros(20),
    ]
    action = np.c_[
        rng.normal(size=(20, 3)),
        Rotation.random(20, random_state=rng).as_matrix()[:, :2].reshape(20, 6),
        np.linspace(0, -0.785, 20),
    ]
    independent = relative_windows(state, action)
    for start in range(5):
        ref = EndEffectorPose.from_action_format(state[start, :9], ActionFormat.XYZ_ROT6D)
        chunk = EndEffectorActionChunk.from_array(
            action[start : start + 16, :9], ActionFormat.XYZ_ROT6D
        )
        official = chunk.relative_chunking(reference_frame=ref).to(ActionFormat.XYZ_ROT6D)
        np.testing.assert_allclose(independent[start], official, atol=1e-6)


def test_real_conversion_preserves_source_and_full_video_correspondence(tmp_path):
    source = tmp_path / "raw"
    episode = make_episode(source)
    terminal_frames = [
        json.loads(line) for line in (episode / "frames.jsonl").read_text().splitlines()
    ]
    terminal_frames[-1]["next_right_ee_pose"][0] += 0.002
    terminal_frames[-1]["action_right_7d"][0] = 0.002
    terminal_frames[-1]["action_14d"][7] = 0.002
    write_frames(episode, terminal_frames)
    for camera in ("head_color", "hand_right"):
        for index in range(48):
            Image.new("RGB", (640, 480), (index * 4, 40, 70)).save(
                episode / "images" / f"{camera}_{index:06d}.jpg"
            )
    profile = {
        "task_id": "synthetic_place",
        "expected_batch": BATCH,
        "accepted_source_tasks": [TASK],
        "accepted_source_prompts": [PROMPT],
        "canonical_prompt": "Place the held part at the bending station and withdraw.",
        "cameras": ["head_color", "hand_right"],
    }
    original = (episode / "frames.jsonl").read_text()
    row = prescreen_one((source, episode.name, profile))
    assert row["status"] == "candidate", row
    out = tmp_path / "converted"
    result = convert_split(source, out, [episode.name], profile, {episode.name: row}, 1, 18)
    assert result["frames"] == 48
    assert (episode / "frames.jsonl").read_text() == original
    task = json.loads((out / "meta/tasks.jsonl").read_text())
    assert task["task"] == profile["canonical_prompt"]
    frames = [json.loads(line) for line in original.splitlines()]
    data = pd.read_parquet(out / "data/chunk-000/episode_000000.parquet")
    action = np.stack(data["action"])
    # Preserve every partial jaw target and the final collector next observation.
    np.testing.assert_allclose(action[:, 9], [f["next_right_gripper"] for f in frames], atol=1e-7)
    np.testing.assert_allclose(
        action[:, :3], np.array([f["next_right_ee_pose"][:3] for f in frames]), atol=1e-7
    )
    report = validate_video(
        out / "videos/chunk-000/observation.images.head_color/episode_000000.mp4",
        episode / "images",
        "head_color",
        48,
        10.0,
        30,
        True,
    )
    assert all(report["checks"].values()), report
    assert report["source_frames_compared"] == 48
    # An interior frame, missed by legacy first/middle/last sampling, must fail.
    Image.new("RGB", (640, 480), "white").save(episode / "images/head_color_000009.jpg")
    corrupted = validate_video(
        out / "videos/chunk-000/observation.images.head_color/episode_000000.mp4",
        episode / "images",
        "head_color",
        48,
        10.0,
        30,
        True,
    )
    assert not corrupted["checks"]["source_frame_correspondence"]
    with pytest.raises(FileExistsError):
        convert_split(source, out, [episode.name], profile, {episode.name: row}, 1, 18)


def test_official_loader_stats_and_processor_end_to_end(tmp_path, monkeypatch):
    import runpy
    import shutil

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.stats import generate_rel_stats, generate_stats
    from gr00t.data.types import EmbodimentTag
    from verify_g2_training_math import verify

    config = Path(__file__).resolve().parents[1] / "configs/zhewan_right_place_config.py"
    runpy.run_path(str(config))
    profile = {
        "task_id": "synthetic_place",
        "expected_batch": BATCH,
        "accepted_source_tasks": [TASK],
        "accepted_source_prompts": [PROMPT],
        "canonical_prompt": "Place the held part at the bending station and withdraw.",
        "cameras": ["head_color", "hand_right"],
    }
    for index, split in enumerate(("train", "heldout")):
        raw = tmp_path / "raw"
        episode = make_episode(raw, index)
        for camera in ("head_color", "hand_right"):
            for frame in range(48):
                Image.new("RGB", (640, 480), (frame * 4, 40, 70)).save(
                    episode / "images" / f"{camera}_{frame:06d}.jpg"
                )
        row = prescreen_one((raw, episode.name, profile))
        assert row["status"] == "candidate"
        convert_split(raw, tmp_path / split, [episode.name], profile, {episode.name: row}, 1, 18)
    generate_stats(tmp_path / "train")
    generate_rel_stats(tmp_path / "train", EmbodimentTag.NEW_EMBODIMENT)
    for filename in ("stats.json", "relative_stats.json"):
        shutil.copyfile(tmp_path / "train/meta" / filename, tmp_path / "heldout/meta" / filename)
    # The real CLI runs stats and verification in separate processes. Recreate
    # that fresh registry for this in-process synthetic test.
    monkeypatch.delitem(MODALITY_CONFIGS, EmbodimentTag.NEW_EMBODIMENT.value)
    result = verify(tmp_path / "train", tmp_path / "heldout", config)
    assert result["status"] == "PASS"
    assert result["counts"]["train"]["complete_h16_windows"] == 33
    assert result["counts"]["heldout"]["official_video_frames"] == 96
    assert result["production_normalization"]["use_percentiles"] is False
    preprocessing = json.loads((tmp_path / "train/meta/training_preprocessing.json").read_text())
    assert preprocessing["official_finetune_flag"] == "--no-use-percentiles"


def test_failed_rerun_cannot_leave_a_stale_pass(tmp_path, monkeypatch):
    from prepare_g2_right_place_training import main

    report = tmp_path / "report"
    report.mkdir()
    status = report / "preparation_status.json"
    status.write_text('{"status":"AUTOMATED_DATA_CHECKS_PASS"}')
    argv = ["prepare", "--report-dir", str(report)]
    for key in (
        "raw-root",
        "train-manifest",
        "heldout-manifest",
        "profile",
        "modality-config",
        "output-root",
    ):
        argv += ["--" + key, str(tmp_path / "does-not-exist")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError):
        main()
    assert json.loads(status.read_text())["status"] == "FAIL"


@pytest.mark.parametrize(
    "recorded,override,expected",
    [
        (None, None, "--use-percentiles"),
        (False, None, "--no-use-percentiles"),
        (True, None, "--use-percentiles"),
        (False, "true", "conflict"),
    ],
)
def test_training_launcher_uses_dataset_normalization(tmp_path, recorded, override, expected):
    import os
    import subprocess

    launcher = Path(__file__).resolve().parents[1] / "training/launch_1xa10080.sh"
    script = launcher.read_text()
    # Exercise just the read-only parameter selector, never the launcher itself,
    # its legacy hash stage, model setup, GPU access or training commands.
    block = script[script.index('DATASET_PERCENTILES="true"') : script.index("COMMON_ARGS=(")]
    env = dict(
        os.environ, DATASET=str(tmp_path), PYTHON_ENV=str(Path(sys.executable).parent.parent)
    )
    env.pop("GROOT_USE_PERCENTILES", None)
    if recorded is not None:
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta/training_preprocessing.json").write_text(
            json.dumps({"use_percentiles": recorded})
        )
    if override is not None:
        env["GROOT_USE_PERCENTILES"] = override
    result = subprocess.run(
        ["bash", "-c", block + '\nprintf "%s" "$NORMALIZATION_FLAG"'],
        env=env,
        text=True,
        capture_output=True,
    )
    if expected == "conflict":
        assert result.returncode == 6 and "conflicts" in result.stderr
    else:
        assert result.returncode == 0 and result.stdout == expected
