"""Focused synthetic regression tests; no network, images, or raw data access."""

from datetime import datetime, timedelta

from agibot.scripts.select_g2_prescreen_candidates import (
    chronological_sessions,
    coverage_mm,
    kcenter_indices,
    phase_features,
    select_candidates,
)
import numpy as np
import pytest


def row(number: int, timestamp: datetime, status: str = "candidate") -> dict:
    return {
        "episode": f"episode_{number:06d}",
        "status": status,
        "identity": {"created_at": timestamp.isoformat()},
        "release_index": 2,
        "trajectory": {
            "timestamps": [0, 0.1, 0.3, 0.4, 0.8],
            "xyz": [[number * 0.001 + x, 0, 0] for x in [0, 0.02, 0.04, 0.01, -0.1]],
            "quaternion": [[0, 0, 0, 1]] * 5,
            "gripper": [0, 0, -0.75, -0.78, -0.78],
        },
        "metrics": {"episode_total_bytes": 100, "total_file_count": 20, "train_image_bytes": 70},
    }


def dataset() -> dict:
    start = datetime(2026, 9, 3, 10)
    # Real ID gaps are intentionally preserved; three real sessions, 3/7/6 candidates.
    return {
        "rows": [
            row(group * 100 + index * 2, start + timedelta(days=group, seconds=60 * index))
            for group, count in enumerate((3, 7, 6))
            for index in range(count)
        ]
    }


def test_id_gaps_counts_and_whole_session_disjoint_holdout():
    report, manifests = select_candidates(dataset(), [5, 10, 13, 14], 3, min_session_candidates=2)
    assert manifests["heldout.txt"] == ["episode_000000", "episode_000002", "episode_000004"]
    assert report["infeasible_counts"] == [14]
    for count in (5, 10, 13):
        train, held = set(manifests[f"train_{count}.txt"]), set(manifests["heldout.txt"])
        assert len(train) == count and train.isdisjoint(held)
        assert len(manifests[f"candidate_{count}.txt"]) == count + 3
        assert (
            report["alternatives"][str(count)]["transfer"]["episode_total_bytes"]
            == (count + 3) * 100
        )
    assert manifests["reserve_13.txt"] == []
    assert report["alternatives"]["13"]["reserve"]["episode_total_bytes"] == 0
    assert set(manifests["train_5.txt"]) <= set(manifests["train_10.txt"])


def test_all_rows_bridge_sessions_before_filtering_and_day_break():
    start = datetime(2026, 9, 3, 10)
    rows = [
        row(0, start),
        row(1, start + timedelta(seconds=800), "review"),
        row(2, start + timedelta(seconds=1600)),
    ]
    assert len(chronological_sessions(rows)) == 1
    midnight = datetime(2026, 9, 3, 23, 59, 59)
    assert (
        len(chronological_sessions([row(0, midnight), row(1, midnight + timedelta(seconds=2))]))
        == 2
    )


def test_no_suitable_whole_session_fails_not_split():
    with pytest.raises(ValueError, match="No entire chronological session"):
        select_candidates(dataset(), [14], 3, min_session_candidates=2)


def test_review_invalid_and_failed_images_are_never_selected():
    data = dataset()
    data["rows"][4]["status"] = "review"
    data["rows"][5]["status"] = "invalid"
    images = {
        "episodes": [{"episode": entry["episode"], "status": "pass"} for entry in data["rows"]]
    }
    images["episodes"][-1]["status"] = "fail"
    report, manifests = select_candidates(data, [5], 3, images, min_session_candidates=2)
    all_selected = set(manifests["candidate_5.txt"] + manifests["reserve_5.txt"])
    assert report["eligible"] == 13
    assert not {data["rows"][i]["episode"] for i in (4, 5, -1)} & all_selected


def test_zero_count_guard_and_bad_count():
    assert kcenter_indices({}, 0) == []
    with pytest.raises(ValueError):
        kcenter_indices({}, 1)
    with pytest.raises(ValueError):
        select_candidates(dataset(), [0], 3)


def test_time_alignment_quaternion_sign_and_geometric_rms():
    first = row(0, datetime(2026, 9, 3))
    second = row(10, datetime(2026, 9, 3))
    second["trajectory"]["quaternion"][2] = [0, 0, 0, -1]
    a, b = phase_features(first), phase_features(second)
    assert a["xyz"].shape == (32, 3)
    assert a["xyz"][19, 0] == pytest.approx(0.04)
    assert a["xyz"][1, 0] == pytest.approx(0.02 * (0.3 / 19) / 0.1)
    assert np.abs(b["quaternion"][:, 3]).min() == pytest.approx(1)
    result = coverage_mm(a["xyz"][None], b["xyz"][None])
    assert result["p50"] == pytest.approx(10)
    assert coverage_mm(np.empty((0, 32, 3)), b["xyz"][None]) is None


def test_missing_candidate_timestamp_is_error():
    data = dataset()
    data["rows"][0]["identity"] = {}
    with pytest.raises(ValueError, match="no created_at"):
        select_candidates(data, [5], 3, min_session_candidates=2)


def test_image_report_must_match_dataset_and_policy_cameras():
    data = dataset()
    data["raw_root"] = "/source/batch_a"
    images = {
        "source_root": "/source/batch_b",
        "episodes": [{"episode": r["episode"], "status": "pass"} for r in data["rows"]],
        "cameras": ["head_color", "hand_right"],
    }
    with pytest.raises(ValueError, match="different raw dataset"):
        select_candidates(data, [5], 3, images, min_session_candidates=2)
    images["source_root"] = data["raw_root"]
    images["cameras"] = ["head_color", "hand_left"]
    with pytest.raises(ValueError, match="both head_color and hand_right"):
        select_candidates(data, [5], 3, images, min_session_candidates=2)
