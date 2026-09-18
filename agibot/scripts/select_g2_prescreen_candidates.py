#!/usr/bin/env python3
"""Compare candidate episode counts without copying data or modifying raw episodes.

Input is the G2 raw prescreen report. Only ``candidate`` episodes are eligible;
with --image-report they must additionally have a successful full-image decode.
Session boundaries are inferred from ALL dated audit rows, not filtered samples.
One whole session is embargoed from training. This reduces temporal adjacency;
it is not proof that scenes, objects, or demonstrations are independent.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def report_rows(report: dict) -> list[dict]:
    rows = report.get("episodes", report.get("rows"))
    if not isinstance(rows, list):
        raise ValueError("Report must contain an episodes or rows list")
    names = [row["episode"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate episode names in report")
    return rows


def chronological_sessions(rows: list[dict], gap_seconds: float = 900) -> list[dict]:
    dated = []
    for row in rows:
        value = row.get("identity", {}).get("created_at")
        if value:
            dated.append((datetime.fromisoformat(value.replace("Z", "+00:00")), row))
        elif row["status"] == "candidate":
            raise ValueError(f"Candidate {row['episode']} has no created_at timestamp")
    try:
        dated.sort(key=lambda pair: (pair[0], pair[1]["episode"]))
    except TypeError as error:
        raise ValueError("created_at mixes timezone-aware and naive timestamps") from error
    sessions = []
    previous = None
    for timestamp, row in dated:
        if (
            previous is None
            or timestamp.date() != previous.date()
            or (timestamp - previous).total_seconds() > gap_seconds
        ):
            sessions.append({"session": f"session_{len(sessions):03d}", "rows": []})
        sessions[-1]["rows"].append(row)
        sessions[-1].setdefault("start", timestamp.isoformat())
        sessions[-1]["end"] = timestamp.isoformat()
        previous = timestamp
    return sessions


def phase_features(row: dict) -> dict[str, np.ndarray | float]:
    """Interpolate real timestamps, preserving release phase and quaternion geometry."""
    trajectory = row["trajectory"]
    time = np.asarray(trajectory["timestamps"], dtype=float)
    xyz = np.asarray(trajectory["xyz"], dtype=float)
    quaternion = np.asarray(trajectory["quaternion"], dtype=float)  # XYZW
    gripper = np.asarray(trajectory["gripper"], dtype=float)
    release = int(row["release_index"])
    if (
        len(time) < 3
        or not 0 < release < len(time) - 1
        or np.any(np.diff(time) <= 0)
        or xyz.shape != (len(time), 3)
        or quaternion.shape != (len(time), 4)
        or gripper.shape != (len(time),)
        or not all(np.isfinite(v).all() for v in (time, xyz, quaternion, gripper))
    ):
        raise ValueError(f"Invalid candidate trajectory: {row['episode']}")
    samples = np.r_[
        np.linspace(time[0], time[release], 20),
        np.linspace(time[release], time[-1], 13)[1:],
    ]
    return {
        "xyz": np.column_stack([np.interp(samples, time, xyz[:, axis]) for axis in range(3)]),
        "quaternion": Slerp(time, Rotation.from_quat(quaternion))(samples).as_quat(),
        "gripper": np.interp(samples, time, gripper),
        "duration": np.tanh((time[-1] - time[0]) / 30),
        "path": np.tanh(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
    }


def stack_features(rows: list[dict]) -> dict[str, np.ndarray]:
    features = [phase_features(row) for row in rows]
    return {key: np.asarray([f[key] for f in features]) for key in features[0]} if features else {}


def kcenter_indices(features: dict[str, np.ndarray], count: int) -> list[int]:
    """Deterministic farthest-point traversal; scales are descriptive, not safety limits."""
    if count == 0:
        return []
    size = len(features.get("xyz", []))
    if not 0 < count <= size:
        raise ValueError(f"Cannot select {count} from {size} candidates")
    xyz = features["xyz"]
    seed = int(np.argmin(np.square(xyz - np.median(xyz, axis=0)).sum(axis=(1, 2))))
    nearest = np.full(size, np.inf)
    selected = []
    for _ in range(count):
        index = seed if not selected else int(np.argmax(nearest))
        selected.append(index)
        # Rotation distance comes from relative unit quaternions, not Euler subtraction.
        dot = np.abs(np.einsum("ntd,td->nt", features["quaternion"], features["quaternion"][index]))
        angle = 2 * np.arccos(np.clip(dot, 0, 1))
        distance = np.square((xyz - xyz[index]) / 0.020).sum(axis=2).mean(axis=1)
        distance += np.square(angle / np.deg2rad(10)).mean(axis=1)
        distance += np.square((features["gripper"] - features["gripper"][index]) / 0.2).mean(axis=1)
        for key in ("duration", "path"):
            distance += 0.1 * np.square(features[key] - features[key][index])
        nearest = np.minimum(nearest, distance)
        nearest[selected] = -np.inf
    return selected


def coverage_mm(query: np.ndarray, references: np.ndarray) -> dict | None:
    if not len(query) or not len(references):
        return None
    a, b = query.reshape(len(query), -1), references.reshape(len(references), -1)
    squared = np.square(a).sum(axis=1)[:, None] + np.square(b).sum(axis=1)[None, :] - 2 * a @ b.T
    nearest = 1000 * np.sqrt(np.maximum(squared.min(axis=1), 0) / query.shape[1])
    return dict(
        zip(
            ("min", "p50", "p90", "p95", "p99", "max"),
            np.quantile(nearest, [0, 0.5, 0.9, 0.95, 0.99, 1]).tolist(),
        )
    )


def accounting(rows: list[dict]) -> dict:
    result = {"episodes": len(rows)}
    for key in ("episode_total_bytes", "total_file_count", "train_image_bytes"):
        values = [row.get("metrics", {}).get(key) for row in rows]
        result[key] = sum(values) if all(value is not None for value in values) else None
    return result


def select_candidates(
    audit: dict,
    counts: list[int],
    heldout_target: int = 60,
    image_report: dict | None = None,
    min_session_candidates: int = 20,
    gap_seconds: float = 900,
) -> tuple[dict, dict[str, list[str]]]:
    if not counts or any(count <= 0 for count in counts) or heldout_target <= 0:
        raise ValueError("Training counts and heldout target must be positive")
    counts = sorted(set(counts))
    rows = report_rows(audit)
    if image_report is not None:
        image_root = image_report.get("source_root")
        audit_root = audit.get("raw_root")
        if image_root and audit_root and Path(image_root) != Path(audit_root):
            raise ValueError("Image decode report belongs to a different raw dataset")
        cameras = image_report.get("cameras")
        if cameras is not None and not {"head_color", "hand_right"}.issubset(cameras):
            raise ValueError("Image report must decode both head_color and hand_right")
    decoded = (
        None
        if image_report is None
        else {row["episode"] for row in report_rows(image_report) if row["status"] == "pass"}
    )
    eligible = {
        row["episode"]: row
        for row in rows
        if row["status"] == "candidate" and (decoded is None or row["episode"] in decoded)
    }
    sessions = chronological_sessions(rows, gap_seconds)
    for session in sessions:
        session["candidates"] = [row for row in session["rows"] if row["episode"] in eligible]
    suitable = [
        session
        for session in sessions
        if len(session["candidates"]) >= min_session_candidates
        and len(eligible) - len(session["candidates"]) >= min(counts)
    ]
    if not suitable:
        raise ValueError(
            "No entire chronological session can supply the held-out set while leaving "
            f"at least {min(counts)} training candidates; do not silently split a session."
        )
    held_session = min(
        suitable, key=lambda s: (abs(len(s["candidates"]) - heldout_target), s["start"])
    )
    heldout = sorted(held_session["candidates"], key=lambda row: row["episode"])
    held_names = {row["episode"] for row in held_session["rows"]}
    pool = sorted(
        (row for name, row in eligible.items() if name not in held_names),
        key=lambda row: row["episode"],
    )
    feasible = [count for count in counts if count <= len(pool)]
    features, held_features = stack_features(pool), stack_features(heldout)
    order = kcenter_indices(features, max(feasible))
    manifests = {"heldout.txt": [row["episode"] for row in heldout]}
    report = {
        "source": audit.get("raw_root", audit.get("source")),
        "input_rows": len(rows),
        "eligible": len(eligible),
        "image_decode_required": image_report is not None,
        "eligibility_note": "Metadata candidates only; no image completion claim unless image_decode_required is true. Visual task success is not established by selection.",
        "session_gap_seconds": gap_seconds,
        "session_day_boundary": True,
        "heldout_reason": "Whole chronological session closest to requested candidate count; ties use earliest start. All rows in its session are embargoed from training.",
        "heldout_target": heldout_target,
        "heldout_session": held_session["session"],
        "heldout": accounting(heldout),
        "training_pool": accounting(pool),
        "feature_policy": {
            "phase_samples": [20, 12],
            "time_alignment": "actual timestamps",
            "xyz_scale_m": 0.02,
            "orientation_scale_deg": 10,
            "gripper_scale": 0.2,
            "duration_path_weight_each": 0.1,
            "duration_path_transform": "tanh(duration_s / 30), tanh(path_m)",
        },
        "coverage_note": "Nearest phase-aligned XYZ RMS distance in mm: sqrt(mean(sum(XYZ error squared))). Cross-held-out distances are diagnostics, not proof against leakage.",
        "sessions": [
            {key: value for key, value in session.items() if key not in ("rows", "candidates")}
            | {
                "all_episode_count": len(session["rows"]),
                "candidate_count": len(session["candidates"]),
                "episodes": [row["episode"] for row in session["rows"]],
            }
            for session in sessions
        ],
        "infeasible_counts": [count for count in counts if count > len(pool)],
        "alternatives": {},
    }
    for count in feasible:
        chosen = order[:count]
        selected_set = set(chosen)
        training = [pool[index] for index in sorted(chosen)]
        reserve = [row for index, row in enumerate(pool) if index not in selected_set]
        for name, group in (
            (f"train_{count}.txt", training),
            (f"reserve_{count}.txt", reserve),
            (f"candidate_{count}.txt", training + heldout),
        ):
            manifests[name] = sorted(row["episode"] for row in group)
        report["alternatives"][str(count)] = {
            "train": accounting(training),
            "heldout": accounting(heldout),
            "transfer": accounting(training + heldout),
            "reserve": accounting(reserve),
            "training_pool_nearest_xyz_rms_mm": coverage_mm(
                features["xyz"], features["xyz"][chosen]
            ),
            "heldout_nearest_train_xyz_rms_mm": coverage_mm(
                held_features["xyz"], features["xyz"][chosen]
            ),
        }
    return report, manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--image-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[300, 400, 500])
    parser.add_argument("--heldout-target", type=int, default=60)
    args = parser.parse_args()
    try:
        report, manifests = select_candidates(
            json.loads(args.audit_report.read_text()),
            args.counts,
            args.heldout_target,
            None if args.image_report is None else json.loads(args.image_report.read_text()),
        )
        held_path = args.output_dir / "heldout.txt"
        if held_path.exists() and held_path.read_text().splitlines() != manifests["heldout.txt"]:
            raise ValueError(
                "Existing frozen heldout differs; use a new output directory and review the split"
            )
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, episodes in manifests.items():
        (args.output_dir / name).write_text("".join(episode + "\n" for episode in episodes))
    report["audit_report"] = str(args.audit_report.resolve())
    report["image_report"] = str(args.image_report.resolve()) if args.image_report else None
    (args.output_dir / "selection_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "eligible": report["eligible"],
                "heldout": report["heldout"],
                "counts": list(report["alternatives"]),
                "infeasible_counts": report["infeasible_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
