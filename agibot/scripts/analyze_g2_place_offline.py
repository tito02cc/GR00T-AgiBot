#!/usr/bin/env python3
"""Summarize recorded predictions and replay request mapping into an in-memory GDK fake."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agibot.robot.g2_groot_continuous_sender import NativeTrajectorySender  # noqa: E402
from agibot.scripts.evaluate_g2_place_offline import stats  # noqa: E402


class MemoryRobot:
    """No SDK, networking, DDS, hardware or simulated physical dynamics."""

    def __init__(self):
        self.last = None
        self.calls = 0

    def set_reference_frame_poses(self, *args):
        return 0

    def trajectory_tracking_control(self, timestamp, state, actions, **kwargs):
        assert kwargs["robot_link"] == "base_link"
        assert kwargs["trajectory_reference_time"] == 0.02
        self.last = actions[0]
        assert set(self.last) == {"right_arm", "right_effector"}
        assert self.last["right_arm"]["control_type"] == "ABS_POSE"
        assert self.last["right_effector"]["control_type"] == "ABS_JOINT"
        self.calls += 1
        return 0


class MemoryTF:
    def get_tf_from_base_link(self, frame):
        return SimpleNamespace(rotation=SimpleNamespace(x=0, y=0, z=0, w=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.evaluation_dir
    summary = json.loads((root / "summary.json").read_text())
    findings = {}
    total_rows = total_ticks = 0
    all_details = {}
    for split in ("train", "heldout"):
        rows = [json.loads(p.read_text()) for p in sorted(root.glob(f"{split}_*.json"))]
        all_details[split] = rows
        by_horizon = [[] for _ in range(16)]
        times, jumps, in_range_delta, clipping, release_xyz = [], [], [], [], []
        within_chunk_recloses = recorded_reset_recloses = 0
        for row in rows:
            pred, gt = np.asarray(row["pred_eef_gripper"]), np.asarray(row["gt_eef"])
            errors = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1) * 1000
            for h in range(16):
                by_horizon[h].extend(errors[h::16])
            if row["open_timing_error_steps"] is not None:
                times.append(row["open_timing_error_steps"])
                release_xyz.append(
                    float(
                        np.linalg.norm(
                            pred[row["pred_first_open_step"], :3]
                            - gt[row["gt_first_open_step"], :3]
                        )
                        * 1000
                    )
                )
                start = row["pred_first_open_step"]
                for index in range(start + 1, len(pred)):
                    if pred[index, 9] > -0.55 and pred[index - 1, 9] <= -0.55:
                        if index % 16 == 0:
                            recorded_reset_recloses += 1
                        else:
                            within_chunk_recloses += 1
            robot = MemoryRobot()
            sender = NativeTrajectorySender(0.0)
            sender.initialize_references(robot, MemoryTF())
            for chunk in row["chunks"]:
                for target in chunk["full_decoded_targets"]:
                    sender.set_target(target[7])
                    # Pose interpolation/real-time scheduling is tested separately.
                    # This test proves the final waypoint reaches BOTH request fields.
                    for tick in range(5):
                        sender(robot, target[:7])
                    np.testing.assert_allclose(
                        robot.last["right_arm"]["action_data"], target[:7], atol=1e-6
                    )
                    np.testing.assert_allclose(
                        robot.last["right_effector"]["action_data"], [target[7]], atol=1e-6
                    )
                    assert sender.snapshot()["remaining_ticks"] == 0
                    total_rows += 1
            total_ticks += robot.calls
            raw = pred[:, 9]
            clipping.extend(np.abs(raw - np.clip(raw, -0.785, 0)))
            within = (raw >= -0.785) & (raw <= 0)
            in_range_delta.extend(np.abs(raw[within] - np.clip(raw[within], -0.785, 0)))
            jumps.append(row["within_chunk_xyz_step_mm"]["max"])
        findings[split] = {
            "horizon_xyz_error_mm": {str(h + 1): stats(v) for h, v in enumerate(by_horizon)},
            "open_timing_signed_steps": stats(times),
            "open_timing_absolute_steps": stats(np.abs(times)),
            "open_early_episodes": sum(t < 0 for t in times),
            "open_late_episodes": sum(t > 0 for t in times),
            "first_open_target_vs_recorded_first_open_target_mm": stats(release_xyz),
            "reclose_crossings_within_chunk": within_chunk_recloses,
            "reclose_crossings_at_recorded_observation_reset": recorded_reset_recloses,
            "max_native_range_clipping_rad": float(max(clipping)),
            "in_range_mapping_change_rad": float(max(in_range_delta)),
            "within_chunk_step_max_over_50mm_episodes": sum(j > 50 for j in jumps),
        }
    report = {
        "status": "OFFLINE_REQUEST_MAPPING_PASS",
        "splits": findings,
        "decoded_waypoints": total_rows,
        "combined_arm_gripper_requests": total_ticks,
        "ticks_per_waypoint": 5,
        "sender": "NativeTrajectorySender",
        "scope": "in-memory GDK fake; request mapping only, not DDS, physical jaw release, IK, tracking or real-time scheduling",
    }
    (root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = sorted(all_details["heldout"], key=lambda r: r["xyz_error_mm"]["mean"])
    chosen = [rows[0], rows[len(rows) // 2], rows[-1]]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for i, row in enumerate(chosen):
        gt, pred = np.asarray(row["gt_eef"]), np.asarray(row["pred_eef_gripper"])
        t = np.arange(len(gt)) / 10
        axes[i, 0].plot(t, np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1) * 1000)
        axes[i, 0].set_title(f"{row['source']} | XYZ reference error")
        axes[i, 0].set_ylabel("mm")
        axes[i, 1].plot(t, row["gt_gripper"], label="Recorded")
        axes[i, 1].plot(t, pred[:, 9], label="Policy (raw)")
        axes[i, 1].axhline(-0.72, color="gray", ls=":", label="Open criterion")
        axes[i, 1].set_title("Gripper: 0 closed, -0.785 open")
        axes[i, 1].set_ylabel("native radians")
        for ax in axes[i]:
            for k in range(16, len(gt), 16):
                ax.axvline(k / 10, alpha=0.18, color="gray")
            ax.set_xlabel("recorded time (s); vertical lines = new H16")
            ax.grid(alpha=0.2)
        axes[i, 1].legend(loc="best")
    fig.suptitle(
        "Heldout best / median / worst mean XYZ error; recorded observations, not closed loop"
    )
    fig.savefig(root / "heldout_examples.png", dpi=130)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=2))
    print("Heldout XYZ mean:", summary["splits"]["heldout"]["xyz_error_mm"]["mean"])


if __name__ == "__main__":
    main()
