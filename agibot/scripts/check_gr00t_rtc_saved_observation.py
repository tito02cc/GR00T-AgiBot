#!/usr/bin/env python3
"""Exercise baseline + RTC on saved images/state. No robot connections/actions.

This is an interface/conditioning check on a static observation, not an offline
task-success score, real-time rollout, or evidence of reduced physical jitter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-port", type=int, default=5565)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--frozen", type=int, default=4)
    parser.add_argument("--ramp-rate", type=float, default=3.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.frozen <= args.overlap < 16:
        parser.error("require 0 < frozen <= overlap < 16")
    if not np.isfinite(args.ramp_rate) or args.ramp_rate <= 0:
        parser.error("ramp-rate must be finite and positive")

    from agibot.rtc.adapter import load_saved_observation, targets_to_prefix
    from agibot.rtc.contract import validate_capability, validate_receipt
    from agibot.tools.g2_gr00t_shadow_adapter import decode_action_chunk

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "saved_observation_interface_only", "robot_commands_sent": 0}
    policy = None
    with args.report.open("x") as output:
        try:
            observation = load_saved_observation(args.snapshot_dir, args.prompt)
            if args.model_path is not None:
                from agibot.rtc.policy import RtcGr00tPolicy
                from gr00t.data.embodiment_tags import EmbodimentTag

                policy = RtcGr00tPolicy(
                    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                    model_path=str(args.model_path),
                    device=args.device,
                    strict=True,
                )
                capability = policy.get_rtc_config()
            else:
                from gr00t.policy.server_client import PolicyClient

                policy = PolicyClient(host="127.0.0.1", port=args.model_port, timeout_ms=60000)
                capability = policy.call_endpoint("get_rtc_config", requires_input=False)
            report["capability"] = capability
            validate_capability(capability)
            start = time.monotonic()
            baseline, _ = policy.get_action(observation)
            report["baseline_inference_s"] = time.monotonic() - start
            # Keep the same time origin as this unchanged observation. A tail
            # from steps 8..15 paired with a step-0 observation is NOT an aligned
            # RTC request; that requires a later, timestamped observation.
            previous_targets = decode_action_chunk(baseline)[: args.overlap]
            report["prefix_source"] = "baseline_first_O_at_same_saved_observation_synthetic"
            prefix = targets_to_prefix(previous_targets)
            start = time.monotonic()
            predicted, info = policy.get_action(
                observation,
                options={
                    "rtc": {
                        "previous_actions": prefix,
                        "frozen_steps": args.frozen,
                        "ramp_rate": args.ramp_rate,
                    }
                },
            )
            report["rtc_inference_s"] = time.monotonic() - start
            report["rtc_info"] = info
            validate_receipt(
                info, overlap=args.overlap, frozen=args.frozen, ramp_rate=args.ramp_rate
            )
            targets = decode_action_chunk(predicted)
            exact = all(
                np.array_equal(predicted[key][:, : args.frozen], value[:, : args.frozen])
                for key, value in prefix.items()
            )
            report.update(
                exact_frozen_physical_fields=exact,
                baseline_targets=decode_action_chunk(baseline).tolist(),
                previous_targets=previous_targets.tolist(),
                rtc_targets=targets.tolist(),
                frozen_decoded_position_max_error_m=float(
                    np.linalg.norm(
                        targets[: args.frozen, :3] - previous_targets[: args.frozen, :3], axis=1
                    ).max()
                ),
                frozen_decoded_gripper_max_error_rad=float(
                    np.abs(targets[: args.frozen, 7] - previous_targets[: args.frozen, 7]).max()
                ),
            )
            if not exact or targets.shape != (16, 8) or not np.isfinite(targets).all():
                raise RuntimeError("RTC output failed physical-prefix/horizon check")
            report["status"] = "PASS_INTERFACE_ONLY"
        except Exception as error:
            report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            try:
                if policy is not None and hasattr(policy, "close"):
                    policy.close()
            except Exception as error:
                report["cleanup_error"] = f"{type(error).__name__}: {error}"
                if report.get("status") != "FAIL":
                    report["status"] = "PASS_INTERFACE_WITH_CLEANUP_ERROR"
            output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if not key.endswith("targets")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
