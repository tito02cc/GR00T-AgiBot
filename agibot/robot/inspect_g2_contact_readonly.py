#!/usr/bin/env python3
"""Read GDK contact telemetry only. Does not calibrate or choose safe thresholds."""

import argparse
import json
import math
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.samples <= 250:
        parser.error("samples must be in [1, 250]")
    import agibot_gdk

    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK init failed")
    try:
        robot = agibot_gdk.Robot()
        time.sleep(2)
        config = robot.get_collision_detection_config()
        samples = []
        for _ in range(args.samples):
            started = time.monotonic()
            state = robot.get_motion_control_status()
            read_s = time.monotonic() - started
            names = list(state.frame_names)
            if names.count("arm_r_end_link") != 1 or len(names) != len(state.wrenches):
                raise RuntimeError("right EEF wrench frame missing or ambiguous")
            wrench = state.wrenches[names.index("arm_r_end_link")]
            force = [float(getattr(wrench.force, axis)) for axis in "xyz"]
            torque = [float(getattr(wrench.torque, axis)) for axis in "xyz"]
            if not all(math.isfinite(x) for x in force + torque):
                raise RuntimeError("non-finite wrench feedback")
            samples.append({
                "force_norm_n": math.hypot(*force), "torque_norm_nm": math.hypot(*torque),
                "read_s": read_s, "mode": int(state.mode), "control_mode": int(state.control_mode),
                "error_code": int(state.error_code),
                "collision_pairs_1": list(state.collision_pairs_1),
                "collision_pairs_2": list(state.collision_pairs_2),
            })
            time.sleep(0.02)
        print(json.dumps({
            "result": "READ_ONLY_NOT_CALIBRATION", "eef_frame": "arm_r_end_link",
            "source_timestamp_available": False,
            "collision_detection_config": {
                key: getattr(config, key) for key in ("is_enabled", "sensitivity", "checkout_timeout_ms")
            },
            "summary": {
                key: {"min": min(row[key] for row in samples), "max": max(row[key] for row in samples)}
                for key in ("force_norm_n", "torque_norm_nm", "read_s")
            },
            "samples": samples,
        }, ensure_ascii=False), flush=True)
    finally:
        agibot_gdk.gdk_release()


if __name__ == "__main__":
    main()
