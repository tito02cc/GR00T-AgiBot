"""Right-EEF publication interlock, NOT a certified collision/force controller.

All limits require site confirmation. Wrench norms use the installed GDK's
frame_names/wrenches pairing; never assume an arm index or auto-zero a payload.
The binding exposes no source timestamp: successful reads do not prove freshness.
A trip only prevents subsequent SDK publications; it cannot cancel an in-flight
point or replace the physical emergency stop. No GDK setters are used here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import threading
from typing import Any


SCHEMA = "g2_right_contact_guard_v1"
LIMIT_KEYS = ("eef_min_z_m", "max_force_norm_n", "max_torque_norm_nm")
CONFIG_KEYS = {
    "schema", "robot_ip", "base_frame", "eef_frame", "confirmed",
    "calibration_note", *LIMIT_KEYS,
}


class ContactGuardTrip(RuntimeError):
    pass


class ContactGuard:
    def __init__(self, config: dict):
        if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
            raise ValueError("contact guard config has missing or unknown fields")
        if config["schema"] != SCHEMA:
            raise ValueError("unsupported contact guard schema")
        if config["base_frame"] != "base_link" or config["eef_frame"] != "arm_r_end_link":
            raise ValueError("contact guard requires base_link / arm_r_end_link")
        if not isinstance(config["robot_ip"], str) or not config["robot_ip"].strip():
            raise ValueError("robot_ip must identify the calibrated robot")
        if type(config["confirmed"]) is not bool:
            raise ValueError("confirmed must be a boolean")
        if not isinstance(config["calibration_note"], str):
            raise ValueError("calibration_note must be a string")
        for key in LIMIT_KEYS:
            value = config[key]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a finite number or null")
            if not math.isfinite(value) or (key != "eef_min_z_m" and value <= 0):
                raise ValueError(f"invalid {key}")
        self._config = dict(config)
        self._lock = threading.RLock()
        self._fault: str | None = None
        self._last_feedback: dict | None = None

    @classmethod
    def from_file(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text()))

    def snapshot(self) -> dict:
        with self._lock:
            missing = [key for key in LIMIT_KEYS if self._config[key] is None]
            configured = (
                self._config["confirmed"] and not missing
                and bool(self._config["calibration_note"].strip())
            )
            return {
                "schema": SCHEMA,
                "configured": bool(configured),
                "config": dict(self._config),
                "missing_limits": missing,
                "fault": self._fault,
                "last_feedback": None if self._last_feedback is None else dict(self._last_feedback),
                "source_timestamp_available": False,
                "stop_semantics": "reject_further_sdk_publications_no_auto_release_or_reset",
            }

    def ensure_ready(self):
        with self._lock:
            if self._fault is not None:
                raise ContactGuardTrip(f"contact guard fault latched: {self._fault}")
            if not self.snapshot()["configured"]:
                raise ValueError("contact guard limits require site confirmation; not activating")

    def trip(self, message: str):
        with self._lock:
            self._fault = self._fault or message
            raise ContactGuardTrip(self._fault)

    def check_pose(self, pose, *, source: str = "sdk_command"):
        self.ensure_ready()
        try:
            values = list(pose)
            if len(values) != 7 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError("expected finite XYZ+XYZW")
            z = float(values[2])
        except (ValueError, TypeError, OverflowError) as error:
            self.trip(f"{source}: invalid pose: {error}")
        if z < self._config["eef_min_z_m"]:
            self.trip(f"{source}: Z={z:.6f} below confirmed EEF floor {self._config['eef_min_z_m']:.6f}")

    def check_feedback(self, motion: Any, live_pose):
        self.check_pose(live_pose, source="live_feedback")
        try:
            error = int(motion.error_code)
            pairs1, pairs2 = list(motion.collision_pairs_1), list(motion.collision_pairs_2)
            names, wrenches = list(motion.frame_names), list(motion.wrenches)
            if len(names) != len(wrenches) or names.count(self._config["eef_frame"]) != 1:
                raise ValueError("ambiguous or missing right-EEF wrench frame")
            wrench = wrenches[names.index(self._config["eef_frame"])]
            force = [float(getattr(wrench.force, axis)) for axis in "xyz"]
            torque = [float(getattr(wrench.torque, axis)) for axis in "xyz"]
            if not all(math.isfinite(value) for value in force + torque):
                raise ValueError("non-finite right-EEF wrench")
            force_norm, torque_norm = math.hypot(*force), math.hypot(*torque)
        except (AttributeError, ValueError, TypeError, OverflowError) as error:
            self.trip(f"invalid GDK contact feedback: {error}")
        with self._lock:
            self._last_feedback = {
                "frame": self._config["eef_frame"], "force_norm_n": force_norm,
                "torque_norm_nm": torque_norm, "motion_error": error,
                "collision_pair_count": max(len(pairs1), len(pairs2)),
            }
        if error:
            self.trip(f"GDK motion error: {error}")
        if pairs1 or pairs2:
            self.trip("GDK reports collision pairs")
        if force_norm > self._config["max_force_norm_n"]:
            self.trip(f"right-EEF force norm {force_norm:.6f} N exceeds confirmed limit")
        if torque_norm > self._config["max_torque_norm_nm"]:
            self.trip(f"right-EEF torque norm {torque_norm:.6f} Nm exceeds confirmed limit")
