"""Latch observed native GDK collisions/mode changes; never auto-resume commands.

This is a publication interlock, not a hardware e-stop or force-limit detector.
The SDK provides no source timestamp. Startup config is read back; other clients
must not change it during execution. Firmware collision sensitivity/calibration
and actual physical stopping still require onsite validation.
"""

from __future__ import annotations

import threading


PROTOCOL = "g2_native_collision_latch_v1"
MIN_RECOVERY_MS = 500


class NativeCollisionLatch:
    def __init__(self, required_control_mode=3):
        if required_control_mode not in (2, 3):
            raise ValueError("native collision latch requires impedance control mode 2 or 3")
        self.required_control_mode = required_control_mode
        self._lock = threading.RLock()
        self.armed = False
        self.fault = None
        self.configuration = None
        self.last_status = None

    def snapshot(self):
        with self._lock:
            return {
                "protocol": PROTOCOL, "required_control_mode": self.required_control_mode,
                "minimum_recovery_ms": MIN_RECOVERY_MS, "armed": self.armed,
                "fault": self.fault,
                "configuration_at_arm": None if self.configuration is None else dict(self.configuration),
                "last_status": None if self.last_status is None else dict(self.last_status),
                "source_timestamp_available": False,
                "stop_semantics": "latch_fault_discard_queue_no_further_sdk_publications",
            }

    def _healthy(self):
        if self.fault is not None:
            raise RuntimeError(f"native collision fault latched: {self.fault}")

    def _trip(self, message):
        with self._lock:
            self.fault = self.fault or message
            raise RuntimeError(self.fault)

    def arm(self, robot):
        with self._lock:
            self._healthy()
            try:
                config = robot.get_collision_detection_config()
                self.configuration = {
                    "is_enabled": config.is_enabled,
                    "sensitivity": int(config.sensitivity),
                    "checkout_timeout_ms": int(config.checkout_timeout_ms),
                }
                if config.is_enabled is not True or not 1 <= config.sensitivity <= 3:
                    raise ValueError("native collision protection is not enabled/valid")
                if config.checkout_timeout_ms < MIN_RECOVERY_MS:
                    raise ValueError(f"native collision recovery must be at least {MIN_RECOVERY_MS} ms")
                self._check_status(robot.get_motion_control_status())
                self.armed = True
            except Exception as error:
                self._trip(f"native collision arming rejected: {error}")

    def _check_status(self, status):
        mode, control_mode, error_code = int(status.mode), int(status.control_mode), int(status.error_code)
        collision = bool(list(status.collision_pairs_1) or list(status.collision_pairs_2))
        self.last_status = {
            "mode": mode, "control_mode": control_mode,
            "error_code": error_code, "collision_reported": collision,
        }
        if collision or error_code or mode != 1 or control_mode != self.required_control_mode:
            raise RuntimeError(f"native collision/mode/error interlock: {self.last_status}")

    def check_robot(self, robot):
        with self._lock:
            self._healthy()
            if not self.armed:
                self._trip("native collision latch is not armed")
            try:
                self._check_status(robot.get_motion_control_status())
            except Exception as error:
                self._trip(f"native collision check failed: {error}")
