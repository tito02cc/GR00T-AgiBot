"""Candidate feedback controller for the official combined arm/tool API.

Reuse the legacy interpolation and bounded feedback law, but learn correction
from zero for this backend. The legacy module, defaults and sender stay intact.
This is a control candidate, not evidence that native hardware drift is fixed.
"""

from __future__ import annotations

import time
from typing import Any

from g2_groot_persistent_right_arm_controller import (
    CONTROL_PERIOD_S,
    LEFT_FRAME,
    RIGHT_FRAME,
    PersistentRightArmController,
    normalize,
    pose_values,
    require_safe_state,
    snapshot,
)
import numpy as np


class ContinuousRightArmController(PersistentRightArmController):
    """A single 50 Hz combined sender with the existing adaptive feedback law."""

    def __init__(
        self,
        robot: Any,
        tf: Any,
        *,
        sender: Any,
        workspace_min: Any,
        workspace_max: Any,
        required_motion_mode: int = 1,
        freeze_compensation_after_calibration: bool = False,
    ):
        if not callable(sender):
            raise ValueError("sender must be callable")
        super().__init__(
            robot,
            tf,
            initial_translation_compensation=[0.0, 0.0, 0.0],
            initial_rotation_compensation=[0.0, 0.0, 0.0, 1.0],
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            required_motion_mode=required_motion_mode,
        )
        self.command_sender = sender
        self.freeze_compensation_after_calibration = freeze_compensation_after_calibration
        self._calibration_in_progress = False
        self.last_tick: dict[str, Any] | None = None
        self.fault: str | None = None

    def _adapt(self, desired: np.ndarray, live: np.ndarray):
        # A moving target's tracking lag is not a static pose bias. The optional
        # calibration-only mode retains the measured bias without integrating
        # dynamic error into extra motion during the policy rollout or idle hold.
        if not self.freeze_compensation_after_calibration or self._calibration_in_progress:
            super()._adapt(desired, live)

    def tick(self, desired: np.ndarray):
        if self.fault is not None:
            raise RuntimeError(f"continuous controller fault latched: {self.fault}")
        try:
            target = np.asarray(desired, dtype=np.float64).copy()
            if target.shape != (7,) or not np.isfinite(target).all():
                raise ValueError("desired pose must be finite XYZ + XYZW")
            if np.linalg.norm(target[3:]) < 1e-12:
                raise ValueError("desired quaternion is zero")
            target[3:] = normalize(target[3:])
            left = pose_values(self.tf, LEFT_FRAME)
            right = pose_values(self.tf, RIGHT_FRAME)
            metrics = self._metrics(target, left, right)
            self._guard(target, metrics)
            require_safe_state(snapshot(self.robot, self.tf), self.required_motion_mode)
            if time.monotonic() - self.next_tick > CONTROL_PERIOD_S:
                raise RuntimeError("50 Hz control deadline missed before publication")
            self._adapt(target, right)
            command = self._command(target)
            result = self.command_sender(
                self.robot, command,
                latest_start_monotonic_ns=int((self.next_tick + CONTROL_PERIOD_S) * 1e9),
            )
            if result != 0:
                raise RuntimeError(f"combined sender returned {result!r}")
            self.last_tick = {
                "desired_pose": target.tolist(),
                "feedback_pose_before": right.tolist(),
                "published_pose": command.tolist(),
                "translation_compensation_m": self.translation_compensation.tolist(),
                "rotation_compensation_xyzw": self.rotation_compensation.tolist(),
                **metrics.as_dict(),
            }
            self.desired = target
            self.tick_count += 1
            for key, value in metrics.as_dict().items():
                self.maxima[key] = max(self.maxima.get(key, 0.0), value)
            self.next_tick += CONTROL_PERIOD_S
            wait = self.next_tick - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            elif wait < -CONTROL_PERIOD_S:
                raise RuntimeError(f"50 Hz control deadline missed by {-wait:.6f}s")
            return metrics
        except Exception as error:
            self.fault = f"{type(error).__name__}: {error}"
            self.calibrated = False
            raise

    def calibrate(self, duration_s: float = 2.0):
        # Reference initialization and SDK warmup happen before activation;
        # neither may leave the first publishing deadline in the past.
        if not np.isfinite(duration_s) or duration_s < CONTROL_PERIOD_S:
            raise ValueError("calibration duration must be at least one control period")
        self.next_tick = time.monotonic()
        self._calibration_in_progress = True
        try:
            result = super().calibrate(duration_s)
        except Exception as error:
            self.fault = f"{type(error).__name__}: {error}"
            self.calibrated = False
            raise
        finally:
            self._calibration_in_progress = False
        result["compensation_source"] = "online_feedback_from_zero_for_combined_api"
        result["compensation_mode"] = (
            "calibration_only" if self.freeze_compensation_after_calibration else "online_adaptive"
        )
        return result
