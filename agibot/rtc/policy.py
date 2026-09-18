"""Opt-in GR00T RTC policy adaptation; no robot, queue, or server side effects.

RTC inputs are an unexecuted absolute action tail aligned to the new observation.
Only B=1 and the validated task's effective H16/right-EEF+gripper schema are
supported. Tensor capacity 40 is not treated as 40 task-trained action steps.
The caller owns execution timing and must not replay the frozen prefix.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any

from gr00t.data.state_action.pose import EndEffectorPose
from gr00t.data.types import ActionFormat, ActionRepresentation, ActionType, MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
import numpy as np
import torch


RTC_SCHEMA = "g2_groot_rtc_policy_v1"
ACTION_DIMS = {"right_eef": 9, "right_gripper": 1}
EFFECTIVE_HORIZON = 16
ENCODING_TOLERANCES = {
    "eef_position_max_m": 1e-6,
    "eef_rotation_max_deg": 1e-3,
    "gripper_max_rad": 1e-6,
}


class RtcPrefixEncodingError(ValueError):
    """Prefix is not physically representable by the checkpoint's decoder."""

    def __init__(self, diagnostic: dict[str, Any]):
        self.diagnostic = diagnostic
        super().__init__("RTC prefix encoding is lossy: " + json.dumps(diagnostic, sort_keys=True))


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _pose(value: np.ndarray) -> EndEffectorPose:
    """Reject degenerate Rot6D before invoking the official pose conversion."""
    first, second = value[3:6], value[6:9]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        raise ValueError("RTC EEF has degenerate Rot6D first row")
    unit = first / first_norm
    if np.linalg.norm(second - np.dot(second, unit) * unit) < 1e-8:
        raise ValueError("RTC EEF has degenerate Rot6D second row")
    return EndEffectorPose.from_action_format(value, ActionFormat.XYZ_ROT6D)


def _errors(expected: dict[str, np.ndarray], actual: dict[str, np.ndarray]) -> dict[str, Any]:
    errors: dict[str, Any] = {
        "max_abs_by_key": {
            key: float(np.max(np.abs(actual[key].astype(np.float64) - value)))
            for key, value in expected.items()
        }
    }
    before = expected["right_eef"].reshape(-1, 9).astype(np.float64)
    after = actual["right_eef"].reshape(-1, 9).astype(np.float64)
    errors["eef_position_max_m"] = float(
        np.linalg.norm(after[:, :3] - before[:, :3], axis=-1).max()
    )
    rotations = []
    for first, second in zip(before, after):
        relative = _pose(first).rotation_matrix.T @ _pose(second).rotation_matrix
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        rotations.append(math.degrees(math.acos(float(cosine))))
    errors["eef_rotation_max_deg"] = max(rotations)
    errors["gripper_max_rad"] = errors["max_abs_by_key"]["right_gripper"]
    return errors


class RtcGr00tPolicy(Gr00tPolicy):
    """Use the official low-level RTC primitive only for explicit RTC requests.

    A private action processor disables input clipping, then verifies the official
    decoder can roundtrip the prefix. Its min/max decoder clips unconditionally;
    prefixes made unrepresentable by the new reference are explicitly rejected.
    Normal observation processing/checkpoint parameters remain unchanged. Model
    conditioning is still bf16; its error is separate from frozen-output repair.
    """

    def get_rtc_config(self) -> dict[str, Any]:
        action = self.modality_configs["action"]
        if action.delta_indices != list(range(EFFECTIVE_HORIZON)):
            raise ValueError("RTC adapter requires effective task horizon 16, not padded capacity")
        if (
            action.modality_keys != list(ACTION_DIMS)
            or action.action_configs is None
            or len(action.action_configs) != len(ACTION_DIMS)
        ):
            raise ValueError("RTC adapter requires right_eef followed by right_gripper")
        expected = (
            (ActionRepresentation.RELATIVE, ActionType.EEF, ActionFormat.XYZ_ROT6D, "right_eef"),
            (
                ActionRepresentation.ABSOLUTE,
                ActionType.NON_EEF,
                ActionFormat.DEFAULT,
                "right_gripper",
            ),
        )
        for key, config, contract in zip(action.modality_keys, action.action_configs, expected):
            state_key = config.state_key or key
            if (config.rep, config.type, config.format, state_key) != contract:
                raise ValueError(f"unsupported RTC action semantics for {key}")
        state_action = self.processor.state_action_processor
        if not state_action.use_relative_action:
            raise ValueError("RTC adapter requires the checkpoint's relative EEF processing")
        tag = self.embodiment_tag.value
        dims = {
            key: int(state_action.norm_params[tag]["action"][key]["dim"])
            for key in action.modality_keys
        }
        if dims != ACTION_DIMS:
            raise ValueError("RTC adapter requires physical action dimensions 9 + 1")
        if (
            self.processor.max_action_horizon < EFFECTIVE_HORIZON
            or self.processor.max_action_dim < 10
        ):
            raise ValueError("checkpoint action padding capacity is too small")
        return {
            "schema": RTC_SCHEMA,
            "enabled": True,
            "batch_size": 1,
            "effective_action_horizon": EFFECTIVE_HORIZON,
            "padded_action_horizon": int(self.processor.max_action_horizon),
            "padded_action_dim": int(self.processor.max_action_dim),
            "action_dims": dict(ACTION_DIMS),
            "maximum_overlap_steps": EFFECTIVE_HORIZON - 1,
            "previous_actions": "absolute_physical_float32_B_O_D_aligned_to_new_observation",
            "model_primitive": "official_gr00t_velocity_ramp",
            "prefix_encoding": "new_raw_state_relative_eef_absolute_gripper_no_prefix_clipping",
            "prefix_statistics_time_axis": "new_chunk_first_O_slots_of_effective_H16",
            "lossy_prefix_policy": "reject_with_diagnostic_before_model_call",
            "encoding_roundtrip_tolerances": dict(ENCODING_TOLERANCES),
            "conditioning_dtype": "bfloat16",
            "frozen_output": "restore_exact_previous_absolute_float32_with_error_report",
            "execution_queue_managed": False,
        }

    def _prefix_processor(self):
        if getattr(self, "_rtc_prefix_processor", None) is None:
            prefix_processor = deepcopy(self.processor.state_action_processor)
            prefix_processor.clip_outliers = False
            prefix_processor.eval()
            self._rtc_prefix_processor = prefix_processor
        return self._rtc_prefix_processor

    def _validate_rtc(self, request: Any) -> tuple[dict[str, np.ndarray], int, int, float]:
        if not isinstance(request, dict):
            raise ValueError("options['rtc'] must be a dict")
        if set(request) - {"previous_actions", "frozen_steps", "ramp_rate"}:
            raise ValueError("unknown RTC option")
        previous = request.get("previous_actions")
        if not isinstance(previous, dict) or set(previous) != set(ACTION_DIMS):
            raise ValueError("previous_actions must contain exactly right_eef and right_gripper")
        copied = {}
        overlap = None
        for key, dim in ACTION_DIMS.items():
            value = previous[key]
            if not isinstance(value, np.ndarray) or value.dtype != np.float32:
                raise ValueError(f"previous_actions[{key}] must be a float32 ndarray")
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != dim:
                raise ValueError(f"previous_actions[{key}] must have shape (1, O, {dim})")
            if not np.isfinite(value).all():
                raise ValueError(f"previous_actions[{key}] contains nonfinite values")
            if overlap is not None and value.shape[1] != overlap:
                raise ValueError("previous action keys must have the same overlap length")
            overlap = value.shape[1]
            copied[key] = value.copy()
        frozen = _integer(request.get("frozen_steps"), "frozen_steps")
        if overlap is None or not 0 < frozen <= overlap < EFFECTIVE_HORIZON:
            raise ValueError(
                "RTC requires 0 < frozen_steps <= overlap_steps < effective horizon 16"
            )
        ramp = request.get("ramp_rate", 3.0)
        if isinstance(ramp, bool) or not isinstance(ramp, (int, float, np.integer, np.floating)):
            raise ValueError("ramp_rate must be a finite positive number")
        ramp = float(ramp)
        if not math.isfinite(ramp) or ramp <= 0:
            raise ValueError("ramp_rate must be a finite positive number")
        for value in copied["right_eef"].reshape(-1, 9):
            _pose(value.astype(np.float64))
        return copied, overlap, frozen, ramp

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if options is None or "rtc" not in options:
            return super()._get_action(observation, options)
        capability = self.get_rtc_config()
        previous, overlap, frozen, ramp = self._validate_rtc(options["rtc"])
        observations = self._unbatch_observation(observation)
        if len(observations) != 1:
            raise ValueError("RTC adapter currently requires observation batch size 1")
        step = self._to_vla_step_data(observations[0])
        for key in self.modality_configs["state"].modality_keys:
            state = step.states[key]
            if (
                state.ndim != 2
                or state.shape
                != (len(self.modality_configs["state"].delta_indices), ACTION_DIMS[key])
                or not np.isfinite(state).all()
            ):
                raise ValueError(f"invalid raw RTC reference state for {key}")
        _pose(np.asarray(step.states["right_eef"][-1], dtype=np.float64))

        # Preserve the official normal-observation path (including its clipping).
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step}]
        processed = self.processor(messages)
        prefix_processor = self._prefix_processor()
        tag = self.embodiment_tag.value
        # Checkpoints may store (H16, D) per-timestep relative statistics. Encoding
        # an O-row tail directly cannot broadcast, and selecting the old chunk's
        # tail statistics would use the wrong new-time reference. Pad physical
        # values to H16 first, run the unchanged official processor, then take the
        # new chunk's first O slots. This temporary padding never goes to the model.
        full_previous = {
            key: np.concatenate(
                [value[0], np.repeat(value[0, -1:], EFFECTIVE_HORIZON - overlap, axis=0)],
                axis=0,
            )
            for key, value in previous.items()
        }
        normalized_full = prefix_processor.apply_action(full_previous, tag, state=step.states)
        encoded_roundtrip = prefix_processor.unapply_action(normalized_full, tag, state=step.states)
        float_errors = _errors(
            previous, {key: value[None, :overlap] for key, value in encoded_roundtrip.items()}
        )
        if any(float_errors[key] > tolerance for key, tolerance in ENCODING_TOLERANCES.items()):
            raise RtcPrefixEncodingError(
                {
                    "reason": "official_decoder_roundtrip_exceeds_tolerance",
                    "encoding_roundtrip": float_errors,
                    "tolerances": dict(ENCODING_TOLERANCES),
                    "prefix_clip_outliers": False,
                    "official_minmax_decoder_clips": True,
                }
            )
        concatenated = np.concatenate(
            [normalized_full[key][:overlap] for key in ACTION_DIMS], axis=-1
        )
        if not np.isfinite(concatenated).all():
            raise ValueError("RTC prefix normalization produced nonfinite values")
        # Same action order and zero-padding shape as the official VLA processor.
        padded = torch.zeros(
            (self.processor.max_action_horizon, self.processor.max_action_dim), dtype=torch.float32
        )
        padded[:overlap, :10] = torch.from_numpy(concatenated).to(torch.float32)
        mask = torch.zeros_like(padded)
        mask[:overlap, :10] = 1
        processed["action"], processed["action_mask"] = padded, mask
        collated = _rec_to_dtype(self.collate_fn([processed]), dtype=torch.bfloat16)
        conditioned = collated["inputs"]["action"]
        if not torch.isfinite(conditioned).all():
            raise ValueError("RTC prefix is nonfinite after bfloat16 quantization")
        conditioned_values = conditioned.float().cpu().numpy()[:, :overlap, :10]
        # Again decode H16 to respect per-timestep statistics. Only first O rows
        # are actual model conditioning; remaining rows are diagnostic filler.
        conditioned_actions = {key: value[None].copy() for key, value in normalized_full.items()}
        conditioned_actions["right_eef"][:, :overlap] = conditioned_values[..., :9]
        conditioned_actions["right_gripper"][:, :overlap] = conditioned_values[..., 9:10]
        batched_states = {key: value[None] for key, value in step.states.items()}
        conditioned_physical = prefix_processor.unapply_action(
            conditioned_actions, tag, state=batched_states
        )
        conditioning_errors = _errors(
            previous, {key: value[:, :overlap] for key, value in conditioned_physical.items()}
        )
        model_options = {
            "action_horizon": overlap,
            "rtc_overlap_steps": overlap,
            "rtc_frozen_steps": frozen,
            "rtc_ramp_rate": ramp,
        }
        with torch.inference_mode():
            prediction = self.model.get_action(**collated, options=model_options)
        normalized_action = prediction["action_pred"].float().cpu().numpy()
        decoded = self.processor.decode_action(
            normalized_action, self.embodiment_tag, batched_states
        )
        output = {key: value.astype(np.float32) for key, value in decoded.items()}
        for key, dim in ACTION_DIMS.items():
            if (
                output[key].shape != (1, EFFECTIVE_HORIZON, dim)
                or not np.isfinite(output[key]).all()
            ):
                raise ValueError(f"invalid RTC decoded action output for {key}")
        expected_frozen = {key: value[:, :frozen] for key, value in previous.items()}
        raw_frozen = {key: value[:, :frozen] for key, value in output.items()}
        restoration_errors = _errors(expected_frozen, raw_frozen)
        for key in ACTION_DIMS:
            output[key][:, :frozen] = previous[key][:, :frozen]
        return output, {
            "rtc": {
                "schema": RTC_SCHEMA,
                "enabled": True,
                "capability": capability,
                "overlap_steps": overlap,
                "frozen_steps": frozen,
                "ramp_rate": ramp,
                "model_options": model_options,
                "prefix_clip_outliers": False,
                "observation_clip_outliers": bool(
                    self.processor.state_action_processor.clip_outliers
                ),
                "normalized_prefix_outside_unit_interval": int(
                    np.count_nonzero(np.abs(concatenated) > 1)
                ),
                "encoding_roundtrip": float_errors,
                "conditioning_bfloat16_roundtrip": conditioning_errors,
                "frozen_return_correction": restoration_errors,
                "frozen_return_restored_exactly": True,
                "frozen_prefix_must_not_be_replayed": True,
            }
        }
