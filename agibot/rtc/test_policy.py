"""CPU-only RTC adapter tests using the official normalization/collation/decode.

Only image preprocessing and the neural network are replaced; no checkpoint,
CUDA, server, or robot is loaded.
"""

from copy import deepcopy
import unittest
from unittest.mock import patch

from agibot.configs.xichong_right_place_r0002_config import XICHONG_RIGHT_PLACE_R0002_CONFIG
from agibot.rtc.policy import ACTION_DIMS, RTC_SCHEMA, RtcGr00tPolicy, RtcPrefixEncodingError
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.state_action.pose import EndEffectorPose
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.types import ActionFormat, ActionRepresentation
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator, Gr00tN1d7Processor
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
from scipy.spatial.transform import Rotation
import torch


TAG = EmbodimentTag.NEW_EMBODIMENT


def stats(low, high):
    return {
        "min": list(low),
        "max": list(high),
        "q01": list(low),
        "q99": list(high),
        "mean": [0.0] * len(low),
        "std": [1.0] * len(low),
    }


def eef(xyz=(0.62, -0.18, 1.13), angles=(9.0, -6.0, 73.0)):
    matrix = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    return np.concatenate([xyz, matrix[:2].reshape(-1)]).astype(np.float32)


class FakeImageProcessor:
    """Keep official numeric processors; replace only VLM image/token work."""

    max_action_horizon = 40
    max_action_dim = 132
    decode_action = Gr00tN1d7Processor.decode_action

    def __init__(self, config):
        self.modality_configs = {TAG.value: config}
        eef_stats = stats([-2.0] * 9, [2.0] * 9)
        grip_stats = stats([-0.785], [0.0])
        relative_stats = stats([-0.06] * 3 + [-1.0] * 6, [0.06] * 3 + [1.0] * 6)
        self.state_action_processor = StateActionProcessor(
            self.modality_configs,
            statistics={
                TAG.value: {
                    "state": {"right_eef": eef_stats, "right_gripper": grip_stats},
                    "action": {"right_eef": eef_stats, "right_gripper": grip_stats},
                    "relative_action": {"right_eef": relative_stats},
                }
            },
            use_relative_action=True,
            use_percentiles=True,
            clip_outliers=True,
        )
        self.state_action_processor.eval()
        self.observation_calls = []

    def __call__(self, messages):
        step = messages[0]["content"]
        assert step.actions == {}, "Normal observation path must not receive RTC prefix actions"
        self.observation_calls.append(deepcopy(step.states))
        normalized = self.state_action_processor.apply_state(step.states, TAG.value)
        return {
            "state": np.concatenate(list(normalized.values()), axis=-1).astype(np.float32),
            "embodiment_id": np.array(0, dtype=np.int64),
        }


class FakeModel:
    """Echo the frozen conditioning; perturb unfrozen values to expose restoration scope."""

    def __init__(self):
        self.calls = []
        self.prediction = None

    def get_action(self, *, inputs, options):
        self.calls.append(({key: value.clone() for key, value in inputs.items()}, dict(options)))
        predicted = inputs["action"][:, :1].expand(-1, 40, -1).clone()
        overlap = options["rtc_overlap_steps"]
        frozen = options["rtc_frozen_steps"]
        predicted[:, :overlap] = inputs["action"][:, :overlap]
        predicted[:, frozen:, 0] += 0.25
        predicted[:, frozen:, 9] += 0.125
        self.prediction = predicted
        return {"action_pred": predicted}


def make_policy():
    policy = RtcGr00tPolicy.__new__(RtcGr00tPolicy)
    policy.strict = True
    policy.embodiment_tag = TAG
    policy.modality_configs = deepcopy(XICHONG_RIGHT_PLACE_R0002_CONFIG)
    policy.language_key = "annotation.human.task_description"
    policy.processor = FakeImageProcessor(policy.modality_configs)
    # Official collator has no tokenizer work for this numeric-only test input.
    policy.collate_fn = Gr00tN1d7DataCollator.__new__(Gr00tN1d7DataCollator)
    policy.model = FakeModel()
    return policy


def observation():
    return {
        "video": {
            "head_color": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8),
            "hand_right": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8),
        },
        "state": {
            "right_eef": eef()[None, None],
            "right_gripper": np.array([[[-0.1]]], dtype=np.float32),
        },
        "language": {"annotation.human.task_description": [["Place the workpiece"]]},
    }


def request(overlap=6, frozen=3):
    previous = {
        "right_eef": np.stack(
            [
                eef((0.654321 + i * 0.001357, -0.166789, 1.13713), (9.7, -5.2, 74.3))
                for i in range(overlap)
            ]
        )[None]
        if overlap
        else np.zeros((1, 0, 9), dtype=np.float32),
        "right_gripper": np.linspace(-0.71321, -0.23719, overlap, dtype=np.float32)[None, :, None],
    }
    return {"rtc": {"previous_actions": previous, "frozen_steps": frozen, "ramp_rate": 3.0}}


class RtcPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = make_policy()
        self.observation = observation()
        self.options = request()

    def run_rtc(self, options=None, obs=None):
        return self.policy.get_action(
            self.observation if obs is None else obs,
            self.options if options is None else options,
        )

    def test_no_rtc_is_exact_super_delegation(self):
        for options in (None, {}, {"unrelated": "untouched"}):
            with (
                self.subTest(options=options),
                patch.object(
                    Gr00tPolicy, "_get_action", return_value=("actions", "info")
                ) as original,
            ):
                result = self.policy._get_action(self.observation, options)
                self.assertEqual(result, ("actions", "info"))
                original.assert_called_once_with(self.observation, options)
        self.assertEqual(self.policy.model.calls, [])

    def test_capability_distinguishes_trained16_and_padded40(self):
        config = self.policy.get_rtc_config()
        self.assertEqual(config["schema"], RTC_SCHEMA)
        self.assertEqual(config["effective_action_horizon"], 16)
        self.assertEqual(config["padded_action_horizon"], 40)
        self.assertEqual(config["maximum_overlap_steps"], 15)
        self.assertFalse(config["execution_queue_managed"])
        config["action_dims"]["right_eef"] = 42
        self.assertEqual(self.policy.get_rtc_config()["action_dims"], ACTION_DIMS)

    def test_official_collation_bfloat16_padding_and_model_options(self):
        output, info = self.run_rtc()
        inputs, options = self.policy.model.calls[0]
        self.assertEqual(
            options,
            {
                "action_horizon": 6,
                "rtc_overlap_steps": 6,
                "rtc_frozen_steps": 3,
                "rtc_ramp_rate": 3.0,
            },
        )
        self.assertEqual(inputs["action"].shape, (1, 40, 132))
        self.assertEqual(inputs["action"].dtype, torch.bfloat16)
        self.assertEqual(inputs["embodiment_id"].dtype, torch.int64)
        self.assertEqual(int(inputs["action_mask"].sum()), 60)
        self.assertTrue(torch.all(inputs["action_mask"][:, :6, :10] == 1))
        self.assertEqual(int(torch.count_nonzero(inputs["action"][:, 6:])), 0)
        self.assertEqual(int(torch.count_nonzero(inputs["action"][:, :, 10:])), 0)
        self.assertEqual(output["right_eef"].shape, (1, 16, 9))
        self.assertEqual(output["right_gripper"].shape, (1, 16, 1))
        self.assertEqual(info["rtc"]["model_options"], options)

    def test_new_reference_is_full_se3_not_xyz_subtraction(self):
        self.run_rtc()
        normalized = self.policy.model.calls[0][0]["action"].float().numpy()[0, 0, :9]
        prior = self.options["rtc"]["previous_actions"]["right_eef"][0, 0]
        reference = self.observation["state"]["right_eef"][0, -1]
        old_pose = EndEffectorPose.from_action_format(prior, ActionFormat.XYZ_ROT6D)
        ref_pose = EndEffectorPose.from_action_format(reference, ActionFormat.XYZ_ROT6D)
        expected_xyz = ref_pose.rotation_matrix.T @ (old_pose.translation - ref_pose.translation)
        np.testing.assert_allclose(normalized[:3] * 0.06, expected_xyz, atol=0.00015)
        self.assertGreater(np.linalg.norm(expected_xyz - (prior[:3] - reference[:3])), 0.02)

    def test_gripper_is_absolute_under_changed_new_gripper_state(self):
        self.run_rtc()
        encoded_first = self.policy.model.calls[-1][0]["action"][..., 9].clone()
        changed = deepcopy(self.observation)
        changed["state"]["right_gripper"][:] = -0.65
        self.run_rtc(obs=changed)
        torch.testing.assert_close(self.policy.model.calls[-1][0]["action"][..., 9], encoded_first)

    def test_h16_per_timestep_statistics_use_new_prefix_slots_not_old_tail_slots(self):
        processor = self.policy.processor.state_action_processor
        statistics = deepcopy(processor.statistics)
        relative = statistics[TAG.value]["relative_action"]["right_eef"]
        for name, values in relative.items():
            repeated = np.repeat(np.array(values)[None], 16, axis=0)
            if name in ("min", "max", "q01", "q99"):
                repeated[:, :3] += np.arange(16)[:, None] * 0.0009
            relative[name] = repeated.tolist()
        processor.set_statistics(statistics, override=True)
        self.assertEqual(
            processor.norm_params[TAG.value]["action"]["right_eef"]["min"].shape, (16, 9)
        )
        output, info = self.run_rtc()
        inputs = self.policy.model.calls[-1][0]
        previous = self.options["rtc"]["previous_actions"]
        physical_full = {
            key: np.concatenate([value[0], np.repeat(value[0, -1:], 10, axis=0)])
            for key, value in previous.items()
        }
        states = {key: value[0] for key, value in self.observation["state"].items()}
        normalized = self.policy._prefix_processor().apply_action(physical_full, TAG.value, states)
        expected = torch.from_numpy(np.concatenate(list(normalized.values()), axis=-1)[:6]).to(
            torch.bfloat16
        )
        torch.testing.assert_close(inputs["action"][0, :6, :10], expected)
        self.assertFalse(
            torch.equal(
                inputs["action"][0, :6, :9],
                torch.from_numpy(normalized["right_eef"][-6:]).to(torch.bfloat16),
            )
        )
        self.assertLess(info["rtc"]["encoding_roundtrip"]["eef_position_max_m"], 1e-6)
        self.assertEqual(output["right_eef"].shape, (1, 16, 9))
        for key in ACTION_DIMS:
            np.testing.assert_array_equal(output[key][:, :3], previous[key][:, :3])

    def test_prefix_clipping_disabled_only_on_private_processor(self):
        original = self.policy.processor.state_action_processor
        original_statistics = deepcopy(original.statistics)
        _, info = self.run_rtc()
        self.assertTrue(original.clip_outliers)
        self.assertFalse(self.policy._prefix_processor().clip_outliers)
        self.assertIsNot(original, self.policy._prefix_processor())
        self.assertEqual(original.statistics, original_statistics)
        self.assertEqual(len(self.policy.processor.observation_calls), 1)
        self.assertTrue(info["rtc"]["observation_clip_outliers"])
        self.assertEqual(info["rtc"]["normalized_prefix_outside_unit_interval"], 0)
        self.assertLess(info["rtc"]["encoding_roundtrip"]["eef_position_max_m"], 1e-6)

    def test_out_of_range_prefix_rejected_with_actual_decoder_loss_diagnostic(self):
        original = self.policy.processor.state_action_processor
        self.options["rtc"]["previous_actions"]["right_eef"][..., 0] += 0.2
        state = {key: value[0] for key, value in self.observation["state"].items()}
        previous = {key: value[0] for key, value in self.options["rtc"]["previous_actions"].items()}
        lossy = original.unapply_action(
            original.apply_action(previous, TAG.value, state), TAG.value, state
        )
        self.assertGreater(
            np.max(np.abs(lossy["right_eef"][:, :3] - previous["right_eef"][:, :3])), 0.005
        )
        with self.assertRaises(RtcPrefixEncodingError) as caught:
            self.run_rtc()
        self.assertGreater(
            caught.exception.diagnostic["encoding_roundtrip"]["eef_position_max_m"], 0.1
        )
        self.assertIn("official_decoder_roundtrip", str(caught.exception))
        self.assertEqual(self.policy.model.calls, [])
        self.assertTrue(original.clip_outliers)

    def test_bfloat16_conditioning_loss_reported_despite_exact_frozen_return(self):
        output, info = self.run_rtc()
        rtc = info["rtc"]
        self.assertGreater(rtc["conditioning_bfloat16_roundtrip"]["eef_position_max_m"], 1e-6)
        self.assertGreater(rtc["conditioning_bfloat16_roundtrip"]["gripper_max_rad"], 1e-6)
        self.assertGreater(rtc["frozen_return_correction"]["eef_position_max_m"], 1e-6)
        self.assertTrue(rtc["frozen_return_restored_exactly"])
        self.assertTrue(rtc["frozen_prefix_must_not_be_replayed"])
        for key, previous in self.options["rtc"]["previous_actions"].items():
            np.testing.assert_array_equal(output[key][:, :3], previous[:, :3])

    def test_unfrozen_overlap_and_new_tail_remain_model_outputs(self):
        output, _ = self.run_rtc()
        raw_output = self.policy.processor.decode_action(
            self.policy.model.prediction.float().numpy(), TAG, self.observation["state"]
        )
        for key in ACTION_DIMS:
            np.testing.assert_array_equal(
                output[key][:, 3:], raw_output[key][:, 3:].astype(np.float32)
            )
        self.assertFalse(
            np.array_equal(
                output["right_gripper"][:, 3:6],
                self.options["rtc"]["previous_actions"]["right_gripper"][:, 3:6],
            )
        )

    def test_inputs_not_mutated(self):
        original_options, original_obs = deepcopy(self.options), deepcopy(self.observation)
        self.run_rtc()
        for key in ACTION_DIMS:
            np.testing.assert_array_equal(
                original_options["rtc"]["previous_actions"][key],
                self.options["rtc"]["previous_actions"][key],
            )
            np.testing.assert_array_equal(
                original_obs["state"][key], self.observation["state"][key]
            )
        for key in original_obs["video"]:
            np.testing.assert_array_equal(
                original_obs["video"][key], self.observation["video"][key]
            )

    def test_overlap_and_frozen_boundaries(self):
        for overlap, frozen in (
            (0, 0),
            (16, 3),
            (40, 3),
            (6, 0),
            (6, 7),
            (6, -1),
            (6, True),
            (6, 2.5),
        ):
            with self.subTest(overlap=overlap, frozen=frozen), self.assertRaises(ValueError):
                self.run_rtc(request(overlap, frozen))
        self.assertEqual(self.policy.model.calls, [])
        for overlap, frozen in ((1, 1), (15, 15), (15, 1)):
            output, _ = self.run_rtc(request(overlap, frozen))
            self.assertEqual(output["right_eef"].shape[1], 16)

    def test_bad_ramp_values_and_unknown_options(self):
        for ramp in (0, -1, float("nan"), float("inf"), True, "3", 1j):
            invalid = deepcopy(self.options)
            invalid["rtc"]["ramp_rate"] = ramp
            with self.subTest(ramp=ramp), self.assertRaises(ValueError):
                self.run_rtc(invalid)
        for rtc in (None, {}, {**self.options["rtc"], "unknown": 1}):
            with self.subTest(rtc=rtc), self.assertRaises(ValueError):
                self.run_rtc({"rtc": rtc})
        self.assertEqual(self.policy.model.calls, [])

    def test_previous_arrays_require_float32_finite_b1_complete_consistent_shapes(self):
        bad_values = (
            np.zeros((2, 6, 9), np.float32),
            np.zeros((1, 6, 8), np.float32),
            np.zeros((6, 9), np.float32),
            np.zeros((1, 6, 9), np.float64),
            np.full((1, 6, 9), np.nan, np.float32),
            np.full((1, 6, 9), np.inf, np.float32),
            [],
        )
        for value in bad_values:
            invalid = deepcopy(self.options)
            invalid["rtc"]["previous_actions"]["right_eef"] = value
            with self.subTest(shape=getattr(value, "shape", None)), self.assertRaises(ValueError):
                self.run_rtc(invalid)
        invalid = deepcopy(self.options)
        invalid["rtc"]["previous_actions"]["right_gripper"] = np.zeros((1, 5, 1), np.float32)
        with self.assertRaises(ValueError):
            self.run_rtc(invalid)
        invalid["rtc"]["previous_actions"].pop("right_gripper")
        with self.assertRaises(ValueError):
            self.run_rtc(invalid)
        self.assertEqual(self.policy.model.calls, [])

    def test_degenerate_previous_rotation_rejected_before_model(self):
        for first, second in (([0, 0, 0], [0, 1, 0]), ([1, 0, 0], [2, 0, 0])):
            invalid = deepcopy(self.options)
            invalid["rtc"]["previous_actions"]["right_eef"][0, 0, 3:] = first + second
            with self.assertRaisesRegex(ValueError, "degenerate"):
                self.run_rtc(invalid)
        self.assertEqual(self.policy.model.calls, [])

    def test_raw_state_nonfinite_rejected_even_without_strict_checks(self):
        self.policy.strict = False
        self.observation["state"]["right_eef"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "raw RTC reference"):
            self.run_rtc()
        self.assertEqual(self.policy.model.calls, [])

    def test_observation_batch_two_rejected(self):
        self.policy.strict = False
        for group in ("video", "state"):
            self.observation[group] = {
                key: np.repeat(value, 2, axis=0) for key, value in self.observation[group].items()
            }
        self.observation["language"][self.policy.language_key] *= 2
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            self.run_rtc()

    def test_capability_rejects_padding_as_trained_horizon_or_wrong_semantics(self):
        self.policy.modality_configs["action"].delta_indices = list(range(40))
        with self.assertRaisesRegex(ValueError, "effective task horizon 16"):
            self.policy.get_rtc_config()
        self.policy.modality_configs["action"].delta_indices = list(range(16))
        self.policy.modality_configs["action"].action_configs[1].rep = ActionRepresentation.RELATIVE
        with self.assertRaisesRegex(ValueError, "right_gripper"):
            self.policy.get_rtc_config()

    def test_model_invalid_decode_is_rejected_not_returned(self):
        original = self.policy.model.get_action

        def invalid_model(**kwargs):
            result = original(**kwargs)
            result["action_pred"][0, 10, 9] = float("nan")
            return result

        self.policy.model.get_action = invalid_model
        with self.assertRaisesRegex(ValueError, "invalid RTC decoded action output"):
            self.run_rtc()


if __name__ == "__main__":
    unittest.main()
