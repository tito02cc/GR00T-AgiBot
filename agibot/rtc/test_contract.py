"""CPU-only tests of explicit RTC capabilities and per-request receipts."""

from copy import deepcopy
import unittest

from agibot.rtc.contract import RTC_SCHEMA, validate_capability, validate_receipt


def capability():
    return {"schema": RTC_SCHEMA, "enabled": True, "effective_action_horizon": 16}


def acknowledgement():
    return {
        "rtc": {
            "schema": RTC_SCHEMA,
            "enabled": True,
            "frozen_return_restored_exactly": True,
            "overlap_steps": 8,
            "frozen_steps": 4,
            "ramp_rate": 3.0,
        }
    }


class ContractTests(unittest.TestCase):
    def validate(self, info):
        validate_receipt(info, overlap=8, frozen=4, ramp_rate=3.0)

    def test_accepts_explicit_h16_capability_without_mutation(self):
        value = capability()
        original = deepcopy(value)
        self.assertIsNone(validate_capability(value))
        self.assertEqual(value, original)

    def test_rejects_missing_or_non_dict_capability(self):
        for value in (None, [], "RTC", {}, {"enabled": True}):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                validate_capability(value)

    def test_rejects_wrong_capability_fields_and_non_boolean_enabled(self):
        for key, invalid in (
            ("schema", "g2_groot_rtc_policy_v0"),
            ("schema", None),
            ("enabled", False),
            ("enabled", 1),
            ("enabled", "true"),
            ("effective_action_horizon", 40),
            ("effective_action_horizon", 15),
            ("effective_action_horizon", "16"),
        ):
            value = capability()
            value[key] = invalid
            with self.subTest(key=key, invalid=invalid), self.assertRaises(RuntimeError):
                validate_capability(value)

    def test_accepts_exact_request_acknowledgement_without_mutation(self):
        value = acknowledgement()
        original = deepcopy(value)
        self.assertIsNone(self.validate(value))
        self.assertEqual(value, original)
        value["rtc"]["ramp_rate"] = 3
        self.assertIsNone(self.validate(value))

    def test_rejects_absent_empty_and_non_dict_receipt(self):
        for value in (None, [], "RTC", {}, {"rtc": None}, {"rtc": []}, {"rtc": {}}):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.validate(value)

    def test_each_receipt_field_is_required(self):
        for key in acknowledgement()["rtc"]:
            value = acknowledgement()
            del value["rtc"][key]
            with self.subTest(missing=key), self.assertRaises(RuntimeError):
                self.validate(value)

    def test_rejects_disabled_wrong_schema_or_unrestored_receipt(self):
        for key, invalid in (
            ("schema", "another_rtc_interface"),
            ("enabled", False),
            ("enabled", 1),
            ("enabled", "true"),
            ("frozen_return_restored_exactly", False),
            ("frozen_return_restored_exactly", 1),
        ):
            value = acknowledgement()
            value["rtc"][key] = invalid
            with self.subTest(key=key, invalid=invalid), self.assertRaises(RuntimeError):
                self.validate(value)

    def test_rejects_mismatched_or_non_integer_index_counts(self):
        for key, invalid in (
            ("overlap_steps", 7),
            ("overlap_steps", 16),
            ("overlap_steps", 8.0),
            ("overlap_steps", "8"),
            ("overlap_steps", True),
            ("frozen_steps", 3),
            ("frozen_steps", 4.0),
            ("frozen_steps", "4"),
            ("frozen_steps", True),
        ):
            value = acknowledgement()
            value["rtc"][key] = invalid
            with self.subTest(key=key, invalid=invalid), self.assertRaises(RuntimeError):
                self.validate(value)

    def test_rejects_nonfinite_wrong_typed_or_mismatched_ramp(self):
        for invalid in (True, None, "3", float("inf"), float("nan"), -3, 0, 3.00001):
            value = acknowledgement()
            value["rtc"]["ramp_rate"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                self.validate(value)


if __name__ == "__main__":
    unittest.main()
