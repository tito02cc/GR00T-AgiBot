"""Explicit acknowledgement of this opt-in RTC protocol, without torch imports."""

import math


RTC_SCHEMA = "g2_groot_rtc_policy_v1"


def validate_capability(value: dict) -> None:
    if (
        not isinstance(value, dict)
        or value.get("schema") != RTC_SCHEMA
        or value.get("enabled") is not True
        or value.get("effective_action_horizon") != 16
    ):
        raise RuntimeError("server does not advertise the required H16 RTC interface")


def validate_receipt(info: dict, *, overlap: int, frozen: int, ramp_rate: float) -> None:
    receipt = info.get("rtc") if isinstance(info, dict) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != RTC_SCHEMA
        or receipt.get("enabled") is not True
        or receipt.get("frozen_return_restored_exactly") is not True
    ):
        raise RuntimeError("model did not acknowledge RTC; no silent baseline fallback")
    for key, expected in (("overlap_steps", overlap), ("frozen_steps", frozen)):
        if type(receipt.get(key)) is not int or receipt[key] != expected:
            raise RuntimeError(f"RTC acknowledgement has mismatched {key}")
    actual_ramp = receipt.get("ramp_rate")
    if (
        isinstance(actual_ramp, bool)
        or not isinstance(actual_ramp, (int, float))
        or not math.isfinite(actual_ramp)
        or actual_ramp != ramp_rate
    ):
        raise RuntimeError("RTC acknowledgement has mismatched ramp_rate")
