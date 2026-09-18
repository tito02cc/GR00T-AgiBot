"""CPU-only request-mapping tests; this does not commission the GDK backend."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace
import unittest

from g2_groot_continuous_sender import (
    CLOSED_GRIPPER,
    CONTROL_PERIOD_S,
    GRIPPER_FLOAT_TOLERANCE,
    MODEL_TICKS,
    OPEN_GRIPPER,
    REFERENCE_FRAMES,
    NativeTrajectorySender,
)


POSE = [0.5, -0.18, 1.05, 0.0, 0.0, 0.0, 1.0]
LEFT_REFERENCE = [0.0, 0.6, 0.0, 0.8]
RIGHT_REFERENCE = [0.6, 0.0, 0.8, 0.0]


class FakeTF:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.references = dict(zip(REFERENCE_FRAMES, [LEFT_REFERENCE, RIGHT_REFERENCE]))

    def get_tf_from_base_link(self, frame):
        self.events.append(("tf", frame))
        value = self.references[frame]
        if isinstance(value, Exception):
            raise value
        if value is None:
            return None
        return SimpleNamespace(rotation=SimpleNamespace(**dict(zip("xyzw", value))))


class FakeRobot:
    def __init__(
        self, result=0, error=None, *, reference_result=0, reference_error=None, events=None
    ):
        self.result = result
        self.error = error
        self.calls = []
        self.reference_result = reference_result
        self.reference_error = reference_error
        self.reference_calls = []
        self.events = events if events is not None else []

    def set_reference_frame_poses(self, *args):
        self.reference_calls.append(args)
        self.events.append(("set_references", args))
        if self.reference_error is not None:
            raise self.reference_error
        return self.reference_result

    def trajectory_tracking_control(self, *args, **kwargs):
        self.events.append(("publish",))
        self.calls.append(copy.deepcopy((args, kwargs)))
        if self.error is not None:
            raise self.error
        return self.result


def gripper_values(robot):
    return [args[2][0]["right_effector"]["action_data"][0] for args, _ in robot.calls]


class NativeTrajectorySenderTest(unittest.TestCase):
    def test_compensated_sdk_pose_guard_rejects_before_publication(self):
        from g2_groot_contact_guard import ContactGuard
        from test_g2_groot_contact_guard import synthetic_config

        guard = ContactGuard(synthetic_config())
        robot = FakeRobot()
        sender = NativeTrajectorySender(0.0, pose_guard=guard.check_pose)
        sender.initialize_references(robot, FakeTF())
        sender.set_target(-0.5)
        unsafe_pose = POSE.copy()
        unsafe_pose[2] = 0.99
        with self.assertRaisesRegex(RuntimeError, "below confirmed EEF floor"):
            sender(robot, unsafe_pose)
        self.assertEqual(robot.calls, [])
        self.assertEqual(sender.snapshot()["remaining_ticks"], 5)
        with self.assertRaisesRegex(RuntimeError, "fault latched"):
            sender(robot, POSE)

    def test_slow_feedback_callback_prevents_expired_publication(self):
        now = [0]

        def slow_feedback():
            now[0] += 30_000_000

        robot = FakeRobot()
        sender = NativeTrajectorySender(0.0, before_publish=slow_feedback, clock_ns=lambda: now[0])
        sender.initialize_references(robot, FakeTF())
        sender.set_target(-0.5)
        with self.assertRaisesRegex(RuntimeError, "deadline missed after feedback"):
            sender(robot, POSE, latest_start_monotonic_ns=20_000_000)
        self.assertEqual(robot.calls, [])
        self.assertEqual(sender.snapshot()["call_count"], 0)
        self.assertEqual(sender.snapshot()["remaining_ticks"], 5)
        with self.assertRaisesRegex(RuntimeError, "fault latched"):
            sender(robot, POSE)

    def test_construction_and_set_target_do_not_call_robot(self):
        robot = FakeRobot()
        sender = NativeTrajectorySender(-0.3)
        sender.set_target(-0.2)
        self.assertEqual(robot.calls, [])
        self.assertEqual(sender.snapshot()["call_count"], 0)
        self.assertIsNone(sender.snapshot()["sent_target"])
        self.assertIsNone(sender.snapshot()["accepted_target"])

    def test_publication_requires_explicit_reference_initialization(self):
        sender = NativeTrajectorySender(0.0)
        robot = FakeRobot()
        sender.set_target(-0.4)
        before = sender.snapshot()
        with self.assertRaisesRegex(RuntimeError, "initialize_references"):
            sender(robot, POSE)
        self.assertEqual(sender.snapshot(), before)
        self.assertEqual(robot.calls, [])
        self.assertEqual(robot.reference_calls, [])

    def test_initialization_reads_both_link3_frames_before_local_setter_and_publication(self):
        events = []
        robot, tf = FakeRobot(events=events), FakeTF(events=events)
        sender = NativeTrajectorySender(0.0)
        sender.set_target(-0.3)
        references = sender.initialize_references(robot, tf)
        self.assertEqual(
            events,
            [
                ("tf", "arm_l_link3"),
                ("tf", "arm_r_link3"),
                ("set_references", tuple(LEFT_REFERENCE + RIGHT_REFERENCE)),
            ],
        )
        self.assertEqual(robot.calls, [])
        self.assertTrue(references["initialized"])
        self.assertEqual(references["base_frame"], "base_link")
        self.assertEqual(references["setter_result"], 0)
        self.assertEqual(
            references["reference_quaternions_xyzw"],
            {"arm_l_link3": LEFT_REFERENCE, "arm_r_link3": RIGHT_REFERENCE},
        )
        sender(robot, POSE)
        self.assertEqual(events[-1], ("publish",))
        self.assertAlmostEqual(gripper_values(robot)[0], -0.06)

    def test_invalid_reference_never_reaches_setter_or_publication(self):
        for frame in REFERENCE_FRAMES:
            for invalid in (
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.1],
                [math.nan, 0.0, 0.0, 1.0],
                [0.0, math.inf, 0.0, 1.0],
                [False, 0.0, 0.0, 1.0],
                ["bad", 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                None,
            ):
                with self.subTest(frame=frame, invalid=invalid):
                    robot, tf = FakeRobot(), FakeTF()
                    tf.references[frame] = invalid
                    sender = NativeTrajectorySender(0.0)
                    with self.assertRaisesRegex(ValueError, frame):
                        sender.initialize_references(robot, tf)
                    self.assertEqual(robot.reference_calls, [])
                    self.assertEqual(robot.calls, [])
                    self.assertFalse(sender.snapshot()["reference_initialization"]["initialized"])
                    with self.assertRaisesRegex(RuntimeError, "initialize_references"):
                        sender(robot, POSE)

    def test_reference_read_failure_can_retry_without_any_sdk_calls(self):
        robot, tf = FakeRobot(), FakeTF()
        tf.references["arm_r_link3"] = RuntimeError("TF unavailable")
        sender = NativeTrajectorySender(0.0)
        with self.assertRaisesRegex(RuntimeError, "TF unavailable"):
            sender.initialize_references(robot, tf)
        self.assertEqual(robot.reference_calls, [])
        self.assertEqual(robot.calls, [])
        self.assertIsNone(sender.snapshot()["fault"])
        sender.initialize_references(robot, FakeTF())
        self.assertTrue(sender.snapshot()["reference_initialization"]["initialized"])

    def test_tiny_reference_norm_errors_normalize_in_xyzw_order(self):
        robot, tf = FakeRobot(), FakeTF()
        tf.references["arm_r_link3"] = [value * 1.00001 for value in RIGHT_REFERENCE]
        sender = NativeTrajectorySender(0.0)
        sender.initialize_references(robot, tf)
        for actual, expected in zip(robot.reference_calls[0], LEFT_REFERENCE + RIGHT_REFERENCE):
            self.assertAlmostEqual(actual, expected)

    def test_reference_setter_failure_latches_and_cannot_publish_or_retry(self):
        for result, error, message in (
            (23, None, "returned 23"),
            (None, None, "returned None"),
            (0, RuntimeError("reference setter failed"), "reference setter failed"),
        ):
            with self.subTest(result=result, error=error):
                robot = FakeRobot(reference_result=result, reference_error=error)
                sender = NativeTrajectorySender(0.0)
                with self.assertRaisesRegex(RuntimeError, message):
                    sender.initialize_references(robot, FakeTF())
                self.assertFalse(sender.snapshot()["reference_initialization"]["initialized"])
                with self.assertRaisesRegex(RuntimeError, "fault latched"):
                    sender(robot, POSE)
                with self.assertRaisesRegex(RuntimeError, "fault latched"):
                    sender.initialize_references(robot, FakeTF())
                self.assertEqual(len(robot.reference_calls), 1)
                self.assertEqual(robot.calls, [])

    def test_repeated_initialization_before_publication_keeps_frozen_references(self):
        robot, tf = FakeRobot(), FakeTF()
        sender = NativeTrajectorySender(0.0)
        original = sender.initialize_references(robot, tf)
        tf.references["arm_r_link3"] = [0.0, 0.0, 0.0, 1.0]
        self.assertEqual(sender.initialize_references(robot, tf), original)
        self.assertEqual(len(robot.reference_calls), 1)
        self.assertEqual(len(tf.events), 2)

    def test_initialization_after_publication_is_rejected(self):
        robot, tf = FakeRobot(), FakeTF()
        sender = NativeTrajectorySender(0.0)
        sender.initialize_references(robot, tf)
        sender(robot, POSE)
        with self.assertRaisesRegex(RuntimeError, "after trajectory publication"):
            sender.initialize_references(robot, tf)
        self.assertEqual(len(robot.reference_calls), 1)
        self.assertEqual(len(tf.events), 2)

    def test_references_cannot_be_reused_with_different_robot_object(self):
        robot, other = FakeRobot(), FakeRobot()
        sender = NativeTrajectorySender(0.0)
        sender.initialize_references(robot, FakeTF())
        before = sender.snapshot()
        with self.assertRaisesRegex(RuntimeError, "different Robot"):
            sender(other, POSE)
        with self.assertRaisesRegex(RuntimeError, "different Robot"):
            sender.initialize_references(other, FakeTF())
        self.assertEqual(sender.snapshot(), before)
        self.assertEqual(robot.calls, [])
        self.assertEqual(other.calls, [])
        self.assertEqual(other.reference_calls, [])

    def test_reference_diagnostics_are_deep_copies(self):
        robot = FakeRobot()
        sender = NativeTrajectorySender(0.0)
        refs = sender.initialize_references(robot, FakeTF())
        refs["reference_quaternions_xyzw"]["arm_r_link3"][0] = 999
        state = sender.snapshot()
        state["reference_initialization"]["reference_quaternions_xyzw"]["arm_l_link3"][0] = 999
        self.assertEqual(
            sender.snapshot()["reference_initialization"]["reference_quaternions_xyzw"],
            {"arm_l_link3": LEFT_REFERENCE, "arm_r_link3": RIGHT_REFERENCE},
        )

    def test_one_native_request_contains_only_right_arm_and_tool(self):
        robot = FakeRobot()
        sender = NativeTrajectorySender(-0.3)
        sender.initialize_references(robot, FakeTF())
        self.assertEqual(sender(robot, POSE), 0)
        self.assertEqual(len(robot.calls), 1)
        args, kwargs = robot.calls[0]
        self.assertEqual(args[:2], (0, {}))
        self.assertEqual(
            args[2],
            [
                {
                    "right_arm": {"control_type": "ABS_POSE", "action_data": POSE},
                    "right_effector": {"control_type": "ABS_JOINT", "action_data": [-0.3]},
                }
            ],
        )
        self.assertEqual(kwargs, {"robot_link": "base_link", "trajectory_reference_time": 0.02})

    def test_partial_opening_is_interpolated_without_endpoint_thresholds(self):
        robot = FakeRobot()
        sender = NativeTrajectorySender(-0.1)
        sender.initialize_references(robot, FakeTF())
        sender.set_target(-0.35)
        for _ in range(MODEL_TICKS):
            sender(robot, POSE)
        for actual, expected in zip(gripper_values(robot), [-0.15, -0.2, -0.25, -0.3, -0.35]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(gripper_values(robot)[-1], -0.35)

    def test_all_16_model_waypoints_reach_exact_targets_with_one_call_per_tick(self):
        targets = [
            0.0,
            -0.1,
            -0.22,
            -0.48,
            -0.7,
            -0.785,
            -0.75,
            -0.4,
            -0.25,
            -0.03,
            0.0,
            -0.2,
            -0.51,
            -0.6,
            -0.785,
            -0.32,
        ]
        sender = NativeTrajectorySender(0.0)
        robot = FakeRobot()
        sender.initialize_references(robot, FakeTF())
        previous = 0.0
        for index, target in enumerate(targets):
            pose = [0.5 + index * 0.001, *POSE[1:]]
            sender.set_target(target)
            for tick in range(1, MODEL_TICKS + 1):
                sender(robot, pose)
                args, _ = robot.calls[-1]
                self.assertEqual(args[2][0]["right_arm"]["action_data"], pose)
                self.assertAlmostEqual(
                    gripper_values(robot)[-1], previous + (target - previous) * tick / MODEL_TICKS
                )
            self.assertEqual(gripper_values(robot)[-1], target)
            previous = target
        state = sender.snapshot()
        self.assertEqual(len(robot.calls), 80)
        self.assertEqual(state["call_count"], 80)
        self.assertEqual(state["success_count"], 80)
        self.assertEqual(state["remaining_ticks"], 0)
        self.assertAlmostEqual(
            sum(kwargs["trajectory_reference_time"] for _, kwargs in robot.calls), 1.6
        )

    def test_close_and_reopen_are_not_latched(self):
        robot = FakeRobot()
        sender = NativeTrajectorySender(OPEN_GRIPPER)
        sender.initialize_references(robot, FakeTF())
        for target in (CLOSED_GRIPPER, OPEN_GRIPPER, CLOSED_GRIPPER):
            sender.set_target(target)
            for _ in range(MODEL_TICKS):
                sender(robot, POSE)
            self.assertEqual(gripper_values(robot)[-1], target)

    def test_idle_holds_last_target_without_restarting_ramp(self):
        robot = FakeRobot()
        sender = NativeTrajectorySender(-0.4)
        sender.initialize_references(robot, FakeTF())
        sender.set_target(-0.55)
        for _ in range(8):
            sender(robot, POSE)
        self.assertEqual(gripper_values(robot)[4:], [-0.55] * 4)
        self.assertEqual(sender.snapshot()["remaining_ticks"], 0)

    def test_new_target_interpolates_from_last_successful_sent_value(self):
        robot = FakeRobot()
        sender = NativeTrajectorySender(0.0)
        sender.initialize_references(robot, FakeTF())
        sender.set_target(-0.5)
        sender(robot, POSE)
        sender(robot, POSE)
        sender.set_target(-0.7)
        sender(robot, POSE)
        self.assertAlmostEqual(gripper_values(robot)[-1], -0.3)

    def test_custom_tick_count_changes_ramp_not_native_control_period(self):
        robot = FakeRobot()
        sender = NativeTrajectorySender(0.0)
        sender.initialize_references(robot, FakeTF())
        sender.set_target(-0.6, ticks=2)
        sender(robot, POSE)
        sender(robot, POSE)
        self.assertEqual(gripper_values(robot), [-0.3, -0.6])
        self.assertTrue(
            all(
                kwargs["trajectory_reference_time"] == CONTROL_PERIOD_S for _, kwargs in robot.calls
            )
        )

    def test_endpoint_float32_overshoot_only_is_clamped(self):
        for initial, expected in ((-0.7850000262260437, -0.785), (1e-8, 0.0)):
            with self.subTest(initial=initial):
                sender = NativeTrajectorySender(initial)
                robot = FakeRobot()
                sender.initialize_references(robot, FakeTF())
                sender.set_target(initial, ticks=1)
                sender(robot, POSE)
                self.assertEqual(gripper_values(robot), [expected])

    def test_in_range_values_are_not_quantized(self):
        robot = FakeRobot()
        sender = NativeTrajectorySender(-0.01)
        sender.initialize_references(robot, FakeTF())
        for target in (-0.7849999, -0.719999, -0.030001, -1e-8):
            sender.set_target(target, ticks=1)
            sender(robot, POSE)
            self.assertEqual(gripper_values(robot)[-1], target)

    def test_invalid_initial_gripper_is_rejected(self):
        for target in (math.nan, math.inf, -math.inf, -0.786, 0.1, True, None, "bad"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                NativeTrajectorySender(target)

    def test_invalid_target_leaves_pending_state_unchanged(self):
        sender = NativeTrajectorySender(-0.3)
        sender.set_target(-0.2)
        state = sender.snapshot()
        for target in (-0.785 - GRIPPER_FLOAT_TOLERANCE * 1.01, 0.001, math.nan, None):
            with self.subTest(target=target), self.assertRaises(ValueError):
                sender.set_target(target)
            self.assertEqual(sender.snapshot(), state)

    def test_invalid_tick_count_is_rejected(self):
        sender = NativeTrajectorySender(-0.3)
        for ticks in (0, -1, 1.5, True, None):
            with self.subTest(ticks=ticks), self.assertRaises(ValueError):
                sender.set_target(-0.2, ticks=ticks)

    def test_invalid_pose_never_reaches_sdk_or_consumes_tick(self):
        sender = NativeTrajectorySender(-0.3)
        sender.set_target(-0.2)
        robot = FakeRobot()
        sender.initialize_references(robot, FakeTF())
        state = sender.snapshot()
        for pose in (
            [],
            POSE[:6],
            POSE + [0],
            [math.nan, *POSE[1:]],
            [math.inf, *POSE[1:]],
            [*POSE[:3], 0, 0, 0, 0],
            [*POSE[:3], 0, 0, 0, 1.1],
            [*POSE[:3], 0, 0, 0, math.nan],
            [True, *POSE[1:]],
            None,
        ):
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                sender(robot, pose)
            self.assertEqual(sender.snapshot(), state)
        self.assertEqual(robot.calls, [])

    def test_tiny_quaternion_norm_error_is_normalized_in_xyzw_order(self):
        robot = FakeRobot()
        sender = NativeTrajectorySender(-0.3)
        sender.initialize_references(robot, FakeTF())
        pose = [*POSE[:3], 0.0, 0.600006, 0.0, 0.800008]
        before = pose.copy()
        sender(robot, pose)
        actual = robot.calls[0][0][2][0]["right_arm"]["action_data"]
        self.assertEqual(actual[:3], POSE[:3])
        for value, expected in zip(actual[3:], [0, 0.6, 0, 0.8]):
            self.assertAlmostEqual(value, expected)
        self.assertEqual(pose, before)

    def test_nonzero_sdk_result_is_recorded_and_latched_without_retry(self):
        robot = FakeRobot(result=17)
        sender = NativeTrajectorySender(-0.3)
        sender.initialize_references(robot, FakeTF())
        sender.set_target(-0.2)
        with self.assertRaisesRegex(RuntimeError, "returned 17"):
            sender(robot, POSE)
        state = sender.snapshot()
        self.assertEqual(state["call_count"], 1)
        self.assertEqual(state["success_count"], 0)
        self.assertEqual(state["last_result"], 17)
        self.assertEqual(state["remaining_ticks"], 5)
        self.assertIsNone(state["accepted_target"])
        self.assertAlmostEqual(state["sent_target"], -0.28)
        with self.assertRaisesRegex(RuntimeError, "fault latched"):
            sender(robot, POSE)
        with self.assertRaisesRegex(RuntimeError, "fault latched"):
            sender.set_target(-0.1)
        self.assertEqual(len(robot.calls), 1)

    def test_sdk_exception_keeps_previous_accepted_target_and_is_not_retried(self):
        robot = FakeRobot()
        sender = NativeTrajectorySender(-0.3)
        sender.initialize_references(robot, FakeTF())
        sender(robot, POSE)
        sender.set_target(-0.2)
        robot.error = RuntimeError("transport outcome unknown")
        with self.assertRaisesRegex(RuntimeError, "outcome unknown"):
            sender(robot, POSE)
        state = sender.snapshot()
        self.assertEqual(state["accepted_target"], -0.3)
        self.assertEqual(state["current_target"], -0.3)
        self.assertEqual(state["call_count"], 2)
        self.assertEqual(state["success_count"], 1)
        self.assertIsNone(state["last_result"])
        with self.assertRaisesRegex(RuntimeError, "fault latched"):
            sender(robot, POSE)
        self.assertEqual(len(robot.calls), 2)

    def test_none_sdk_result_is_not_claimed_success(self):
        robot = FakeRobot(result=None)
        sender = NativeTrajectorySender(-0.3)
        sender.initialize_references(robot, FakeTF())
        with self.assertRaisesRegex(RuntimeError, "returned None"):
            sender(robot, POSE)
        self.assertEqual(sender.snapshot()["success_count"], 0)

    def test_snapshot_is_a_copy_not_mutable_sender_state(self):
        sender = NativeTrajectorySender(-0.3)
        snapshot = sender.snapshot()
        snapshot["requested_target"] = 0.0
        self.assertEqual(sender.snapshot()["requested_target"], -0.3)


class ContinuousDiagnosticsTest(unittest.TestCase):
    def test_callback_runs_only_before_each_sdk_publication(self):
        events = []
        sender = NativeTrajectorySender(0.0, before_publish=lambda: events.append(("check",)))
        robot, tf = FakeRobot(events=events), FakeTF(events=events)
        sender.set_target(-0.35)
        self.assertEqual(events, [])
        sender.initialize_references(robot, tf)
        self.assertNotIn(("check",), events)
        for _ in range(5):
            sender(robot, POSE)
        self.assertEqual(events[3:], [("check",), ("publish",)] * 5)
        self.assertEqual(len(robot.calls), 5)

    def test_invalid_pose_and_uninitialized_reference_do_not_run_callback(self):
        callbacks = []
        sender = NativeTrajectorySender(0.0, before_publish=lambda: callbacks.append(True))
        robot = FakeRobot()
        with self.assertRaisesRegex(RuntimeError, "initialize_references"):
            sender(robot, POSE)
        sender.initialize_references(robot, FakeTF())
        with self.assertRaises(ValueError):
            sender(robot, [math.nan, *POSE[1:]])
        self.assertEqual(callbacks, [])
        self.assertEqual(robot.calls, [])

    def test_callback_failure_latches_without_sdk_attempt_or_consuming_ramp(self):
        callback_calls = []

        def check():
            callback_calls.append(True)
            raise RuntimeError("feedback fault")

        sender = NativeTrajectorySender(-0.3, before_publish=check)
        robot = FakeRobot()
        sender.initialize_references(robot, FakeTF())
        sender.set_target(-0.5)
        with self.assertRaisesRegex(RuntimeError, "feedback fault"):
            sender(robot, POSE)
        state = sender.snapshot()
        self.assertEqual(state["call_count"], 0)
        self.assertEqual(state["remaining_ticks"], 5)
        self.assertIsNone(state["sent_target"])
        self.assertIsNone(state["last_call_started_monotonic_ns"])
        self.assertIsNone(state["last_call_finished_monotonic_ns"])
        self.assertIn("before_publish", state["fault"])
        with self.assertRaisesRegex(RuntimeError, "fault latched"):
            sender(robot, POSE)
        with self.assertRaisesRegex(RuntimeError, "fault latched"):
            sender.set_target(0.0)
        self.assertEqual(callback_calls, [True])
        self.assertEqual(robot.calls, [])

    def test_callback_failure_after_success_keeps_previous_call_diagnostics(self):
        calls = []

        def check():
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError("stale feedback")

        times = iter([100, 150])
        sender = NativeTrajectorySender(-0.3, before_publish=check, clock_ns=lambda: next(times))
        robot = FakeRobot()
        sender.initialize_references(robot, FakeTF())
        sender(robot, POSE)
        sender.set_target(-0.6)
        with self.assertRaisesRegex(RuntimeError, "stale feedback"):
            sender(robot, POSE)
        state = sender.snapshot()
        self.assertEqual(state["last_call_started_monotonic_ns"], 100)
        self.assertEqual(state["last_call_finished_monotonic_ns"], 150)
        self.assertEqual(state["accepted_target"], -0.3)
        self.assertEqual(state["call_count"], 1)
        self.assertEqual(len(robot.calls), 1)

    def test_sdk_timing_is_measured_after_callback_and_records_actual_interval(self):
        events = []
        times = iter([1_000_000, 1_250_000, 21_300_000, 21_450_000])

        def clock():
            events.append(("clock",))
            return next(times)

        sender = NativeTrajectorySender(
            0.0, before_publish=lambda: events.append(("check",)), clock_ns=clock
        )
        robot = FakeRobot(events=events)
        sender.initialize_references(robot, FakeTF())
        self.assertIsNone(sender.snapshot()["last_call_interval_ns"])
        sender(robot, POSE)
        first = sender.snapshot()
        self.assertEqual(first["last_call_started_monotonic_ns"], 1_000_000)
        self.assertEqual(first["last_call_finished_monotonic_ns"], 1_250_000)
        self.assertEqual(first["last_call_duration_ns"], 250_000)
        self.assertIsNone(first["last_call_interval_ns"])
        sender(robot, POSE)
        last = sender.snapshot()
        self.assertEqual(last["last_call_started_monotonic_ns"], 21_300_000)
        self.assertEqual(last["last_call_finished_monotonic_ns"], 21_450_000)
        self.assertEqual(last["last_call_interval_ns"], 20_300_000)
        self.assertEqual(last["last_call_duration_ns"], 150_000)
        self.assertEqual(events[1:], [("check",), ("clock",), ("publish",), ("clock",)] * 2)
        self.assertTrue(
            all(kwargs["trajectory_reference_time"] == 0.02 for _, kwargs in robot.calls)
        )

    def test_failed_sdk_calls_still_record_finish_time_and_are_never_retried(self):
        for result, error in ((17, None), (0, RuntimeError("unknown outcome"))):
            with self.subTest(result=result, error=error):
                times = iter([100, 400])
                sender = NativeTrajectorySender(0.0, clock_ns=lambda: next(times))
                robot = FakeRobot(result=result, error=error)
                sender.initialize_references(robot, FakeTF())
                with self.assertRaises(RuntimeError):
                    sender(robot, POSE)
                state = sender.snapshot()
                self.assertEqual(state["last_call_started_monotonic_ns"], 100)
                self.assertEqual(state["last_call_finished_monotonic_ns"], 400)
                self.assertEqual(state["last_call_duration_ns"], 300)
                self.assertEqual(state["call_count"], 1)
                with self.assertRaisesRegex(RuntimeError, "fault latched"):
                    sender(robot, POSE)
                self.assertEqual(len(robot.calls), 1)

    def test_reference_source_timestamps_are_recorded_without_clock_substitution(self):
        tf = FakeTF()
        stamps = dict(zip(REFERENCE_FRAMES, [1234567890123, 1234567890124]))
        tf.get_latest_timestamp = lambda frame: stamps[frame]
        sender, robot = NativeTrajectorySender(0.0), FakeRobot()
        state = sender.initialize_references(robot, tf)
        self.assertEqual(state["reference_timestamps_ns"], stamps)
        self.assertEqual(state["reference_timestamp_source"], "TF.get_latest_timestamp")
        self.assertEqual(state["reference_timestamp_errors"], {})
        self.assertEqual(robot.calls, [])
        state["reference_timestamps_ns"][REFERENCE_FRAMES[0]] = 1
        self.assertEqual(
            sender.snapshot()["reference_initialization"]["reference_timestamps_ns"], stamps
        )
        stamps[REFERENCE_FRAMES[0]] += 100
        self.assertNotEqual(
            sender.initialize_references(robot, tf)["reference_timestamps_ns"], stamps
        )

    def test_missing_timestamp_api_is_explicitly_unavailable(self):
        sender = NativeTrajectorySender(0.0)
        references = sender.initialize_references(FakeRobot(), FakeTF())
        self.assertEqual(references["reference_timestamps_ns"], dict.fromkeys(REFERENCE_FRAMES))
        self.assertIsNone(references["reference_timestamp_source"])
        self.assertEqual(references["reference_timestamp_errors"], {})

    def test_invalid_or_unavailable_optional_timestamp_does_not_claim_freshness(self):
        for timestamp in (0, -1, None, True, 1.5, "123", RuntimeError("not available")):
            with self.subTest(timestamp=timestamp):
                tf = FakeTF()

                def read_timestamp(frame):
                    if isinstance(timestamp, Exception):
                        raise timestamp
                    return timestamp

                tf.get_latest_timestamp = read_timestamp
                sender, robot = NativeTrajectorySender(0.0), FakeRobot()
                refs = sender.initialize_references(robot, tf)
                self.assertTrue(refs["initialized"])
                self.assertEqual(refs["reference_timestamps_ns"], dict.fromkeys(REFERENCE_FRAMES))
                self.assertEqual(set(refs["reference_timestamp_errors"]), set(REFERENCE_FRAMES))
                self.assertEqual(robot.calls, [])

    def test_invalid_optional_callbacks_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "before_publish"):
            NativeTrajectorySender(0.0, before_publish=False)
        with self.assertRaisesRegex(TypeError, "clock_ns"):
            NativeTrajectorySender(0.0, clock_ns=None)


if __name__ == "__main__":
    unittest.main()
