"""CPU-only RTC timeline and observation-thread tests; no hardware/network calls."""

from __future__ import annotations

import queue
import threading
import time
import unittest

from agibot.rtc.live import ExecutionTimeline, SnapshotWorker


WALL_NS = 1_788_935_053_682_336_436
MONO_NS = 3_685_752_348_810


def metadata(offset_ns=0):
    return {
        "capture_started_wall_ns": WALL_NS,
        "capture_started_monotonic_ns": MONO_NS,
        "right_eef_tf_timestamp_ns": WALL_NS + offset_ns,
    }


def receipt(command_id, start_offset_ns, end_offset_ns):
    return {
        "command_id": command_id,
        "ok": True,
        "accepted": True,
        "execution_started_monotonic_ns": MONO_NS + start_offset_ns,
        "execution_finished_monotonic_ns": MONO_NS + end_offset_ns,
    }


class ExecutionTimelineTest(unittest.TestCase):
    def setUp(self):
        self.timeline = ExecutionTimeline()

    def completed_pair(self):
        self.timeline.submitted(0, "row0")
        self.timeline.submitted(1, "row1")
        self.timeline.update(
            {
                "recent_results": [
                    receipt("row0", 0, 100_000_000),
                    receipt("row1", 120_000_000, 220_000_000),
                ]
            }
        )

    def test_robot_epoch_to_monotonic_mapping_ignores_local_clock(self):
        self.completed_pair()
        index, details = self.timeline.observation_origin(metadata(50_000_000))
        self.assertEqual(index, 0)
        self.assertEqual(details["sample_robot_monotonic_ns"], MONO_NS + 50_000_000)
        self.assertEqual(details["sensor_to_capture_start_ms"], 50.0)
        self.assertEqual(details["alignment_phase"], "within_committed_row")

    def test_delayed_receipt_keeps_historical_observation_index(self):
        self.completed_pair()
        self.timeline.submitted(2, "row2")
        self.timeline.update({"recent_results": [receipt("row2", 220_000_000, 320_000_000)]})
        index, _ = self.timeline.observation_origin(metadata(35_000_000))
        self.assertEqual(index, 0)
        # The image may arrive after row 2, but must not be stamped as row 3.
        index, _ = self.timeline.observation_origin(metadata(170_000_000))
        self.assertEqual(index, 1)

    def test_exact_boundaries_and_inter_row_gap_choose_next_endpoint(self):
        self.completed_pair()
        for offset, expected, phase in (
            (-10_000_000, 0, "before_first_row"),
            (0, 0, "within_committed_row"),
            (99_999_999, 0, "within_committed_row"),
            (100_000_000, 1, "after_completed_row"),
            (110_000_000, 1, "after_completed_row"),
            (120_000_000, 1, "within_committed_row"),
            (220_000_000, 2, "after_completed_row"),
        ):
            with self.subTest(offset=offset):
                index, details = self.timeline.observation_origin(metadata(offset))
                self.assertEqual(index, expected)
                self.assertEqual(details["alignment_phase"], phase)

    def test_active_row_is_committed_even_before_completion_receipt(self):
        self.timeline.submitted(0, "row0")
        self.timeline.submitted(1, "row1")
        self.timeline.update(
            {
                "recent_results": [receipt("row0", 0, 100_000_000)],
                "active_command_id": "row1",
                "active_command_started_monotonic_ns": MONO_NS + 100_000_000,
            }
        )
        index, details = self.timeline.observation_origin(metadata(150_000_000))
        self.assertEqual(index, 1)
        self.assertEqual(details["alignment_phase"], "within_committed_row")
        self.timeline.update({"recent_results": [receipt("row1", 100_000_000, 200_000_000)]})
        index, details = self.timeline.observation_origin(metadata(210_000_000))
        self.assertEqual(index, 2)
        self.assertEqual(details["alignment_phase"], "after_completed_row")

    def test_ack_only_without_execution_time_is_not_completion(self):
        self.timeline.submitted(0, "row0")
        self.timeline.update({"recent_results": [], "queue_depth": 1})
        index, details = self.timeline.observation_origin(metadata())
        self.assertEqual(index, 0)
        self.assertEqual(details["alignment_phase"], "before_first_row")
        self.assertIsNone(self.timeline.rows[0]["start_ns"])

    def test_unknown_receipts_are_ignored_and_duplicate_updates_are_idempotent(self):
        self.timeline.submitted(0, "row0")
        known = receipt("row0", 0, 100_000_000)
        status = {"recent_results": [{"command_id": "foreign", "ok": False}, known]}
        self.timeline.update(status)
        self.timeline.update(status)
        self.assertEqual(len(self.timeline.rows), 1)
        self.assertEqual(self.timeline.observation_origin(metadata(50_000_000))[0], 0)

    def test_duplicate_index_or_command_identity_is_rejected(self):
        self.timeline.submitted(0, "row0")
        for index, command_id in ((0, "new"), (1, "row0")):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "duplicate"):
                self.timeline.submitted(index, command_id)

    def test_failed_receipt_and_invalid_execution_timestamps_are_rejected(self):
        self.timeline.submitted(0, "row0")
        for changes in (
            {"ok": False},
            {"accepted": False},
            {"execution_started_monotonic_ns": None},
            {"execution_finished_monotonic_ns": 0},
            {"execution_started_monotonic_ns": MONO_NS + 200_000_000},
        ):
            item = receipt("row0", 0, 100_000_000)
            item.update(changes)
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                self.timeline.update({"recent_results": [item]})

    def test_capture_processing_end_does_not_retimestamp_sensor(self):
        self.completed_pair()
        sample = metadata(50_000_000)
        sample["capture_finished_monotonic_ns"] = MONO_NS + 500_000_000
        sample["capture_duration_ms"] = 500
        self.assertEqual(self.timeline.observation_origin(sample)[0], 0)

    def test_missing_or_wrong_domain_metadata_does_not_guess_clock(self):
        for sample in (
            {},
            {**metadata(), "right_eef_tf_timestamp_ns": 0},
            {**metadata(), "capture_started_monotonic_ns": 0},
            metadata(1_000_000_001),
            metadata(-1_000_000_001),
            {**metadata(), "right_eef_tf_timestamp_ns": MONO_NS},
        ):
            with self.subTest(sample=sample), self.assertRaises((ValueError, KeyError)):
                self.timeline.observation_origin(sample)


class FakeObservationClient:
    def __init__(self):
        self.samples = queue.Queue()
        self.read_started = threading.Event()
        self.closed = threading.Event()
        self.thread_calls = []

    def __enter__(self):
        self.thread_calls.append(("enter", threading.get_ident()))
        return self

    def __exit__(self, *args):
        self.thread_calls.append(("exit", threading.get_ident()))
        self.closed.set()

    def get_snapshot(self):
        self.thread_calls.append(("snapshot", threading.get_ident()))
        self.read_started.set()
        item = self.samples.get(timeout=2)
        if isinstance(item, Exception):
            raise item
        return item


class SnapshotWorkerTest(unittest.TestCase):
    def setUp(self):
        self.client = FakeObservationClient()
        self.factory_threads = []

        def factory():
            self.factory_threads.append(threading.get_ident())
            return self.client

        self.worker = SnapshotWorker(factory)
        self.assertTrue(self.client.read_started.wait(1))

    def tearDown(self):
        # Release any in-flight fake I/O while exercising the normal close path.
        timer = threading.Timer(0.02, lambda: self.client.samples.put(object()))
        timer.start()
        try:
            self.assertTrue(self.worker.close())
        finally:
            timer.join(1)

    def wait_latest(self, expected):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            latest = self.worker.latest()
            if latest is not None and latest[0] is expected:
                return latest
            time.sleep(0.005)
        self.fail("background observation did not arrive")

    def test_latest_does_not_block_while_socket_thread_waits_for_data(self):
        before = time.monotonic()
        self.assertIsNone(self.worker.latest())
        self.assertLess(time.monotonic() - before, 0.1)
        self.assertFalse(self.client.closed.is_set())

    def test_factory_enter_capture_exit_are_owned_by_one_worker_thread(self):
        sample = object()
        self.client.samples.put(sample)
        self.wait_latest(sample)
        self.assertTrue(self.worker.close())
        self.assertTrue(self.client.closed.is_set())
        thread_ids = {ident for _, ident in self.client.thread_calls}
        self.assertEqual(thread_ids, set(self.factory_threads))
        self.assertEqual(len(thread_ids), 1)
        self.assertNotIn(threading.get_ident(), thread_ids)

    def test_only_latest_observation_is_retained_with_local_request_interval(self):
        first, second = object(), object()
        self.client.samples.put(first)
        first_latest = self.wait_latest(first)
        self.client.samples.put(second)
        second_latest = self.wait_latest(second)
        self.assertIs(self.worker.latest()[0], second)
        self.assertLessEqual(first_latest[1], first_latest[2])
        self.assertLessEqual(first_latest[2], second_latest[1])
        self.assertLessEqual(second_latest[1], second_latest[2])

    def test_background_io_error_is_propagated_to_main_without_deadlock(self):
        error = OSError("camera connection failed")
        self.client.samples.put(error)
        self.assertTrue(self.client.closed.wait(1))
        with self.assertRaisesRegex(RuntimeError, "camera connection failed") as raised:
            self.worker.latest()
        self.assertIs(raised.exception.__cause__, error)
        self.assertTrue(self.worker.close())


if __name__ == "__main__":
    unittest.main()
