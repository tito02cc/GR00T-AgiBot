"""CPU-only queue tests: no robot, model weights, networking or GDK imports."""

import concurrent.futures
import dataclasses
import threading
import unittest

from agibot.rtc.action_queue import ActionQueue, SnapshotToken
import numpy as np


def chunk(offset=0.0, horizon=16):
    values = np.zeros((horizon, 8))
    values[:, 0] = np.arange(horizon) + offset
    values[:, 6] = 1
    values[:, 7] = np.linspace(-0.785, 0.0, horizon)
    return values


class ActionQueueTest(unittest.TestCase):
    def make_queue(self, targets=None):
        queue = ActionQueue()
        queue.initialize(chunk() if targets is None else targets)
        return queue

    def drain(self, queue):
        rows = []
        while (row := queue.pop()) is not None:
            rows.append(row)
        return rows

    def test_initial_chunk_indices_and_terminal_empty(self):
        queue = self.make_queue()
        rows = self.drain(queue)
        self.assertEqual([i for i, _ in rows], list(range(16)))
        np.testing.assert_array_equal(np.stack([target for _, target in rows]), chunk())
        self.assertEqual(queue.next_index, 16)
        self.assertEqual(queue.remaining, 0)
        self.assertIsNone(queue.pop())
        with self.assertRaisesRegex(RuntimeError, "empty"):
            queue.snapshot(0)
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            queue.initialize(chunk())

    def test_uninitialized_operations_fail(self):
        queue = ActionQueue()
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            queue.pop()
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            queue.snapshot(0)

    def test_snapshot_contains_exact_remaining_chunk_and_only_one_pending(self):
        queue = self.make_queue()
        queue.pop()
        queue.pop()
        token = queue.snapshot(4)
        self.assertEqual((token.start_index, token.overlap_steps, token.frozen_steps), (2, 14, 4))
        self.assertEqual(token.frozen_end_index, 6)
        np.testing.assert_array_equal(token.previous_targets, chunk()[2:])
        self.assertTrue(queue.pending)
        self.assertIs(queue.pending_token, token)
        with self.assertRaisesRegex(RuntimeError, "already pending"):
            queue.snapshot(4)

    def test_historical_snapshot_includes_committed_rows_without_replaying_them(self):
        queue = self.make_queue()
        committed = [queue.pop() for _ in range(8)]
        token = queue.snapshot(4, start_index=6)
        self.assertEqual((token.start_index, token.overlap_steps), (6, 10))
        np.testing.assert_array_equal(token.previous_targets, chunk()[6:])
        committed.append(queue.pop())
        metrics = queue.merge(token, chunk(100))
        self.assertTrue(metrics.accepted)
        self.assertEqual(metrics.consumed_steps, 3)
        self.assertEqual(metrics.retained_range, (9, 10))
        self.assertEqual(metrics.replacement_range, (10, 22))
        rows = committed + self.drain(queue)
        self.assertEqual([index for index, _ in rows], list(range(22)))
        np.testing.assert_array_equal(np.stack([row for _, row in rows[:10]]), chunk()[:10])
        np.testing.assert_array_equal(np.stack([row for _, row in rows[10:]]), chunk(100)[4:])

    def test_historical_snapshot_rejects_invalid_start_and_insufficient_freeze(self):
        queue = self.make_queue()
        for _ in range(8):
            queue.pop()
        for start in (-1, 9, 1.5, True):
            with self.subTest(start=start), self.assertRaisesRegex(ValueError, "start_index"):
                queue.snapshot(4, start_index=start)
            self.assertFalse(queue.pending)
        with self.assertRaisesRegex(ValueError, "covering committed"):
            queue.snapshot(1, start_index=6)
        self.assertFalse(queue.pending)
        with self.assertRaisesRegex(ValueError, "horizon"):
            queue.snapshot(8, start_index=0)
        self.assertFalse(queue.pending)
        self.assertEqual(queue.next_index, 8)
        self.assertEqual(queue.remaining, 8)

    def test_explicit_current_start_preserves_contract_and_default_full_horizon(self):
        queue = self.make_queue()
        with self.assertRaisesRegex(ValueError, "horizon"):
            queue.snapshot(4, start_index=0)
        default = queue.snapshot(4)
        self.assertEqual(default.overlap_steps, 16)
        queue.cancel(default)
        queue.pop()
        explicit = queue.snapshot(4, start_index=1)
        self.assertEqual(explicit.overlap_steps, 15)
        np.testing.assert_array_equal(explicit.previous_targets, chunk()[1:])

    def test_historical_snapshot_can_extend_drained_queue_without_replay(self):
        queue = self.make_queue()
        self.drain(queue)
        with self.assertRaisesRegex(ValueError, "horizon"):
            queue.snapshot(0, start_index=16)
        token = queue.snapshot(3, start_index=13)
        np.testing.assert_array_equal(token.previous_targets, chunk()[13:])
        metrics = queue.merge(token, chunk(100))
        self.assertTrue(metrics.accepted)
        self.assertEqual(metrics.consumed_steps, 3)
        self.assertEqual(metrics.replacement_range, (16, 29))
        rows = self.drain(queue)
        self.assertEqual([index for index, _ in rows], list(range(16, 29)))
        np.testing.assert_array_equal(np.stack([row for _, row in rows]), chunk(100)[3:])

    def test_history_retains_only_last_32_committed_rows(self):
        queue = self.make_queue()
        for cycle in range(5):
            for _ in range(8):
                queue.pop()
            token = queue.snapshot(0)
            queue.merge(token, chunk(100 * (cycle + 1)))
        self.assertEqual(queue.next_index, 40)
        self.assertEqual(len(queue._history), 32)
        self.assertEqual([index for index, _ in queue._history], list(range(8, 40)))
        with self.assertRaisesRegex(ValueError, "gap"):
            queue.snapshot(40, start_index=7)
        self.assertFalse(queue.pending)

    def test_history_across_prediction_generations_contains_actual_committed_targets(self):
        queue = self.make_queue()
        for _ in range(8):
            queue.pop()
        token = queue.snapshot(4, start_index=6)
        queue.merge(token, chunk(100))
        for _ in range(5):
            queue.pop()
        token = queue.snapshot(6, start_index=8)
        expected = np.concatenate((chunk()[8:10], chunk(100)[4:]))
        np.testing.assert_array_equal(token.previous_targets, expected)
        with self.assertRaises(ValueError):
            token.previous_targets.setflags(write=True)
        self.assertTrue(queue.merge(token, chunk(200)).accepted)
        np.testing.assert_array_equal(queue.pop()[1], chunk(100)[7])
        np.testing.assert_array_equal(queue.pop()[1], chunk(200)[6])

    def test_historical_late_merge_keeps_all_future_targets_unchanged(self):
        queue = self.make_queue()
        for _ in range(8):
            queue.pop()
        token = queue.snapshot(3, start_index=6)
        queue.pop()
        queue.pop()
        metrics = queue.merge(token, chunk(100))
        self.assertFalse(metrics.accepted)
        self.assertEqual(metrics.consumed_steps, 4)
        rows = self.drain(queue)
        self.assertEqual([index for index, _ in rows], list(range(10, 16)))
        np.testing.assert_array_equal(np.stack([row for _, row in rows]), chunk()[10:])

    def test_simultaneous_historical_snapshot_and_pop_have_consistent_prefix(self):
        for _ in range(30):
            queue = self.make_queue()
            for _ in range(8):
                queue.pop()
            barrier = threading.Barrier(2)

            def consume():
                barrier.wait(timeout=3)
                return queue.pop()

            def request():
                barrier.wait(timeout=3)
                return queue.snapshot(4, start_index=6)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                row_future = pool.submit(consume)
                token_future = pool.submit(request)
                row = row_future.result(timeout=3)
                token = token_future.result(timeout=3)
            self.assertEqual(row[0], 8)
            self.assertEqual(token.start_index, 6)
            np.testing.assert_array_equal(token.previous_targets, chunk()[6:])
            self.assertTrue(queue.merge(token, chunk(100)).accepted)
            self.assertEqual(queue.pop()[0], 9)

    def test_simultaneous_historical_merge_and_pop_do_not_replay(self):
        for _ in range(30):
            queue = self.make_queue()
            for _ in range(8):
                queue.pop()
            token = queue.snapshot(2, start_index=6)
            barrier = threading.Barrier(2)

            def consume():
                barrier.wait(timeout=3)
                return queue.pop()

            def complete():
                barrier.wait(timeout=3)
                return queue.merge(token, chunk(100))

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                row_future = pool.submit(consume)
                merge_future = pool.submit(complete)
                row = row_future.result(timeout=3)
                metrics = merge_future.result(timeout=3)
            self.assertEqual(row[0], 8)
            expected = chunk(100)[2] if metrics.accepted else chunk()[8]
            np.testing.assert_array_equal(row[1], expected)
            rows = self.drain(queue)
            self.assertEqual(
                [index for index, _ in rows], list(range(9, 22 if metrics.accepted else 16))
            )

    def test_merge_retains_exact_frozen_prefix_while_request_is_running(self):
        queue = self.make_queue()
        for _ in range(6):
            queue.pop()
        token = queue.snapshot(4)
        committed = [queue.pop(), queue.pop()]
        predicted = chunk(100)
        metrics = queue.merge(token, predicted)
        self.assertTrue(metrics.accepted)
        self.assertEqual(metrics.consumed_steps, 2)
        self.assertEqual(metrics.retained_range, (8, 10))
        self.assertEqual(metrics.replacement_range, (10, 22))
        self.assertEqual(metrics.remaining, 14)
        self.assertEqual(metrics.queue_end_index, 22)
        self.assertEqual(metrics.to_dict()["request_start_index"], 6)
        rows = committed + self.drain(queue)
        self.assertEqual([i for i, _ in rows], list(range(6, 22)))
        np.testing.assert_array_equal(np.stack([t for _, t in rows[:4]]), chunk()[6:10])
        np.testing.assert_array_equal(np.stack([t for _, t in rows[4:]]), predicted[4:])

    def test_d_equal_frozen_is_allowed_without_replaying_committed(self):
        queue = self.make_queue()
        token = queue.snapshot(3)
        for _ in range(3):
            queue.pop()
        metrics = queue.merge(token, chunk(100))
        self.assertTrue(metrics.accepted)
        self.assertEqual(metrics.retained_range, (3, 3))
        index, target = queue.pop()
        self.assertEqual(index, 3)
        np.testing.assert_array_equal(target, chunk(100)[3])

    def test_zero_frozen_replaces_all_only_if_no_pop(self):
        queue = self.make_queue()
        token = queue.snapshot(0)
        self.assertTrue(queue.merge(token, chunk(100)).accepted)
        np.testing.assert_array_equal(queue.pop()[1], chunk(100)[0])
        token = queue.snapshot(0)
        queue.pop()
        self.assertFalse(queue.merge(token, chunk(200)).accepted)
        np.testing.assert_array_equal(queue.pop()[1], chunk(100)[2])

    def test_late_prediction_does_not_change_or_replay_queue(self):
        queue = self.make_queue()
        token = queue.snapshot(3)
        for _ in range(4):
            queue.pop()
        metrics = queue.merge(token, chunk(100))
        self.assertFalse(metrics.accepted)
        self.assertEqual(metrics.reason, "stale_prediction")
        self.assertEqual(metrics.consumed_steps, 4)
        self.assertEqual(metrics.retained_range, (4, 16))
        self.assertIsNone(metrics.replacement_range)
        self.assertEqual(metrics.remaining, 12)
        self.assertFalse(queue.pending)
        new_token = queue.snapshot(2)
        self.assertGreater(new_token.generation, token.generation)
        queue.cancel(new_token)
        rows = self.drain(queue)
        self.assertEqual([i for i, _ in rows], list(range(4, 16)))
        np.testing.assert_array_equal(np.stack([t for _, t in rows]), chunk()[4:])

    def test_drained_queue_can_receive_aligned_pending_suffix(self):
        queue = self.make_queue()
        for _ in range(12):
            queue.pop()
        token = queue.snapshot(4)
        self.drain(queue)
        self.assertEqual(queue.next_index, 16)
        metrics = queue.merge(token, chunk(100))
        self.assertTrue(metrics.accepted)
        self.assertEqual(metrics.replacement_range, (16, 28))
        self.assertEqual(queue.pop()[0], 16)

    def test_drained_queue_with_late_prediction_stays_empty(self):
        queue = self.make_queue()
        token = queue.snapshot(4)
        self.drain(queue)
        self.assertFalse(queue.merge(token, chunk(100)).accepted)
        self.assertIsNone(queue.pop())
        self.assertEqual(queue.next_index, 16)

    def test_full_frozen_chunk_is_not_replaced(self):
        queue = self.make_queue()
        token = queue.snapshot(16)
        metrics = queue.merge(token, chunk(100))
        self.assertEqual(metrics.replacement_range, (16, 16))
        np.testing.assert_array_equal(np.stack([t for _, t in self.drain(queue)]), chunk())

    def test_invalid_frozen_prefix_leaves_queue_available(self):
        queue = self.make_queue()
        queue.pop()
        for frozen in (-1, 16, 1.5, True, None):
            with self.subTest(frozen=frozen), self.assertRaises(ValueError):
                queue.snapshot(frozen)
            self.assertFalse(queue.pending)
            self.assertEqual(queue.remaining, 15)

    def test_repeated_foreign_and_forged_tokens_cannot_replace_queue(self):
        queue = self.make_queue()
        token = queue.snapshot(2)
        other = self.make_queue()
        foreign = other.snapshot(2)
        forged = SnapshotToken(
            token.start_index, token.generation, token.previous_targets, token.frozen_steps
        )
        for invalid in (foreign, forged, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                queue.merge(invalid, chunk(100))
            self.assertIs(queue.pending_token, token)
        queue.merge(token, chunk(100))
        with self.assertRaises(ValueError):
            queue.merge(token, chunk(200))

    def test_cancel_only_current_request_and_preserves_data(self):
        queue = self.make_queue()
        token = queue.snapshot(3)
        self.assertTrue(queue.cancel(token))
        self.assertFalse(queue.cancel(token))
        new_token = queue.snapshot(3)
        self.assertFalse(queue.cancel(token))
        self.assertIs(queue.pending_token, new_token)
        with self.assertRaises(ValueError):
            queue.merge(token, chunk(100))
        queue.cancel(new_token)
        np.testing.assert_array_equal(np.stack([t for _, t in self.drain(queue)]), chunk())

    def test_bad_predictions_leave_pending_available_for_cancel(self):
        queue = self.make_queue()
        token = queue.snapshot(3)
        cases = [chunk()[:15], np.zeros((16, 8)), chunk().astype(complex), [["x"]]]
        for bad_value in (float("inf"), float("-inf"), float("nan")):
            invalid = chunk()
            invalid[4, 0] = bad_value
            cases.append(invalid)
        for invalid in cases:
            with self.subTest(shape=np.shape(invalid)), self.assertRaises(ValueError):
                queue.merge(token, invalid)
            self.assertIs(queue.pending_token, token)
            self.assertEqual(queue.remaining, 16)
        self.assertTrue(queue.cancel(token))

    def test_horizon_is_explicit_not_model_padding(self):
        for horizon in (0, -1, 1.5, True):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                ActionQueue(horizon)
        with self.assertRaises(ValueError):
            self.make_queue(chunk(horizon=40))
        queue = ActionQueue(horizon=4)
        queue.initialize(chunk(horizon=4))
        self.assertEqual(queue.remaining, 4)

    def test_external_mutation_cannot_change_queue_token_or_popped_target(self):
        source = chunk()
        queue = self.make_queue(source)
        source[:] = 200
        token = queue.snapshot(3)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            token.frozen_steps = 4
        with self.assertRaises(ValueError):
            token.previous_targets[0, 0] = 500
        with self.assertRaises(ValueError):
            token.previous_targets.setflags(write=True)
        index, first = queue.pop()
        self.assertEqual(index, 0)
        np.testing.assert_array_equal(first, chunk()[0])
        with self.assertRaises(ValueError):
            first.setflags(write=True)
        predicted = chunk(100)
        queue.merge(token, predicted)
        predicted[:] = 999
        rows = self.drain(queue)
        np.testing.assert_array_equal(rows[0][1], chunk()[1])
        np.testing.assert_array_equal(rows[2][1], chunk(100)[3])

    def test_gripper_and_quaternion_values_are_never_smoothed_or_normalized(self):
        values = chunk()
        values[:, 3:7] *= 2
        values[:, 7] = np.arange(16) / 13 - 0.83
        queue = self.make_queue(values)
        predicted = values + np.array([10, 0, 0, 0, 0, 0, 0, 0.111])
        token = queue.snapshot(4)
        queue.merge(token, predicted)
        actual = np.stack([t for _, t in self.drain(queue)])
        np.testing.assert_array_equal(actual[:4], values[:4])
        np.testing.assert_array_equal(actual[4:], predicted[4:])

    def test_two_simultaneous_snapshot_requests_have_exactly_one_winner(self):
        queue = self.make_queue()
        barrier = threading.Barrier(2)

        def request():
            barrier.wait(timeout=3)
            try:
                return queue.snapshot(3)
            except RuntimeError:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: request(), range(2)))
        winners = [r for r in results if r is not None]
        self.assertEqual(len(winners), 1)
        self.assertIs(queue.pending_token, winners[0])

    def test_simultaneous_pop_and_merge_have_serializable_boundary(self):
        # At the frozen boundary either merge wins and replaces index 1, or pop
        # commits old index 1 first and makes the prediction stale. Both are valid.
        for _ in range(50):
            queue = self.make_queue()
            token = queue.snapshot(1)
            queue.pop()
            barrier = threading.Barrier(2)

            def consume():
                barrier.wait(timeout=3)
                return queue.pop()

            def complete():
                barrier.wait(timeout=3)
                return queue.merge(token, chunk(100))

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                row_future = pool.submit(consume)
                merge_future = pool.submit(complete)
                row = row_future.result(timeout=3)
                metrics = merge_future.result(timeout=3)
            self.assertEqual(row[0], 1)
            expected = chunk(100) if metrics.accepted else chunk()
            np.testing.assert_array_equal(row[1], expected[1])
            rows = self.drain(queue)
            self.assertEqual([i for i, _ in rows], list(range(2, 16)))
            np.testing.assert_array_equal(np.stack([t for _, t in rows]), expected[2:])

    def test_simultaneous_snapshot_and_pop_capture_consistent_index_and_prefix(self):
        for _ in range(30):
            queue = self.make_queue()
            barrier = threading.Barrier(2)

            def consume():
                barrier.wait(timeout=3)
                return queue.pop()

            def request():
                barrier.wait(timeout=3)
                return queue.snapshot(3)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                row_future = pool.submit(consume)
                token_future = pool.submit(request)
                row = row_future.result(timeout=3)
                token = token_future.result(timeout=3)
            self.assertEqual(row[0], 0)
            self.assertIn(token.start_index, (0, 1))
            np.testing.assert_array_equal(token.previous_targets, chunk()[token.start_index :])
            metrics = queue.merge(token, chunk(100))
            self.assertTrue(metrics.accepted)
            self.assertEqual(metrics.consumed_steps, 1 - token.start_index)


if __name__ == "__main__":
    unittest.main()
