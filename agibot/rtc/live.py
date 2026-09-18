"""Sensor/command time alignment and background observations; no robot commands."""

from __future__ import annotations

import threading
import time


class ExecutionTimeline:
    """Map same-robot timestamps to discrete 10 Hz command rows.

    A sample inside a row is associated with that row's endpoint, which is kept
    frozen because it has already been submitted. Resolution is one action row,
    not servo-tick precision or a claim of physical target settlement.
    """

    def __init__(self):
        self.by_id = {}
        self.rows = {}

    def submitted(self, index, command_id):
        if index in self.rows or command_id in self.by_id:
            raise ValueError("duplicate execution identity")
        self.by_id[command_id] = index
        self.rows[index] = {"command_id": command_id, "start_ns": None, "end_ns": None}

    def update(self, status):
        for receipt in status.get("recent_results", []):
            index = self.by_id.get(receipt.get("command_id"))
            if index is None:
                continue
            if receipt.get("ok") is not True or receipt.get("accepted") is not True:
                raise RuntimeError(f"execution failed: {receipt}")
            start = receipt.get("execution_started_monotonic_ns")
            end = receipt.get("execution_finished_monotonic_ns")
            if type(start) is not int or type(end) is not int or not 0 < start <= end:
                raise RuntimeError("bridge lacks per-row execution timestamps")
            self.rows[index].update(start_ns=start, end_ns=end)
        index = self.by_id.get(status.get("active_command_id"))
        if index is not None:
            start = status.get("active_command_started_monotonic_ns")
            if type(start) is int and start > 0:
                self.rows[index]["start_ns"] = start

    def observation_origin(self, metadata):
        wall = int(metadata["capture_started_wall_ns"])
        mono = int(metadata["capture_started_monotonic_ns"])
        sensor = int(metadata["right_eef_tf_timestamp_ns"])
        if min(wall, mono, sensor) <= 0 or abs(sensor - wall) > 1_000_000_000:
            raise ValueError("sensor timestamp is not in the robot capture wall-clock domain")
        sample_ns = mono + sensor - wall
        started = [
            (i, r)
            for i, r in self.rows.items()
            if r["start_ns"] is not None and r["start_ns"] <= sample_ns
        ]
        if not started:
            index, phase = 0, "before_first_row"
        else:
            index, row = max(started, key=lambda item: item[0])
            if row["end_ns"] is not None and sample_ns >= row["end_ns"]:
                index, phase = index + 1, "after_completed_row"
            else:
                phase = "within_committed_row"
        return index, {
            "observation_start_index": index,
            "sample_robot_monotonic_ns": sample_ns,
            "sensor_to_capture_start_ms": (sensor - wall) / 1e6,
            "alignment_phase": phase,
            "alignment_resolution": "one_10hz_row_endpoint_not_physical_settlement",
        }


class SnapshotWorker:
    """Own the observation socket in one thread; retain only the latest sample."""

    def __init__(self, client_factory):
        self._factory = client_factory
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._error = None
        self._thread = threading.Thread(target=self._work, name="rtc-observation", daemon=True)
        self._thread.start()

    def _work(self):
        try:
            with self._factory() as client:
                while not self._stop.is_set():
                    before = time.monotonic()
                    snapshot = client.get_snapshot()
                    after = time.monotonic()
                    with self._lock:
                        self._latest = (snapshot, before, after)
                    self._stop.wait(max(0.0, 0.08 - (after - before)))
        except Exception as error:
            with self._lock:
                self._error = error

    def latest(self):
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f"observation worker failed: {self._error}") from self._error
            return self._latest

    def close(self):
        self._stop.set()
        self._thread.join(3.0)
        return not self._thread.is_alive()
