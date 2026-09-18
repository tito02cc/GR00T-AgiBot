"""Thread-safe, absolute-index action queue for experimental asynchronous RTC.

This module does not send robot commands or smooth/model-transform targets. A
``pop`` commits one target to the caller: a later prediction can never replace it.
Predictions are aligned with the snapshot's start index, not the merge-time index.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import threading
from typing import Any

import numpy as np


def _immutable_array(values: np.ndarray) -> np.ndarray:
    """Copy onto immutable bytes, including protection against setflags(write=True)."""
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(contiguous.shape)


@dataclass(frozen=True, eq=False)
class SnapshotToken:
    """One outstanding prediction request, with an immutable old action prefix."""

    start_index: int
    generation: int
    previous_targets: np.ndarray
    frozen_steps: int

    @property
    def overlap_steps(self) -> int:
        return len(self.previous_targets)

    @property
    def frozen_end_index(self) -> int:
        return self.start_index + self.frozen_steps


@dataclass(frozen=True)
class MergeMetrics:
    """All index ranges are half-open; rejected merges leave the queue untouched."""

    accepted: bool
    reason: str
    generation: int
    request_start_index: int
    next_index: int
    consumed_steps: int
    frozen_steps: int
    prediction_end_index: int
    retained_range: tuple[int, int]
    replacement_range: tuple[int, int] | None
    remaining: int
    queue_end_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionQueue:
    """Queue of XYZ + XYZW quaternion + native gripper targets.

    ``horizon`` is the effective trained action horizon (16 for this checkpoint),
    not a padded model tensor length. ``initialize`` and ``merge`` require exactly
    that many rows. One pending snapshot is allowed at a time.

    Reserve ``frozen_steps`` to cover actions that may be committed during model
    inference. A merge retains the old frozen prefix verbatim and replaces only
    the remaining future suffix. If more than that prefix has already been popped,
    the prediction is stale: it is rejected without replay or changing the queue.

    Callers must keep execution ordering consistent with pop ordering. The queue
    treats pop as commitment, not confirmation that physical execution completed.
    """

    def __init__(self, horizon: int = 16):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        self._lock = threading.RLock()
        self._targets = _immutable_array(np.empty((0, 8)))
        self._history: deque[tuple[int, np.ndarray]] = deque(maxlen=32)
        self._next_index = 0
        self._generation = 0
        self._initialized = False
        self._pending: SnapshotToken | None = None

    def _validate_targets(self, targets: Any) -> np.ndarray:
        try:
            values = np.asarray(targets)
            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise ValueError("targets must contain real numbers")
            values = np.array(values, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("targets must be a real numeric array") from exc
        if values.shape != (self.horizon, 8):
            raise ValueError(f"targets must have shape ({self.horizon}, 8), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("targets must contain only finite values")
        # Nonzero max magnitude is equivalent to a nonzero norm without norm
        # underflow/overflow. Do not normalize or change predicted quaternions.
        if np.any(np.max(np.abs(values[:, 3:7]), axis=1) == 0):
            raise ValueError("each target quaternion must have nonzero norm")
        return _immutable_array(values)

    def initialize(self, targets: Any) -> None:
        """Seed one complete chunk exactly once; indices cannot be reset/replayed."""
        values = self._validate_targets(targets)
        with self._lock:
            if self._initialized:
                raise RuntimeError("queue is already initialized; create a new queue for a new run")
            self._targets = values
            self._initialized = True

    def pop(self) -> tuple[int, np.ndarray] | None:
        """Atomically commit the next row, or return None when no row remains."""
        with self._lock:
            if not self._initialized:
                raise RuntimeError("queue is not initialized")
            if not len(self._targets):
                return None
            index = self._next_index
            target = self._targets[0]
            self._targets = self._targets[1:]
            self._history.append((index, target))
            self._next_index += 1
            return index, target

    def snapshot(self, frozen_steps: int, *, start_index: int | None = None) -> SnapshotToken:
        """Reserve an aligned prefix, optionally from a recent observation index.

        An explicit past index includes immutable, already committed targets from
        the last 32 pops. Their count must fit in ``frozen_steps``; they remain
        committed and are never replayed when a prediction is merged. Explicit
        requests require fewer than ``horizon`` total overlapping rows, matching
        the RTC model's prefix contract. Omitting ``start_index`` retains the
        original queue-only snapshot behavior.
        """
        with self._lock:
            if not self._initialized:
                raise RuntimeError("queue is not initialized")
            if self._pending is not None:
                raise RuntimeError("an inference snapshot is already pending")
            if start_index is None and not len(self._targets):
                raise RuntimeError("cannot snapshot an empty queue")
            explicit_start = start_index is not None
            if start_index is None:
                start_index = self._next_index
            if (
                isinstance(start_index, bool)
                or not isinstance(start_index, int)
                or not 0 <= start_index <= self._next_index
            ):
                raise ValueError(
                    "start_index must be an integer from 0 to the next committed index"
                )
            consumed = self._next_index - start_index
            historical = [
                (index, target) for index, target in self._history if index >= start_index
            ]
            if [index for index, _ in historical] != list(range(start_index, self._next_index)):
                raise ValueError("requested prefix has a gap in the retained committed history")
            prefix = (
                np.concatenate((np.stack([target for _, target in historical]), self._targets))
                if historical
                else self._targets
            )
            if explicit_start and not 0 < len(prefix) < self.horizon:
                raise ValueError("explicit historical overlap must be between 1 and horizon - 1")
            if (
                isinstance(frozen_steps, bool)
                or not isinstance(frozen_steps, int)
                or not consumed <= frozen_steps <= len(prefix)
            ):
                raise ValueError(
                    "frozen_steps must be an integer covering committed prefix rows "
                    "and no greater than total overlap"
                )
            self._generation += 1
            token = SnapshotToken(
                start_index=start_index,
                generation=self._generation,
                previous_targets=_immutable_array(prefix),
                frozen_steps=frozen_steps,
            )
            self._pending = token
            return token

    def merge(self, token: SnapshotToken, targets: Any) -> MergeMetrics:
        """Merge one start-aligned prediction, preserving all committed/frozen rows.

        Invalid token/targets raise without changing the queue or current pending
        request. A valid but late prediction returns ``accepted=False`` and clears
        its pending request, allowing the caller to request another prediction.
        """
        values = self._validate_targets(targets)
        with self._lock:
            if self._pending is None or token is not self._pending:
                raise ValueError(
                    "token is not this queue's pending snapshot (used, foreign or canceled)"
                )
            consumed = self._next_index - token.start_index
            if consumed > token.frozen_steps:
                metrics = MergeMetrics(
                    accepted=False,
                    reason="stale_prediction",
                    generation=token.generation,
                    request_start_index=token.start_index,
                    next_index=self._next_index,
                    consumed_steps=consumed,
                    frozen_steps=token.frozen_steps,
                    prediction_end_index=token.start_index + self.horizon,
                    retained_range=(self._next_index, self._next_index + len(self._targets)),
                    replacement_range=None,
                    remaining=len(self._targets),
                    queue_end_index=self._next_index + len(self._targets),
                )
            else:
                retained = token.frozen_steps - consumed
                # Existing queued rows are authoritative until frozen_end_index.
                # Model values in that prefix need not be trusted or compared.
                self._targets = _immutable_array(
                    np.concatenate((self._targets[:retained], values[token.frozen_steps :]))
                )
                metrics = MergeMetrics(
                    accepted=True,
                    reason="merged",
                    generation=token.generation,
                    request_start_index=token.start_index,
                    next_index=self._next_index,
                    consumed_steps=consumed,
                    frozen_steps=token.frozen_steps,
                    prediction_end_index=token.start_index + self.horizon,
                    retained_range=(self._next_index, token.frozen_end_index),
                    replacement_range=(token.frozen_end_index, token.start_index + self.horizon),
                    remaining=len(self._targets),
                    queue_end_index=self._next_index + len(self._targets),
                )
            self._pending = None
            return metrics

    def cancel(self, token: SnapshotToken) -> bool:
        """Clear this pending request after inference failure, without changing targets."""
        with self._lock:
            if self._pending is token:
                self._pending = None
                return True
            return False

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self._targets)

    @property
    def next_index(self) -> int:
        with self._lock:
            return self._next_index

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    @property
    def pending_token(self) -> SnapshotToken | None:
        with self._lock:
            return self._pending
