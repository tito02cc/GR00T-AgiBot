"""Background RTC prediction with main-thread queue publication.

No GDK/socket-to-robot code lives here. A future live executor must associate a
source observation with the queue's next uncommitted step, and account for any
commands already buffered by GDK. Queue indices are NOT hardware timestamps.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import queue
import threading
import time
from typing import Callable

from agibot.rtc.action_queue import ActionQueue
from agibot.rtc.adapter import targets_to_prefix
from agibot.rtc.contract import validate_receipt
from agibot.tools.g2_gr00t_shadow_adapter import decode_action_chunk
import numpy as np


class AsyncRtcPlanner:
    """One inference worker; only poll() may merge future actions.

    ``policy_factory`` runs INSIDE the worker so a ZMQ PolicyClient's socket is
    never shared across threads. It must return get_action()/optional close().
    Late results are rejected by consumed-step count, never replayed or retimed.
    """

    def __init__(self, actions: ActionQueue, policy_factory: Callable):
        if actions.horizon != 16:
            raise ValueError("this experiment supports only the trained H16")
        self.actions = actions
        self._factory = policy_factory
        self._jobs = queue.Queue(maxsize=1)
        self._results = queue.Queue()
        self._stop = threading.Event()
        self._closed = False
        self._pending = None
        self._lock = threading.RLock()
        self._thread = threading.Thread(target=self._work, name="groot-rtc-prediction", daemon=True)
        self._thread.start()

    def _work(self):
        policy = None
        try:
            while not self._stop.is_set():
                try:
                    token, observation, options, submitted = self._jobs.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self._stop.is_set():
                    break
                try:
                    if policy is None:
                        policy = self._factory()
                    start = time.monotonic()
                    action, info = policy.get_action(observation, options=options)
                    validate_receipt(
                        info,
                        overlap=token.overlap_steps,
                        frozen=token.frozen_steps,
                        ramp_rate=options["rtc"]["ramp_rate"],
                    )
                    targets = decode_action_chunk(action)
                    result = {
                        "targets": targets,
                        "model_info": info,
                        "inference_s": time.monotonic() - start,
                        "request_elapsed_s": time.monotonic() - submitted,
                    }
                except Exception as error:
                    result = {"error": error}
                self._results.put((token, result))
        finally:
            if policy is not None and hasattr(policy, "close"):
                policy.close()

    def request(
        self,
        observation: dict,
        *,
        observation_step_index: int,
        frozen_steps: int = 4,
        ramp_rate: float = 3.0,
    ) -> dict:
        """Submit an already time-aligned observation, not a blocking camera call."""
        with self._lock:
            if self._closed:
                raise RuntimeError("planner is closed")
            if self._pending is not None:
                raise RuntimeError("RTC request already pending")
            if (
                isinstance(frozen_steps, bool)
                or not isinstance(frozen_steps, int)
                or frozen_steps <= 0
            ):
                raise ValueError("frozen_steps must be a positive integer")
            if not np.isfinite(ramp_rate) or ramp_rate <= 0:
                raise ValueError("ramp_rate must be finite and positive")
            # The observation may have been captured before recently submitted
            # rows. Those exact rows are reconstructed from queue history and
            # counted as already committed inside the frozen prefix.
            if isinstance(observation_step_index, bool) or not isinstance(
                observation_step_index, int
            ):
                raise ValueError("observation step must be an integer")
            token = self.actions.snapshot(frozen_steps, start_index=observation_step_index)
            try:
                if (
                    isinstance(observation_step_index, bool)
                    or not isinstance(observation_step_index, int)
                    or token.start_index != observation_step_index
                ):
                    raise ValueError("observation step does not match queued prefix start")
                options = {
                    "rtc": {
                        "previous_actions": targets_to_prefix(token.previous_targets),
                        "frozen_steps": frozen_steps,
                        "ramp_rate": float(ramp_rate),
                    }
                }
                # Snapshot copied before handing ownership to the worker.
                observation = copy.deepcopy(observation)
                self._pending = token
                self._jobs.put_nowait((token, observation, options, time.monotonic()))
            except Exception:
                self.actions.cancel(token)
                self._pending = None
                raise
            return {
                "start_index": token.start_index,
                "overlap_steps": token.overlap_steps,
                "frozen_steps": frozen_steps,
                "generation": token.generation,
            }

    def poll(self) -> dict | None:
        with self._lock:
            if self._closed:
                return None
            try:
                token, result = self._results.get_nowait()
            except queue.Empty:
                return None
            if token is not self._pending:
                raise RuntimeError("unexpected RTC result identity")
            try:
                if "error" in result:
                    raise RuntimeError(f"RTC prediction failed: {result['error']}") from result[
                        "error"
                    ]
                metrics = self.actions.merge(token, result.pop("targets"))
                return {**result, "merge": asdict(metrics)}
            finally:
                self.actions.cancel(token)
                self._pending = None

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    def close(self, timeout_s: float = 1.0) -> bool:
        """Invalidate pending work immediately; a running model cannot move anything.

        Return whether the worker exited within the requested timeout. This does
        not pretend to cancel an in-flight CUDA/ZMQ operation. No result from a
        closed planner can ever be merged, even if it eventually finishes.
        """
        if not np.isfinite(timeout_s) or not 0 <= timeout_s <= 60:
            raise ValueError("close timeout must be in [0, 60] seconds")
        with self._lock:
            self._closed = True
            self._stop.set()
            if self._pending is not None:
                self.actions.cancel(self._pending)
                self._pending = None
        self._thread.join(timeout_s)
        return not self._thread.is_alive()
