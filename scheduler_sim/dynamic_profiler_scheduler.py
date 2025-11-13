from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Sequence
import math

from .scheduler import Scheduler
from .task import TaskState


@dataclass(slots=True)
class ProfilerSnapshot:
    """Point-in-time view of aggregate system conditions."""

    utilization: float
    avg_queue_length: float
    avg_wait_time: float
    avg_service_time: float
    avg_slack: float
    system_pressure: float
    total_observed_time: float


class SystemProfiler:
    """Lightweight profiler that tracks rolling system-level statistics."""

    def __init__(self, decay: float = 0.9) -> None:
        if not 0 < decay <= 1:
            msg = "decay must be within (0, 1]"
            raise ValueError(msg)
        self._decay = decay
        self._total_time = 0.0
        self._busy_time = 0.0
        self._avg_queue = 0.0
        self._avg_wait = 0.0
        self._avg_service = 0.0
        self._avg_slack = 0.0
        self._last_timestamp = 0.0

    def advance(self, delta: float, *, running: bool, queue_length: int) -> None:
        if delta <= 0:
            return
        self._total_time += delta
        if running:
            self._busy_time += delta
        self._avg_queue = self._ema(self._avg_queue, float(queue_length))

    def record_wait(self, wait_time: float) -> None:
        if wait_time < 0:
            return
        self._avg_wait = self._ema(self._avg_wait, wait_time)

    def record_service(self, service_time: float) -> None:
        if service_time <= 0:
            return
        self._avg_service = self._ema(self._avg_service, service_time)

    def record_slack(self, slack: float) -> None:
        if not math.isfinite(slack):
            return
        self._avg_slack = self._ema(self._avg_slack, slack)

    def snapshot(self) -> ProfilerSnapshot:
        utilization = 0.0
        if self._total_time > 0:
            utilization = self._busy_time / self._total_time
        pressure = min(1.0, max(0.0, self._avg_queue / (1.0 + self._avg_queue)))
        return ProfilerSnapshot(
            utilization=utilization,
            avg_queue_length=self._avg_queue,
            avg_wait_time=self._avg_wait,
            avg_service_time=self._avg_service,
            avg_slack=self._avg_slack,
            system_pressure=pressure,
            total_observed_time=self._total_time,
        )

    @property
    def last_timestamp(self) -> float:
        return self._last_timestamp

    def reset_timestamp(self, timestamp: float) -> None:
        self._last_timestamp = timestamp

    def _ema(self, current: float, value: float) -> float:
        return self._decay * value + (1.0 - self._decay) * current


class LightweightRuntimeModel(ABC):
    """Interface for lightweight runtime predictors that can self-tune online."""

    @abstractmethod
    def predict(self, task: TaskState, snapshot: ProfilerSnapshot) -> float:
        """Predict the total service time for the task."""

    @abstractmethod
    def observe(
        self,
        task: TaskState,
        *,
        runtime: float,
        wait_time: float,
        snapshot: ProfilerSnapshot,
    ) -> None:
        """Update the model with an observed runtime sample."""


class EWMAServiceTimeModel(LightweightRuntimeModel):
    """Exponentially weighted moving-average predictor."""

    def __init__(self, alpha: float = 0.2) -> None:
        if not 0 < alpha <= 1:
            msg = "alpha must be within (0, 1]"
            raise ValueError(msg)
        self._alpha = alpha
        self._estimate: Optional[float] = None

    def predict(self, task: TaskState, snapshot: ProfilerSnapshot) -> float:
        if self._estimate is None:
            base = task.service_time
        else:
            base = self._estimate
        adjustment = 1.0 + 0.15 * snapshot.system_pressure
        return max(task.total_run_time + task.remaining_time, base * adjustment)

    def observe(
        self,
        task: TaskState,
        *,
        runtime: float,
        wait_time: float,
        snapshot: ProfilerSnapshot,
    ) -> None:
        sample = runtime
        if self._estimate is None:
            self._estimate = sample
        else:
            self._estimate = self._alpha * sample + (1.0 - self._alpha) * self._estimate


class SlidingWindowRuntimeModel(LightweightRuntimeModel):
    """Predictor that tracks a sliding window of recent runtimes."""

    def __init__(self, window: int = 32) -> None:
        if window <= 0:
            msg = "window size must be positive"
            raise ValueError(msg)
        self._window: Deque[float] = deque(maxlen=window)

    def predict(self, task: TaskState, snapshot: ProfilerSnapshot) -> float:
        if not self._window:
            return task.service_time
        avg = sum(self._window) / len(self._window)
        penalty = 1.0 + 0.1 * snapshot.utilization
        wait_boost = 1.0
        if snapshot.avg_wait_time > 0:
            wait_boost += min(0.25, snapshot.avg_wait_time / (avg + 1e-6) * 0.1)
        estimate = avg * penalty * wait_boost
        return max(task.total_run_time + task.remaining_time, estimate)

    def observe(
        self,
        task: TaskState,
        *,
        runtime: float,
        wait_time: float,
        snapshot: ProfilerSnapshot,
    ) -> None:
        self._window.append(runtime)


class DynamicRuntimeProfilerScheduler(Scheduler):
    """Adaptive scheduler using slack-aware scoring and self-tuning runtime models."""

    def __init__(
        self,
        models: Optional[Sequence[LightweightRuntimeModel]] = None,
        *,
        preemption_threshold: float = 0.1,
        min_run_time: float = 0.02,
        profiler_decay: float = 0.9,
    ) -> None:
        if preemption_threshold < 0:
            msg = "preemption_threshold cannot be negative"
            raise ValueError(msg)
        if min_run_time < 0:
            msg = "min_run_time cannot be negative"
            raise ValueError(msg)
        self._ready: Dict[str, TaskState] = {}
        self._dispatch_times: Dict[str, float] = {}
        self._started: set[str] = set()
        self._current: Optional[TaskState] = None
        self._last_timestamp = 0.0
        self._preemption_threshold = preemption_threshold
        self._min_run_time = min_run_time
        self._profiler = SystemProfiler(decay=profiler_decay)
        self._profiler.reset_timestamp(0.0)
        if models is None:
            models = (
                EWMAServiceTimeModel(alpha=0.25),
                SlidingWindowRuntimeModel(window=24),
            )
        self._models: tuple[LightweightRuntimeModel, ...] = tuple(models)

    def add_task(self, task: TaskState) -> None:
        self._ready[task.task_id] = task

    def pick_next(self, now: float, current: Optional[TaskState] = None) -> Optional[TaskState]:
        current = self._normalize_current(current)
        self._advance_time(now, running=current is not None)
        self._current = current

        if not self._ready and current is None:
            return None

        snapshot = self._profiler.snapshot()
        candidate = self._select_candidate(now, snapshot, current)
        if candidate is None:
            return current

        if current is not None and candidate is not current:
            if not self._should_preempt(current, candidate, now, snapshot):
                self._ensure_dispatch_time(current, now)
                return current

        if candidate.task_id in self._ready:
            self._ready.pop(candidate.task_id, None)
        self._ensure_dispatch_time(candidate, now)
        self._current = candidate
        return candidate

    def on_task_completed(self, task: TaskState, now: float) -> None:
        self._advance_time(now, running=self._current is task)
        self._finalize_dispatch(task)
        self._ready.pop(task.task_id, None)
        self._current = None
        wait_time = 0.0
        if task.start_time is not None:
            wait_time = max(0.0, task.start_time - task.arrival_time)
            self._profiler.record_wait(wait_time)
        self._profiler.record_service(task.total_run_time)
        slack = task.slack(now)
        self._profiler.record_slack(slack)
        snapshot = self._profiler.snapshot()
        for model in self._models:
            model.observe(
                task,
                runtime=task.total_run_time,
                wait_time=wait_time,
                snapshot=snapshot,
            )

    def on_slice_expired(self, task: TaskState, now: float) -> None:
        self._advance_time(now, running=self._current is task)
        self._finalize_dispatch(task)
        self._ready[task.task_id] = task
        self._current = None

    def _normalize_current(self, current: Optional[TaskState]) -> Optional[TaskState]:
        if current is None:
            return None
        if current.is_complete():
            return None
        return current

    def _advance_time(self, now: float, *, running: bool) -> None:
        delta = now - self._last_timestamp
        if delta <= 0:
            return
        queue_length = len(self._ready)
        self._profiler.advance(delta, running=running, queue_length=queue_length)
        self._last_timestamp = now
        self._profiler.reset_timestamp(now)

    def _select_candidate(
        self,
        now: float,
        snapshot: ProfilerSnapshot,
        current: Optional[TaskState],
    ) -> Optional[TaskState]:
        candidates: list[TaskState] = list(self._ready.values())
        if current is not None and current.task_id not in self._ready:
            candidates.append(current)
        if not candidates:
            return None
        best_score = float("inf")
        best_task: Optional[TaskState] = None
        for task in candidates:
            score = self._score(task, now, snapshot)
            if score < best_score:
                best_score = score
                best_task = task
        return best_task

    def _score(self, task: TaskState, now: float, snapshot: ProfilerSnapshot) -> float:
        wait_time = max(0.0, now - task.arrival_time)
        predicted_total = self._predict_total_runtime(task, snapshot)
        remaining_estimate = max(task.remaining_time, predicted_total - task.total_run_time)
        slack = task.slack(now)
        if math.isinf(slack):
            relative_slack = 1e6
        else:
            relative_slack = slack / max(remaining_estimate, 1e-6)
        load_penalty = snapshot.system_pressure
        wait_pressure = wait_time / max(predicted_total, 1e-6)
        score = 0.6 * relative_slack
        score -= 0.2 * math.log1p(wait_time)
        score += 0.15 * math.log1p(remaining_estimate)
        score += 0.05 * load_penalty
        score -= 0.1 * wait_pressure
        if slack < 0:
            score -= 0.5 * slack
        return score

    def _predict_total_runtime(self, task: TaskState, snapshot: ProfilerSnapshot) -> float:
        predictions: list[float] = []
        for model in self._models:
            estimate = model.predict(task, snapshot)
            if math.isfinite(estimate) and estimate > 0:
                predictions.append(estimate)
        if not predictions:
            return task.total_run_time + task.remaining_time
        blended = sum(predictions) / len(predictions)
        return max(task.total_run_time + task.remaining_time, blended)

    def _should_preempt(
        self,
        current: TaskState,
        challenger: TaskState,
        now: float,
        snapshot: ProfilerSnapshot,
    ) -> bool:
        dispatch_time = self._dispatch_times.get(current.task_id)
        if dispatch_time is not None and now - dispatch_time < self._min_run_time:
            return False
        current_score = self._score(current, now, snapshot)
        challenger_score = self._score(challenger, now, snapshot)
        return challenger_score + self._preemption_threshold < current_score

    def _ensure_dispatch_time(self, task: TaskState, now: float) -> None:
        self._dispatch_times[task.task_id] = now
        if task.task_id not in self._started:
            wait_time = max(0.0, now - task.arrival_time)
            self._profiler.record_wait(wait_time)
            self._profiler.record_slack(task.slack(now))
            self._started.add(task.task_id)

    def _finalize_dispatch(self, task: TaskState) -> None:
        self._dispatch_times.pop(task.task_id, None)
        if task.task_id in self._ready and task is not self._ready[task.task_id]:
            self._ready[task.task_id] = task
