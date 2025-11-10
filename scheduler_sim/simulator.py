from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .scheduler import Scheduler
from .task import TaskSpec, TaskState


@dataclass(slots=True)
class SimulationResult:
    tasks: list[TaskState]
    total_time: float
    cpu_busy_time: float
    context_switches: int

    @property
    def utilization(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.cpu_busy_time / self.total_time


@dataclass(slots=True)
class SimulationConfig:
    time_slice: float = 1.0
    context_switch_cost: float = 0.0
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.time_slice <= 0:
            msg = "time_slice must be strictly positive"
            raise ValueError(msg)
        if self.context_switch_cost < 0:
            msg = "context_switch_cost cannot be negative"
            raise ValueError(msg)


class Simulation:
    """Discrete-time simulation for testing scheduler policies."""

    def __init__(self, scheduler: Scheduler, tasks: Sequence[TaskSpec], config: SimulationConfig | None = None) -> None:
        self.scheduler = scheduler
        self.config = config or SimulationConfig()
        self._tasks = sorted(tasks, key=lambda t: (t.arrival_time, t.task_id))
        self._pending: list[TaskSpec] = list(self._tasks)
        self._completed: list[TaskState] = []
        self._active: list[TaskState] = []
        self._now: float = 0.0
        self._context_switches = 0
        self._cpu_busy = 0.0

    def run(self) -> SimulationResult:
        current: TaskState | None = None
        while self._pending or current is not None or self._active:
            self._enqueue_arrivals(self._now)
            next_task = self.scheduler.pick_next(self._now, current)

            if next_task is None:
                if current is not None and not current.is_complete(epsilon=self.config.epsilon):
                    self.scheduler.on_task_preempted(current, self._now)
                    current = None
                idle_until = self._next_arrival_time()
                if idle_until is None:
                    break
                self._now = max(self._now, idle_until)
                continue

            previous = current
            if next_task is not previous:
                if previous is not None and not previous.is_complete(epsilon=self.config.epsilon):
                    self.scheduler.on_task_preempted(previous, self._now)
                self._apply_context_switch()
                current = next_task
            else:
                current = next_task

            if current is not None and current.start_time is None:
                current.mark_started(self._now)

            if current is None:
                continue

            slice_remaining = self.config.time_slice
            next_arrival = self._next_arrival_time()
            service_remaining = current.remaining_time
            run_limit = min(slice_remaining, service_remaining)
            target_time = self._now + run_limit
            if next_arrival is not None and next_arrival < target_time:
                target_time = next_arrival
            run_time = target_time - self._now

            if run_time <= 0:
                self._now = target_time
                continue

            current.record_run(run_time)
            self._cpu_busy += run_time
            self._now = target_time

            if current.is_complete(epsilon=self.config.epsilon):
                current.finish_time = self._now
                self.scheduler.on_task_completed(current, self._now)
                self._completed.append(current)
                try:
                    self._active.remove(current)
                except ValueError:
                    pass
                current = None
            else:
                if next_arrival is not None and abs(next_arrival - self._now) <= self.config.epsilon:
                    continue
                self.scheduler.on_slice_expired(current, self._now)

        total_time = self._now
        return SimulationResult(self._completed, total_time, self._cpu_busy, self._context_switches)

    def _enqueue_arrivals(self, now: float) -> None:
        while self._pending and self._pending[0].arrival_time <= now + self.config.epsilon:
            spec = self._pending.pop(0)
            state = TaskState(spec=spec, remaining_time=spec.service_time)
            self.scheduler.add_task(state)
            self._active.append(state)

    def _next_arrival_time(self) -> float | None:
        if not self._pending:
            return None
        return self._pending[0].arrival_time

    def _apply_context_switch(self) -> None:
        if self.config.context_switch_cost:
            self._now += self.config.context_switch_cost
            self._cpu_busy += self.config.context_switch_cost
        self._context_switches += 1

    @property
    def tasks(self) -> Iterable[TaskState]:
        return list(self._completed)
