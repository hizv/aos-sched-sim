from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .scheduler import Scheduler
from .task import TaskState


@dataclass(order=True, slots=True)
class _HeapItem:
    priority: float
    seq: int
    task: TaskState = field(compare=False)


class FcfsScheduler(Scheduler):
    """Non-preemptive First-Come, First-Served scheduler."""

    def __init__(self) -> None:
        self._queue: deque[TaskState] = deque()
        self._seq = 0

    def add_task(self, task: TaskState) -> None:
        self._queue.append(task)

    def pick_next(self, now: float, current: TaskState | None = None) -> TaskState | None:
        if current is not None and not current.is_complete():
            return current
        if not self._queue:
            return None
        return self._queue.popleft()


class EdfScheduler(Scheduler):
    """Preemptive Earliest Deadline First scheduler."""

    def __init__(self) -> None:
        self._heap: list[_HeapItem] = []
        self._seq = 0

    def add_task(self, task: TaskState) -> None:
        deadline = task.deadline if task.deadline is not None else float("inf")
        heapq.heappush(self._heap, _HeapItem(deadline, self._seq, task))
        self._seq += 1

    def pick_next(self, now: float, current: TaskState | None = None) -> TaskState | None:
        if current is not None and not current.is_complete():
            current_deadline = current.deadline if current.deadline is not None else float("inf")
            top_deadline = self._heap[0].priority if self._heap else float("inf")
            if current_deadline <= top_deadline:
                return current
            self._push(current_deadline, current)
        if not self._heap:
            return None
        return heapq.heappop(self._heap).task

    def on_task_preempted(self, task: TaskState, now: float) -> None:
        self._push(task.deadline if task.deadline is not None else float("inf"), task)

    def _push(self, priority: float, task: TaskState) -> None:
        heapq.heappush(self._heap, _HeapItem(priority, self._seq, task))
        self._seq += 1


class LeastSlackScheduler(Scheduler):
    """Preemptive scheduler that prioritises least remaining slack time."""

    def __init__(self, clamp_fn: Callable[[float], float] | None = None) -> None:
        self._ready: list[TaskState] = []
        self._clamp_fn = clamp_fn if clamp_fn is not None else lambda slack: slack

    def add_task(self, task: TaskState) -> None:
        self._ensure_ready(task)

    def pick_next(self, now: float, current: TaskState | None = None) -> TaskState | None:
        self._discard_if_present(current)
        candidate = self._best_task(now)
        if candidate is None:
            if current is not None and not current.is_complete():
                return current
            return None

        if current is not None and not current.is_complete():
            current_priority = self._priority(current, now)
            candidate_priority = self._priority(candidate, now)
            if current_priority <= candidate_priority:
                return current
            self._ensure_ready(current)

        self._ready.remove(candidate)
        return candidate

    def on_task_preempted(self, task: TaskState, now: float) -> None:
        self._ensure_ready(task)

    def on_slice_expired(self, task: TaskState, now: float) -> None:
        self._ensure_ready(task)

    def _best_task(self, now: float) -> TaskState | None:
        if not self._ready:
            return None
        return min(self._ready, key=lambda task: self._priority(task, now))

    def _priority(self, task: TaskState, now: float) -> float:
        slack = task.slack(now)
        return self._clamp_fn(slack)

    def _ensure_ready(self, task: TaskState) -> None:
        if task.is_complete():
            return
        if task not in self._ready:
            self._ready.append(task)

    def _discard_if_present(self, task: TaskState | None) -> None:
        if task is None:
            return
        try:
            self._ready.remove(task)
        except ValueError:
            pass
