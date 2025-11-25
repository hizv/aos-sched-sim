from __future__ import annotations

from .scheduler import Scheduler
from .task import TaskState


class IdleMinimizingScheduler(Scheduler):
    """
    Preemptive scheduler that prioritizes tasks to minimize idle time against slack.

    Priority is defined as:
        pi = slack - idle_time
           = (m-1) * service_time - (now - arrival_time - total_run_time)

    A lower priority value means the task is more urgent.
    """

    def __init__(self, m_factor: float = 1.5) -> None:
        if m_factor < 1.0:
            msg = "m_factor must be >= 1.0"
            raise ValueError(msg)
        self._m = m_factor
        self._ready: list[TaskState] = []

    def add_task(self, task: TaskState) -> None:
        if task not in self._ready:
            self._ready.append(task)

    def pick_next(self, now: float, current: TaskState | None = None) -> TaskState | None:
        if current and not current.is_complete() and current not in self._ready:
            self._ready.append(current)

        if not self._ready:
            return None

        # Find the task with the minimum priority
        best_task = min(self._ready, key=lambda task: self._priority(task, now))

        self._ready.remove(best_task)
        return best_task

    def on_task_preempted(self, task: TaskState, now: float) -> None:
        if not task.is_complete() and task not in self._ready:
            self._ready.append(task)

    def on_slice_expired(self, task: TaskState, now: float) -> None:
        if not task.is_complete() and task not in self._ready:
            self._ready.append(task)

    def on_task_completed(self, task: TaskState, now: float) -> None:
        if task in self._ready:
            self._ready.remove(task)

    def _priority(self, task: TaskState, now: float) -> float:
        """Calculates the priority for a task."""
        slack = (self._m - 1.0) * task.service_time
        idle_time = (now - task.arrival_time) - task.total_run_time
        return slack - idle_time