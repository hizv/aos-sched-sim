from __future__ import annotations

from abc import ABC, abstractmethod

from .task import TaskState


class Scheduler(ABC):
    """Abstract scheduler interface for selecting the next task to run."""

    @abstractmethod
    def add_task(self, task: TaskState) -> None:
        """Add a newly arrived task to the ready queue."""

    @abstractmethod
    def pick_next(self, now: float, current: TaskState | None = None) -> TaskState | None:
        """Select the next task to run at the given time."""

    def on_task_completed(self, task: TaskState, now: float) -> None:
        """Hook invoked when a task completes."""

    def on_task_preempted(self, task: TaskState, now: float) -> None:
        """Hook invoked when a running task is preempted."""

    def on_slice_expired(self, task: TaskState, now: float) -> None:
        """Hook invoked when a time slice expires for a running task."""
