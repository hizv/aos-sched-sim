from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Immutable task specification for the simulation."""

    task_id: str
    arrival_time: float
    service_time: float
    deadline: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.service_time <= 0:
            msg = "service_time must be strictly positive"
            raise ValueError(msg)
        if self.arrival_time < 0:
            msg = "arrival_time cannot be negative"
            raise ValueError(msg)
        if self.deadline is not None and self.deadline <= self.arrival_time:
            msg = "deadline must be greater than arrival_time"
            raise ValueError(msg)


@dataclass(slots=True)
class TaskState:
    """Mutable runtime state for a task."""

    spec: TaskSpec
    remaining_time: float
    start_time: Optional[float] = None
    first_response_time: Optional[float] = None
    finish_time: Optional[float] = None
    total_run_time: float = 0.0

    def mark_started(self, now: float) -> None:
        if self.start_time is None:
            self.start_time = now
        if self.first_response_time is None:
            self.first_response_time = now

    def record_run(self, delta: float) -> None:
        self.remaining_time -= delta
        self.total_run_time += delta

    def is_complete(self, *, epsilon: float = 1e-9) -> bool:
        return self.remaining_time <= epsilon

    @property
    def task_id(self) -> str:
        return self.spec.task_id

    @property
    def arrival_time(self) -> float:
        return self.spec.arrival_time

    @property
    def service_time(self) -> float:
        return self.spec.service_time

    @property
    def deadline(self) -> Optional[float]:
        return self.spec.deadline

    def slack(self, now: float) -> float:
        """Compute remaining slack time, falling back to +inf when undefined."""

        if self.deadline is None:
            return float("inf")
        return self.deadline - now - self.remaining_time
