from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from random import Random

from .task import TaskSpec


@dataclass(slots=True)
class Arrival:
    at: float
    service_time: float
    deadline: float | None = None


def periodic_workload(
    period: float,
    service_time: float,
    count: int,
    *,
    deadline_offset: float | None = None,
) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for i in range(count):
        arrival = i * period
        deadline = arrival + deadline_offset if deadline_offset is not None else None
        tasks.append(TaskSpec(task_id=f"T{i}", arrival_time=arrival, service_time=service_time, deadline=deadline))
    return tasks


def poisson_workload(
    rate: float,
    service_time_sampler: Sequence[float] | Callable[[Random], float] | Iterable[float],
    count: int,
    *,
    seed: int | None = None,
    deadline_sampler: Sequence[float] | Callable[[Random], float] | Iterable[float] | None = None,
) -> list[TaskSpec]:
    rng = Random(seed)
    tasks: list[TaskSpec] = []
    current_time = 0.0
    for i in range(count):
        inter_arrival = rng.expovariate(rate)
        current_time += inter_arrival
        service_time = _sample_positive(service_time_sampler, rng)
        deadline = None
        if deadline_sampler is not None:
            offset = _sample_value(deadline_sampler, rng)
            if offset <= 0:
                msg = "deadline offset must be positive"
                raise ValueError(msg)
            deadline = current_time + offset
        tasks.append(TaskSpec(task_id=f"P{i}", arrival_time=current_time, service_time=service_time, deadline=deadline))
    return tasks


def from_arrivals(arrivals: Sequence[Arrival]) -> list[TaskSpec]:
    return [TaskSpec(task_id=f"A{idx}", arrival_time=a.at, service_time=a.service_time, deadline=a.deadline) for idx, a in enumerate(arrivals)]


def _sample_positive(source: Sequence[float] | Callable[[Random], float] | Iterable[float], rng: Random) -> float:
    value = _sample_value(source, rng)
    if value <= 0:
        msg = "sampled service time must be positive"
        raise ValueError(msg)
    return value


def _sample_value(source: Sequence[float] | Callable[[Random], float] | Iterable[float], rng: Random) -> float:
    if isinstance(source, Sequence):
        if not source:
            msg = "sampler sequence must not be empty"
            raise ValueError(msg)
        return rng.choice(source)
    if isinstance(source, Iterable):
        materialised = tuple(source)
        if not materialised:
            msg = "sampler iterable must not be empty"
            raise ValueError(msg)
        return rng.choice(materialised)
    return source(rng)
