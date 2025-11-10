from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence

from .task import TaskState


@dataclass(slots=True)
class TaskMetrics:
    task_id: str
    arrival_time: float
    start_time: float
    finish_time: float
    service_time: float
    wait_time: float
    response_time: float
    first_response_latency: float
    slowdown: float
    deadline: float | None
    deadline_met: bool | None


@dataclass(slots=True)
class AggregateMetrics:
    count: int
    mean_wait_time: float
    mean_response_time: float
    mean_slowdown: float
    deadline_miss_rate: float
    p50_wait: float
    p90_wait: float
    p99_wait: float
    p50_response: float
    p90_response: float
    p99_response: float
    throughput: float


def build_task_metrics(tasks: Iterable[TaskState]) -> list[TaskMetrics]:
    metrics: list[TaskMetrics] = []
    for task in tasks:
        if task.finish_time is None or task.start_time is None or task.first_response_time is None:
            continue
        wait_time = task.start_time - task.arrival_time
        response_time = task.finish_time - task.arrival_time
        first_latency = task.first_response_time - task.arrival_time
        slowdown = response_time / task.service_time if task.service_time else float("inf")
        deadline = task.deadline
        deadline_met: bool | None = None
        if deadline is not None:
            deadline_met = task.finish_time <= deadline
        metrics.append(
            TaskMetrics(
                task_id=task.task_id,
                arrival_time=task.arrival_time,
                start_time=task.start_time,
                finish_time=task.finish_time,
                service_time=task.service_time,
                wait_time=wait_time,
                response_time=response_time,
                first_response_latency=first_latency,
                slowdown=slowdown,
                deadline=deadline,
                deadline_met=deadline_met,
            ),
        )
    return metrics


def summarise(metrics: Sequence[TaskMetrics], total_time: float) -> AggregateMetrics:
    if not metrics:
        return AggregateMetrics(
            count=0,
            mean_wait_time=0.0,
            mean_response_time=0.0,
            mean_slowdown=0.0,
            deadline_miss_rate=0.0,
            p50_wait=0.0,
            p90_wait=0.0,
            p99_wait=0.0,
            p50_response=0.0,
            p90_response=0.0,
            p99_response=0.0,
            throughput=0.0,
        )
    wait_values = [m.wait_time for m in metrics]
    response_values = [m.response_time for m in metrics]
    slowdowns = [m.slowdown for m in metrics]
    deadline_miss_rate = _deadline_miss_rate(metrics)
    return AggregateMetrics(
        count=len(metrics),
        mean_wait_time=mean(wait_values),
        mean_response_time=mean(response_values),
        mean_slowdown=mean(slowdowns),
        deadline_miss_rate=deadline_miss_rate,
        p50_wait=_percentile(wait_values, 50),
        p90_wait=_percentile(wait_values, 90),
        p99_wait=_percentile(wait_values, 99),
        p50_response=_percentile(response_values, 50),
        p90_response=_percentile(response_values, 90),
        p99_response=_percentile(response_values, 99),
        throughput=len(metrics) / total_time if total_time else 0.0,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * percentile / 100
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def _deadline_miss_rate(metrics: Sequence[TaskMetrics]) -> float:
    considered = [m for m in metrics if m.deadline is not None]
    if not considered:
        return 0.0
    misses = sum(0 if m.deadline_met else 1 for m in considered)
    return misses / len(considered)
