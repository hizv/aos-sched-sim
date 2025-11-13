from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Sequence

from . import metrics
from .scheduler import Scheduler
from .simulator import Simulation, SimulationConfig, SimulationResult
from .task import TaskSpec


SchedulerFactory = Callable[[], Scheduler]


@dataclass(slots=True)
class EvaluationOutcome:
    name: str
    simulation: SimulationResult
    per_task: list[metrics.TaskMetrics]
    aggregate: metrics.AggregateMetrics


def evaluate_scheduler(
    name: str,
    factory: SchedulerFactory,
    tasks: Sequence[TaskSpec],
    *,
    config: SimulationConfig | None = None,
) -> EvaluationOutcome:
    scheduler = factory()
    simulation = Simulation(scheduler=scheduler, tasks=tasks, config=config)
    result = simulation.run()
    per_task = metrics.build_task_metrics(result.tasks)
    aggregate = metrics.summarise(per_task, result.total_time)
    return EvaluationOutcome(name=name, simulation=result, per_task=per_task, aggregate=aggregate)


def evaluate_suite(
    factories: Sequence[tuple[str, SchedulerFactory]],
    tasks: Sequence[TaskSpec],
    *,
    config: SimulationConfig | None = None,
) -> list[EvaluationOutcome]:
    return [evaluate_scheduler(name, factory, tasks, config=config) for name, factory in factories]
