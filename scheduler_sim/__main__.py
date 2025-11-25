from __future__ import annotations

import argparse
from collections.abc import Callable
from random import Random
from typing import Sequence

from . import evaluation, schedulers, workload
from .simulator import SimulationConfig
from .idle_minimizing_scheduler import IdleMinimizingScheduler


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline scheduler simulations.")
    parser.add_argument("--arrival-rate", type=float, default=0.8, help="Poisson arrival rate (jobs per time unit).")
    parser.add_argument("--tasks", type=int, default=32, help="Number of tasks to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for workload generation.")
    parser.add_argument(
        "--service-times",
        type=str,
        default="0.2,0.4,0.6,1.0",
        help="Comma-separated list of possible service times (time units).",
    )
    parser.add_argument(
        "--deadline-range",
        type=str,
        default="1.0,2.0",
        help="Inclusive range for deadline offsets (time units) as min,max. Use 0 to disable deadlines.",
    )
    parser.add_argument("--slice", type=float, default=0.3, help="Time slice length for the CPU (time units).")
    parser.add_argument(
        "--context-switch-cost",
        type=float,
        default=0.0,
        help="Cost of a context switch (time units) added to CPU busy time.",
    )
    parser.add_argument("--m-factor", type=float, default=1.5, help="Deadline factor for IdleMinimizingScheduler.")
    return parser.parse_args(argv)


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        msg = "service-times must contain at least one positive value"
        raise ValueError(msg)
    return values


def parse_deadline_range(raw: str) -> tuple[float, float] | None:
    parts = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not parts:
        return None
    if len(parts) != 2:
        msg = "deadline-range must be provided as min,max or left empty"
        raise ValueError(msg)
    low, high = parts
    if low < 0 or high <= 0 or high < low:
        msg = "deadline-range values must satisfy 0 <= min <= max and max > 0"
        raise ValueError(msg)
    if high == 0:
        return None
    return (low, high)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    service_times = parse_float_list(args.service_times)
    deadline_range = parse_deadline_range(args.deadline_range)

    deadline_sampler_callable: Callable[[Random], float] | None = None
    if deadline_range is not None:
        def deadline_sampler(rng: Random) -> float:
            return rng.uniform(*deadline_range)

        deadline_sampler_callable = deadline_sampler

    tasks = workload.poisson_workload(
        rate=args.arrival_rate,
        service_time_sampler=service_times,
        count=args.tasks,
        seed=args.seed,
    deadline_sampler=deadline_sampler_callable,
    )

    config = SimulationConfig(time_slice=args.slice, context_switch_cost=args.context_switch_cost)
    factories = [
        ("FCFS", schedulers.FcfsScheduler),
        ("EDF", schedulers.EdfScheduler),
        ("LRS", schedulers.LeastSlackScheduler),
        ("IdleMin", lambda: IdleMinimizingScheduler(m_factor=args.m_factor)),
    ]

    outcomes = evaluation.evaluate_suite(factories, tasks, config=config)

    print(f"Simulated {len(tasks)} tasks with arrival rate {args.arrival_rate} using slice {args.slice}\n")
    header_fmt = "{:<10} {:>9} {:>9} {:>9} {:>8} {:>9} {:>9} {:>10}"
    row_fmt = "{:<10} {:>9.3f} {:>9.3f} {:>9.3f} {:>8.3f} {:>9.3f} {:>9.3f} {:>10.3f}"
    print(header_fmt.format("Scheduler", "MeanWait", "MeanResp", "MeanSlow", "DeadMiss", "p90Wait", "p99Wait", "Throughput"))
    for outcome in outcomes:
        m = outcome.aggregate
        print(
            row_fmt.format(
                outcome.name,
                m.mean_wait_time,
                m.mean_response_time,
                m.mean_slowdown,
                m.deadline_miss_rate,
                m.p90_wait,
                m.p99_wait,
                m.throughput,
            ),
        )


if __name__ == "__main__":
    main()
