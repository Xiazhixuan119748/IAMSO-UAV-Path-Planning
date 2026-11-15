"""Aquila Optimizer planner for continuous waypoint paths."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .encoding import clip_vector, decode, encode_from_path, smooth_path
from .fitness import CostFunction, FitnessResult, evaluate_path, evaluate_vector


def _levy(rng: np.random.Generator, dim: int, beta: float = 1.5) -> np.ndarray:
    """Mantegna Levy-flight step without scipy."""
    sigma_u = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0.0, sigma_u, dim)
    v = rng.normal(0.0, 1.0, dim)
    return u / (np.abs(v) ** (1 / beta) + 1e-12)


def _initial_population(
    rng: np.random.Generator,
    population_size: int,
    waypoint_count: int,
    size: int,
    start: Sequence[float],
    goal: Sequence[float],
    seed_path: Sequence[Sequence[float]] | None,
) -> np.ndarray:
    dim = waypoint_count * 2
    population = rng.uniform(0.0, float(size - 1), size=(population_size, dim))
    line = np.linspace(np.asarray(start, dtype=float), np.asarray(goal, dtype=float), waypoint_count + 2)[1:-1]
    line_vec = line.reshape(-1)
    population[0] = line_vec
    if seed_path:
        population[1 % population_size] = encode_from_path(seed_path, waypoint_count)
    for i in range(2, min(population_size, 8)):
        population[i] = line_vec + rng.normal(0.0, max(2.0, size * 0.08), dim)
    return clip_vector(population, size)


def _evaluation_key(result: FitnessResult) -> tuple[int, float]:
    return int(result.collision_count), float(result.value)


def _evaluate_candidate(
    vector: np.ndarray,
    passable: np.ndarray,
    speed_map: np.ndarray,
    cost_func: CostFunction,
    start: Sequence[int],
    goal: Sequence[int],
    collision_safe_smoothing: bool,
) -> FitnessResult:
    if not collision_safe_smoothing:
        return evaluate_vector(vector, passable, speed_map, cost_func, start, goal)

    decoded = decode(vector, start, goal)
    smoothed_eval = evaluate_path(smooth_path(decoded), passable, speed_map, cost_func)
    return (
        smoothed_eval
        if smoothed_eval.collision_count == 0
        else evaluate_path(decoded, passable, speed_map, cost_func)
    )


def ao_optimize(
    passable: np.ndarray,
    speed_map: np.ndarray,
    cost_func: CostFunction,
    start: Sequence[int],
    goal: Sequence[int],
    population_size: int = 30,
    iterations: int = 100,
    waypoint_count: int = 6,
    seed_path: Sequence[Sequence[float]] | None = None,
    random_seed: int | None = None,
    feasibility_priority: bool = False,
    collision_safe_smoothing: bool = False,
) -> dict[str, object]:
    """Optimize a continuous path using the four AO hunting phases."""
    rng = np.random.default_rng(random_seed)
    size = int(passable.shape[0])
    dim = waypoint_count * 2
    lower = np.zeros(dim)
    upper = np.full(dim, float(size - 1))

    population = _initial_population(rng, population_size, waypoint_count, size, start, goal, seed_path)
    evaluations = [
        _evaluate_candidate(
            ind,
            passable,
            speed_map,
            cost_func,
            start,
            goal,
            collision_safe_smoothing,
        )
        for ind in population
    ]
    fitness = np.array([result.value for result in evaluations])
    best_idx = (
        min(range(population_size), key=lambda idx: _evaluation_key(evaluations[idx]))
        if feasibility_priority
        else int(np.argmin(fitness))
    )
    best = population[best_idx].copy()
    best_eval = evaluations[best_idx]
    best_feasible = None
    for vector, result in zip(population, evaluations):
        if result.collision_count == 0 and (
            best_feasible is None or result.value < best_feasible[1].value
        ):
            best_feasible = (vector.copy(), result)
    convergence = [float(best_eval.value)]

    for iteration in range(1, iterations + 1):
        t = iteration / iterations
        mean = population.mean(axis=0)
        worst = population[int(np.argmax(fitness))]

        for i in range(population_size):
            current = population[i]
            if t <= 2.0 / 3.0:
                if rng.random() < 0.5:
                    # X1: expanded exploration from high soaring.
                    candidate = best * (1.0 - t) + (mean - best * rng.random(dim))
                else:
                    # X2: narrowed exploration with Levy flight.
                    peer = population[int(rng.integers(0, population_size))]
                    candidate = best * _levy(rng, dim) + peer + rng.random(dim) * (peer - current)
            else:
                quality = 2.0 * rng.random(dim) - 1.0
                if rng.random() < 0.5:
                    # X3: expanded exploitation around best and mean.
                    candidate = (best - mean) * quality - rng.random(dim) * ((upper - lower) * t + lower)
                else:
                    # X4: narrowed exploitation near prey.
                    levy = _levy(rng, dim)
                    candidate = best + quality * (best - current) + rng.random(dim) * levy * (best - worst)

            if rng.random() < 0.10:
                candidate += rng.normal(0.0, size * 0.02 * (1.0 - t), dim)
            candidate = clip_vector(candidate, size)
            candidate_eval = _evaluate_candidate(
                candidate,
                passable,
                speed_map,
                cost_func,
                start,
                goal,
                collision_safe_smoothing,
            )
            if candidate_eval.collision_count == 0 and (
                best_feasible is None or candidate_eval.value < best_feasible[1].value
            ):
                best_feasible = (candidate.copy(), candidate_eval)

            replaces_current = (
                _evaluation_key(candidate_eval) < _evaluation_key(evaluations[i])
                if feasibility_priority
                else candidate_eval.value <= fitness[i]
            )
            if replaces_current:
                population[i] = candidate
                evaluations[i] = candidate_eval
                fitness[i] = candidate_eval.value
                improves_best = (
                    _evaluation_key(candidate_eval) < _evaluation_key(best_eval)
                    if feasibility_priority
                    else candidate_eval.value < best_eval.value
                )
                if improves_best:
                    best = candidate.copy()
                    best_eval = candidate_eval

        convergence.append(float(best_eval.value))

    if feasibility_priority and best_feasible is not None:
        best, best_eval = best_feasible
    return {
        "path": best_eval.path,
        "fitness": float(best_eval.value),
        "collision_count": int(best_eval.collision_count),
        "convergence": convergence,
    }
