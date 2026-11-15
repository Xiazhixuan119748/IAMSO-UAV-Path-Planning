"""Mirage Search Optimization planner with fitness-distance balance selection."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .encoding import clip_vector, decode, encode_from_path, smooth_path
from .fitness import CostFunction, evaluate_vector


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

    noise_scale = max(2.0, size * 0.08)
    for i in range(2, min(population_size, 8)):
        population[i] = line_vec + rng.normal(0.0, noise_scale, size=dim)
    return clip_vector(population, size)


def _fitness_distance_best(population: np.ndarray, fitness: np.ndarray, best: np.ndarray) -> int:
    """Select an individual balancing low fitness and distance from current best."""
    if len(population) == 1:
        return 0
    fit_span = float(np.ptp(fitness))
    fit_score = np.ones_like(fitness) if fit_span <= 1e-12 else (fitness.max() - fitness) / fit_span
    distances = np.linalg.norm(population - best, axis=1)
    dist_span = float(np.ptp(distances))
    dist_score = np.zeros_like(distances) if dist_span <= 1e-12 else (distances - distances.min()) / dist_span
    return int(np.argmax(0.65 * fit_score + 0.35 * dist_score))


def mso_optimize(
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
) -> dict[str, object]:
    """Optimize a continuous path using a practical MSO-inspired update scheme."""
    rng = np.random.default_rng(random_seed)
    size = int(passable.shape[0])
    dim = waypoint_count * 2
    population = _initial_population(rng, population_size, waypoint_count, size, start, goal, seed_path)
    fitness = np.array([
        evaluate_vector(ind, passable, speed_map, cost_func, start, goal).value for ind in population
    ])

    best_idx = int(np.argmin(fitness))
    best = population[best_idx].copy()
    best_eval = evaluate_vector(best, passable, speed_map, cost_func, start, goal)
    convergence = [float(best_eval.value)]

    lower = np.zeros(dim)
    upper = np.full(dim, float(size - 1))

    for iteration in range(1, iterations + 1):
        ratio = iteration / iterations
        global_count = max(1, int(round(population_size * (1.0 - ratio))))
        fdb_idx = _fitness_distance_best(population, fitness, best)
        fdb = population[fdb_idx]
        mean = population.mean(axis=0)

        order = rng.permutation(population_size)
        for rank, i in enumerate(order):
            current = population[i]
            if rank < global_count:
                # Upper mirage: broad refraction around best, FDB, and population mean.
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                wave = np.sin(2.0 * np.pi * rng.random(dim))
                candidate = (
                    current
                    + r1 * (best - np.abs(current))
                    + r2 * (fdb - mean)
                    + wave * (1.0 - ratio) * (upper - lower) * 0.12
                )
            else:
                # Lower mirage: local exploitation around elite solutions.
                peer = population[int(rng.integers(0, population_size))]
                shrink = (1.0 - ratio) ** 2
                candidate = best + rng.normal(0.0, 1.0, dim) * shrink * (np.abs(peer - current) + 1.0)

            if rng.random() < 0.15:
                candidate += rng.normal(0.0, size * 0.03 * (1.0 - ratio), dim)

            candidate = clip_vector(candidate, size)
            candidate_eval = evaluate_vector(candidate, passable, speed_map, cost_func, start, goal)
            if candidate_eval.value <= fitness[i]:
                population[i] = candidate
                fitness[i] = candidate_eval.value
                if candidate_eval.value < best_eval.value:
                    best = candidate.copy()
                    best_eval = candidate_eval

        convergence.append(float(best_eval.value))

    decoded = decode(best, start, goal)
    smooth = smooth_path(decoded)
    final_eval = evaluate_vector(best, passable, speed_map, cost_func, start, goal)
    return {
        "path": smooth,
        "fitness": float(final_eval.value),
        "collision_count": int(final_eval.collision_count),
        "convergence": convergence,
    }
