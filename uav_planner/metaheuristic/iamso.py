"""Improved Adaptive Mirage Search Optimization planner."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from uav_planner.config import EXPOSURE_PATH_WEIGHT, MODEL_WEIGHTS
from uav_planner.costmap import C_cruise, C_inf, C_rain, C_wind, heading_vector, is_layer_map
from uav_planner.geometry import collision_cells_for_path, collision_segment_count_for_path
from uav_planner.metrics import calculate_exposure_metrics

from .encoding import clip_vector, decode, encode_from_path, polyline_length, sample_path, smooth_path
from .feasibility import FeasibleArchive, feasibility_key
from .fitness import CostFunction


def _normalize_weights(raw_weights: np.ndarray) -> tuple[float, float, float, float]:
    total = float(raw_weights.sum())
    if total <= 1e-12:
        return MODEL_WEIGHTS
    normalized = raw_weights / total
    return tuple(float(value) for value in normalized)


def adaptive_weights(layers, passable: np.ndarray, base_weights=MODEL_WEIGHTS) -> tuple[float, float, float, float]:
    """Adjust layer weights from map statistics without changing global defaults."""
    base = np.asarray(base_weights, dtype=float)
    if not is_layer_map(layers):
        return _normalize_weights(base)

    blocked_ratio = 1.0 - float(passable.sum() / passable.size)
    mean_wind = float(np.linalg.norm(layers["W"], axis=2).mean())
    mean_rain = float(layers["R"].mean())
    mean_cruise = float(layers["V"].mean())
    raw = base * np.array(
        [
            1.0 + blocked_ratio,
            1.0 + mean_wind,
            1.0 + mean_rain,
            1.0 + mean_cruise,
        ],
        dtype=float,
    )
    return _normalize_weights(raw)


def _turn_penalty(path: Sequence[Sequence[float]]) -> float:
    penalty = 0.0
    pts = np.asarray(path, dtype=float)
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-9 or n2 <= 1e-9:
            continue
        cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        penalty += math.acos(cosine) ** 2
    return penalty


def _nearest_cell(point: Sequence[float], size: int) -> tuple[int, int] | None:
    row = int(round(float(point[0])))
    col = int(round(float(point[1])))
    if row < 0 or col < 0 or row >= size or col >= size:
        return None
    return row, col


def _evaluate_path_adaptive(
    path,
    passable: np.ndarray,
    layers,
    cost_func: CostFunction,
    weights: tuple[float, float, float, float],
    collision_weight: float,
    smooth_weight: float,
    length_weight: float,
) -> dict[str, float]:
    # IAMSO evaluates the smoothed continuous path directly rather than the raw waypoint vector.
    size = int(passable.shape[0])
    samples = sample_path(path, step=0.5)
    collisions = set(collision_cells_for_path(passable, path))
    segment_collision_count = collision_segment_count_for_path(passable, path)
    layer_cost = 0.0

    prev = None
    for point in samples:
        cell = _nearest_cell(point, size)
        if cell is None:
            collisions.add(("out", int(round(float(point[0]))), int(round(float(point[1])))))
            prev = point
            continue

        row, col = cell
        step_distance = 0.0 if prev is None else math.dist(prev, point)
        if is_layer_map(layers):
            q = (row, col)
            heading = (0.0, 0.0) if prev is None else heading_vector(prev, point)
            w_inf, w_wind, w_rain, w_cruise = weights
            local_cost = (
                w_inf * C_inf(q, layers)
                + w_wind * C_wind(q, heading, layers)
                + w_rain * C_rain(q, layers)
                + w_cruise * C_cruise(q, layers)
            )
        else:
            local_cost = float(cost_func(row, col, layers))
        layer_cost += step_distance * local_cost
        prev = point

    length = polyline_length(path)
    smoothness = _turn_penalty(path)
    collision_count = len(collisions)
    exposure = calculate_exposure_metrics(path, layers)["total_exposure"] if is_layer_map(layers) else 0.0
    value = (
        layer_cost
        + length_weight * length
        + collision_weight * segment_collision_count
        + smooth_weight * smoothness
        + EXPOSURE_PATH_WEIGHT * exposure
    )
    return {
        "value": float(value),
        "collision_count": float(collision_count),
        "segment_collision_count": float(segment_collision_count),
        "length": float(length),
        "smoothness": float(smoothness),
        "layer_cost": float(layer_cost),
        "exposure": float(exposure),
    }


def _evaluate_vector_adaptive(
    vector: np.ndarray,
    passable: np.ndarray,
    layers,
    cost_func: CostFunction,
    start: Sequence[float],
    goal: Sequence[float],
    weights: tuple[float, float, float, float],
    collision_weight: float,
    smooth_weight: float,
    length_weight: float,
) -> dict[str, object]:
    decoded = decode(vector, start, goal)
    path = smooth_path(decoded)
    metrics = _evaluate_path_adaptive(
        path,
        passable,
        layers,
        cost_func,
        weights,
        collision_weight,
        smooth_weight,
        length_weight,
    )
    metrics["path"] = path
    return metrics


def _line_seed(start: Sequence[float], goal: Sequence[float], waypoint_count: int) -> np.ndarray:
    line = np.linspace(np.asarray(start, dtype=float), np.asarray(goal, dtype=float), waypoint_count + 2)[1:-1]
    return line.reshape(-1)


def _initial_population_multi_source(
    rng: np.random.Generator,
    population_size: int,
    waypoint_count: int,
    size: int,
    start: Sequence[float],
    goal: Sequence[float],
    seed_paths: Sequence[Sequence[Sequence[float]]] | None,
) -> np.ndarray:
    # The initial population mixes one straight-line seed, external algorithm seeds,
    # seed-neighborhood perturbations, and fully random individuals.
    dim = waypoint_count * 2
    population = rng.uniform(0.0, float(size - 1), size=(population_size, dim))

    line_vec = _line_seed(start, goal, waypoint_count)
    population[0] = line_vec

    write_index = 1
    encoded_seeds: list[np.ndarray] = []
    for path in seed_paths or []:
        if not path:
            continue
        try:
            encoded_seeds.append(encode_from_path(path, waypoint_count))
        except ValueError:
            continue

    for seed_vec in encoded_seeds:
        if write_index >= population_size:
            break
        population[write_index] = seed_vec
        write_index += 1

    noise_sources = encoded_seeds or [line_vec]
    noise_scale = max(2.0, size * 0.08)
    while write_index < min(population_size, 12):
        source = noise_sources[write_index % len(noise_sources)]
        population[write_index] = source + rng.normal(0.0, noise_scale, size=dim)
        write_index += 1

    return clip_vector(population, size)


def _fitness_distance_best(population: np.ndarray, fitness: np.ndarray, best: np.ndarray) -> int:
    if len(population) == 1:
        return 0
    fit_span = float(np.ptp(fitness))
    fit_score = np.ones_like(fitness) if fit_span <= 1e-12 else (fitness.max() - fitness) / fit_span
    distances = np.linalg.norm(population - best, axis=1)
    dist_span = float(np.ptp(distances))
    dist_score = np.zeros_like(distances) if dist_span <= 1e-12 else (distances - distances.min()) / dist_span
    return int(np.argmax(0.65 * fit_score + 0.35 * dist_score))


def _adaptive_factor(convergence: list[float]) -> float:
    if len(convergence) < 8:
        return 1.0
    recent = convergence[-1]
    previous = convergence[-7]
    improvement = previous - recent
    if improvement <= max(1e-6, abs(previous) * 1e-4):
        return 1.5
    return 0.75


def _superior_candidate(
    rng: np.random.Generator,
    current: np.ndarray,
    population: np.ndarray,
    best: np.ndarray,
    fdb: np.ndarray,
    mean: np.ndarray,
    ratio: float,
    adaptive_factor: float,
    upper: np.ndarray,
    lower: np.ndarray,
    size: int,
) -> np.ndarray:
    """Build a superior-phase candidate focused on broad search and guided exploration."""
    dim = len(current)
    if ratio <= 0.4:
        r1 = rng.random(dim)
        r2 = rng.random(dim)
        wave = np.sin(2.0 * np.pi * rng.random(dim))
        return (
            current
            + r1 * (best - np.abs(current))
            + r2 * (fdb - mean)
            + wave * adaptive_factor * (1.0 - ratio) * (upper - lower) * 0.14
        )

    peer = population[int(rng.integers(0, len(population)))]
    r1 = rng.random(dim)
    r2 = rng.random(dim)
    candidate = current + r1 * (best - current) + r2 * 0.5 * (peer - current)
    candidate += rng.normal(0.0, size * 0.02 * adaptive_factor * max(0.1, 1.0 - ratio), dim)
    if rng.random() < 0.12:
        candidate += rng.normal(0.0, size * 0.025 * adaptive_factor * max(0.05, 1.0 - ratio), dim)
    return candidate


def _inferior_candidate(
    rng: np.random.Generator,
    current: np.ndarray,
    population: np.ndarray,
    best: np.ndarray,
    ratio: float,
    adaptive_factor: float,
) -> np.ndarray:
    """Build an inferior-phase candidate focused on local refinement near elite regions."""
    dim = len(current)
    peer = population[int(rng.integers(0, len(population)))]
    pull = rng.random(dim) * (0.30 + 0.70 * ratio)
    shrink = max(0.01, (1.0 - ratio) ** 2)
    local_scale = adaptive_factor * shrink * (0.15 + 0.85 * ratio)
    candidate = current + pull * (best - current)
    candidate += rng.normal(0.0, 1.0, dim) * local_scale * (np.abs(peer - current) + 1.0)
    if rng.random() < 0.08:
        candidate += rng.normal(0.0, 1.0, dim) * 0.5 * local_scale
    return candidate


def _accept_candidate(
    candidate: np.ndarray,
    index: int,
    population: np.ndarray,
    fitness: np.ndarray,
    evaluations: list[dict[str, object]],
    passable: np.ndarray,
    layers,
    cost_func: CostFunction,
    start: Sequence[float],
    goal: Sequence[float],
    weights: tuple[float, float, float, float],
    collision_weight: float,
    smooth_weight: float,
    length_weight: float,
    size: int,
    best: np.ndarray,
    best_eval: dict[str, object],
    feasible_archive: FeasibleArchive,
) -> tuple[np.ndarray, dict[str, object]]:
    """Evaluate and accept a candidate using strict feasibility-first ordering."""
    candidate = clip_vector(candidate, size)
    candidate_eval = _evaluate_vector_adaptive(
        candidate,
        passable,
        layers,
        cost_func,
        start,
        goal,
        weights,
        collision_weight,
        smooth_weight,
        length_weight,
    )
    feasible_archive.consider(candidate_eval, vector=candidate, source="optimized")
    if feasibility_key(candidate_eval) <= feasibility_key(evaluations[index]):
        population[index] = candidate
        fitness[index] = float(candidate_eval["value"])
        evaluations[index] = candidate_eval
        if feasibility_key(candidate_eval) < feasibility_key(best_eval):
            best = candidate.copy()
            best_eval = candidate_eval
    return best, best_eval


def _run_superior_phase(
    rng: np.random.Generator,
    population: np.ndarray,
    fitness: np.ndarray,
    evaluations: list[dict[str, object]],
    passable: np.ndarray,
    layers,
    cost_func: CostFunction,
    start: Sequence[float],
    goal: Sequence[float],
    weights: tuple[float, float, float, float],
    collision_weight: float,
    smooth_weight: float,
    length_weight: float,
    size: int,
    lower: np.ndarray,
    upper: np.ndarray,
    best: np.ndarray,
    best_eval: dict[str, object],
    feasible_archive: FeasibleArchive,
    ratio: float,
    adaptive_factor: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run the superior mirage phase."""
    fdb = population[_fitness_distance_best(population, fitness, best)]
    mean = population.mean(axis=0)
    for index in rng.permutation(len(population)):
        current = population[index]
        candidate = _superior_candidate(
            rng,
            current,
            population,
            best,
            fdb,
            mean,
            ratio,
            adaptive_factor,
            upper,
            lower,
            size,
        )
        best, best_eval = _accept_candidate(
            candidate,
            index,
            population,
            fitness,
            evaluations,
            passable,
            layers,
            cost_func,
            start,
            goal,
            weights,
            collision_weight,
            smooth_weight,
            length_weight,
            size,
            best,
            best_eval,
            feasible_archive,
        )
    return best, best_eval


def _run_inferior_phase(
    rng: np.random.Generator,
    population: np.ndarray,
    fitness: np.ndarray,
    evaluations: list[dict[str, object]],
    passable: np.ndarray,
    layers,
    cost_func: CostFunction,
    start: Sequence[float],
    goal: Sequence[float],
    weights: tuple[float, float, float, float],
    collision_weight: float,
    smooth_weight: float,
    length_weight: float,
    size: int,
    best: np.ndarray,
    best_eval: dict[str, object],
    feasible_archive: FeasibleArchive,
    ratio: float,
    adaptive_factor: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run the inferior mirage phase after early exploration burn-in."""
    if ratio < 0.15:
        return best, best_eval

    for index in rng.permutation(len(population)):
        current = population[index]
        candidate = _inferior_candidate(
            rng,
            current,
            population,
            best,
            ratio,
            adaptive_factor,
        )
        best, best_eval = _accept_candidate(
            candidate,
            index,
            population,
            fitness,
            evaluations,
            passable,
            layers,
            cost_func,
            start,
            goal,
            weights,
            collision_weight,
            smooth_weight,
            length_weight,
            size,
            best,
            best_eval,
            feasible_archive,
        )
    return best, best_eval


def iamso_optimize(
    passable: np.ndarray,
    layers,
    cost_func: CostFunction,
    start: Sequence[int],
    goal: Sequence[int],
    population_size: int = 24,
    iterations: int = 80,
    waypoint_count: int = 6,
    seed_paths: Sequence[Sequence[Sequence[float]]] | None = None,
    fallback_path: Sequence[Sequence[float]] | None = None,
    base_weights=MODEL_WEIGHTS,
    random_seed: int | None = None,
    collision_weight: float = 1000.0,
    smooth_weight: float = 2.0,
    length_weight: float = 1.0,
    initial_population_builder=None,
) -> dict[str, object]:
    """Optimize a path with multi-source initialization and explicit superior/inferior phases."""
    rng = np.random.default_rng(random_seed)
    size = int(passable.shape[0])
    dim = waypoint_count * 2
    weights = adaptive_weights(layers, passable, base_weights)
    if initial_population_builder is None:
        population = _initial_population_multi_source(
            rng,
            population_size,
            waypoint_count,
            size,
            start,
            goal,
            seed_paths,
        )
    else:
        population = initial_population_builder(
            rng,
            population_size,
            waypoint_count,
            size,
            start,
            goal,
            seed_paths,
            passable=passable,
        )
    evaluations = [
        _evaluate_vector_adaptive(
            ind,
            passable,
            layers,
            cost_func,
            start,
            goal,
            weights,
            collision_weight,
            smooth_weight,
            length_weight,
        )
        for ind in population
    ]
    fitness = np.array([float(item["value"]) for item in evaluations])

    best_idx = min(range(len(evaluations)), key=lambda index: feasibility_key(evaluations[index]))
    best = population[best_idx].copy()
    best_eval = evaluations[best_idx]
    feasible_archive = FeasibleArchive()
    for vector, evaluation in zip(population, evaluations):
        feasible_archive.consider(evaluation, vector=vector, source="initial_population")

    if fallback_path:
        fallback_eval = _evaluate_path_adaptive(
            [(float(point[0]), float(point[1])) for point in fallback_path],
            passable,
            layers,
            cost_func,
            weights,
            collision_weight,
            smooth_weight,
            length_weight,
        )
        fallback_eval["path"] = [(float(point[0]), float(point[1])) for point in fallback_path]
        feasible_archive.consider(fallback_eval, source="seed_fallback")
    convergence = [float(best_eval["value"])]

    lower = np.zeros(dim)
    upper = np.full(dim, float(size - 1))

    for iteration in range(1, iterations + 1):
        ratio = iteration / iterations
        adaptive_factor = _adaptive_factor(convergence)
        # Explicit phase 1: superior mirage search for broad exploration and guided search.
        best, best_eval = _run_superior_phase(
            rng,
            population,
            fitness,
            evaluations,
            passable,
            layers,
            cost_func,
            start,
            goal,
            weights,
            collision_weight,
            smooth_weight,
            length_weight,
            size,
            lower,
            upper,
            best,
            best_eval,
            feasible_archive,
            ratio,
            adaptive_factor,
        )

        # Explicit phase 2: inferior mirage search for local refinement near elite paths.
        best, best_eval = _run_inferior_phase(
            rng,
            population,
            fitness,
            evaluations,
            passable,
            layers,
            cost_func,
            start,
            goal,
            weights,
            collision_weight,
            smooth_weight,
            length_weight,
            size,
            best,
            best_eval,
            feasible_archive,
            ratio,
            adaptive_factor,
        )

        convergence.append(float(best_eval["value"]))

    if feasible_archive.evaluation is not None:
        final_eval = feasible_archive.evaluation
        final_path = list(final_eval["path"])
        final_source = feasible_archive.source
    else:
        final_eval = best_eval
        final_path = list(best_eval["path"])
        final_source = "best_infeasible"

    return {
        "path": final_path,
        "fitness": float(final_eval["value"]),
        "collision_count": int(final_eval["collision_count"]),
        "segment_collision_count": int(final_eval.get("segment_collision_count", 0)),
        "feasible": final_source != "best_infeasible",
        "used_feasible_archive": feasible_archive.evaluation is not None,
        "used_seed_fallback": final_source == "seed_fallback",
        "solution_source": final_source,
        "convergence": convergence,
        "adaptive_weights": weights,
    }
