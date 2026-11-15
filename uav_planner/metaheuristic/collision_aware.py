"""Collision-aware fixed-waypoint encoding for IAMSO seed paths."""

from __future__ import annotations

import math

import numpy as np

from uav_planner.metaheuristic.encoding import clip_vector
from uav_planner.geometry import segment_has_collision


def line_of_sight_simplify(path, passable) -> list[tuple[float, float]]:
    """Keep only path points required to preserve collision-free visibility."""
    points = [(float(point[0]), float(point[1])) for point in path]
    if len(points) <= 2:
        return points

    simplified = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        next_index = anchor + 1
        for candidate in range(len(points) - 1, anchor, -1):
            if not segment_has_collision(passable, points[anchor], points[candidate]):
                next_index = candidate
                break
        simplified.append(points[next_index])
        anchor = next_index
    return simplified


def _densify_to_count(
    points: list[tuple[float, float]], target_count: int
) -> list[tuple[float, float]]:
    dense = list(points)
    while len(dense) < target_count:
        segment_index = max(
            range(len(dense) - 1),
            key=lambda index: math.dist(dense[index], dense[index + 1]),
        )
        a = dense[segment_index]
        b = dense[segment_index + 1]
        midpoint = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        dense.insert(segment_index + 1, midpoint)
    return dense


def collision_aware_encode(path, passable, waypoint_count: int) -> np.ndarray | None:
    """Encode a safe path without dropping visibility-critical corner points."""
    if waypoint_count <= 0 or not path:
        return None
    simplified = line_of_sight_simplify(path, passable)
    target_point_count = waypoint_count + 2
    if len(simplified) > target_point_count:
        return None
    fixed_points = _densify_to_count(simplified, target_point_count)
    return np.asarray(fixed_points[1:-1], dtype=float).reshape(-1)


def collision_aware_initial_population(
    rng: np.random.Generator,
    population_size: int,
    waypoint_count: int,
    size: int,
    start,
    goal,
    seed_paths,
    *,
    passable,
) -> np.ndarray:
    """Build the IAMSO population using collision-aware seed encoding."""
    dim = waypoint_count * 2
    population = rng.uniform(0.0, float(size - 1), size=(population_size, dim))
    line = np.linspace(
        np.asarray(start, dtype=float),
        np.asarray(goal, dtype=float),
        waypoint_count + 2,
    )[1:-1]
    line_vector = line.reshape(-1)
    population[0] = line_vector

    encoded_seeds = []
    for path in seed_paths or []:
        encoded = collision_aware_encode(path, passable, waypoint_count)
        if encoded is not None:
            encoded_seeds.append(encoded)

    write_index = 1
    for encoded in encoded_seeds:
        if write_index >= population_size:
            break
        population[write_index] = encoded
        write_index += 1

    noise_sources = encoded_seeds or [line_vector]
    noise_scale = max(2.0, size * 0.08)
    while write_index < min(population_size, 12):
        source = noise_sources[write_index % len(noise_sources)]
        population[write_index] = source + rng.normal(0.0, noise_scale, size=dim)
        write_index += 1
    return clip_vector(population, size)
