"""Fitness function for continuous waypoint paths."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from uav_planner.config import EXPOSURE_PATH_WEIGHT
from uav_planner.geometry import collision_cells_for_path
from uav_planner.metrics import calculate_exposure_metrics
from uav_planner.costmap import is_layer_map

from .encoding import decode, polyline_length, sample_path, smooth_path


CostFunction = Callable[[int, int, np.ndarray], float]


@dataclass(frozen=True)
class FitnessResult:
    value: float
    collision_count: int
    length: float
    smoothness: float
    path: list[tuple[float, float]]


def _nearest_cell(point: Sequence[float], size: int) -> tuple[int, int] | None:
    row = int(round(float(point[0])))
    col = int(round(float(point[1])))
    if row < 0 or col < 0 or row >= size or col >= size:
        return None
    return row, col


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
        angle = math.acos(cosine)
        penalty += angle * angle
    return penalty


def evaluate_path(
    path: Sequence[Sequence[float]],
    passable: np.ndarray,
    speed_map: np.ndarray,
    cost_func: CostFunction,
    collision_weight: float = 1000.0,
    smooth_weight: float = 2.0,
    exposure_weight: float = EXPOSURE_PATH_WEIGHT,
    sample_step: float = 0.5,
) -> FitnessResult:
    """Evaluate a continuous path with length/cost, collision, and smoothness terms."""
    size = int(passable.shape[0])
    sampled = sample_path(path, step=sample_step)
    collisions = set(collision_cells_for_path(passable, path))
    weighted_length = 0.0

    prev = None
    for point in sampled:
        cell = _nearest_cell(point, size)
        if cell is None:
            collisions.add(("out", int(round(float(point[0]))), int(round(float(point[1])))))
            cost = 1.0
        else:
            row, col = cell
            cost = float(cost_func(row, col, speed_map))
        if prev is not None:
            step_distance = math.dist(prev, point)
            weighted_length += step_distance * max(cost, 1e-9)
        prev = point

    collision_count = len(collisions)
    length = polyline_length(path)
    smoothness = _turn_penalty(path)
    exposure = calculate_exposure_metrics(path, speed_map)["total_exposure"] if is_layer_map(speed_map) else 0.0
    value = (
        weighted_length
        + collision_weight * collision_count
        + smooth_weight * smoothness
        + exposure_weight * exposure
    )
    return FitnessResult(value, collision_count, length, smoothness, [(float(r), float(c)) for r, c in path])


def evaluate_vector(
    vector: np.ndarray,
    passable: np.ndarray,
    speed_map: np.ndarray,
    cost_func: CostFunction,
    start: Sequence[float],
    goal: Sequence[float],
    collision_weight: float = 1000.0,
    smooth_weight: float = 2.0,
    exposure_weight: float = EXPOSURE_PATH_WEIGHT,
    smooth: bool = True,
) -> FitnessResult:
    """Decode and evaluate a waypoint vector."""
    decoded = decode(vector, start, goal)
    path = smooth_path(decoded) if smooth else decoded
    return evaluate_path(
        path,
        passable,
        speed_map,
        cost_func,
        collision_weight=collision_weight,
        smooth_weight=smooth_weight,
        exposure_weight=exposure_weight,
    )
