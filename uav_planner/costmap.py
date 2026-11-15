"""Layered cost-map helpers for the mathematical UAV path model."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from uav_planner.config import (
    EXPOSURE_INTENSITY_GAMMA,
    GAMMA,
    MAX_TAILWIND_SPEED_FACTOR,
    MIN_SPEED_FACTOR,
    MODEL_COST_WEIGHTS,
    MODEL_WEIGHTS,
    NORMAL_SPEED,
    V_MAX,
    WIND_SPEED_EFFECT,
)


def is_layer_map(data) -> bool:
    return isinstance(data, Mapping) and {"O", "I", "W", "R", "V"}.issubset(data.keys())


def speed_grid(data):
    if is_layer_map(data):
        return data["speed_map"]
    return data


def passable_grid(data):
    if is_layer_map(data):
        return 1 - data["O"]
    return data


def heading_vector(p: Sequence[float], q: Sequence[float]) -> tuple[float, float]:
    dr = float(q[0]) - float(p[0])
    dc = float(q[1]) - float(p[1])
    length = math.hypot(dr, dc)
    if length <= 1e-12:
        return 0.0, 0.0
    return dr / length, dc / length


def C_inf(q: Sequence[int], layers) -> float:
    return float(layers["I"][int(q[0]), int(q[1])])


def C_inf_exposure(q: Sequence[int], layers) -> float:
    return C_inf(q, layers) ** EXPOSURE_INTENSITY_GAMMA


def C_wind(q: Sequence[int], h_hat: Sequence[float], layers) -> float:
    wind = layers["W"][int(q[0]), int(q[1])]
    along_track = float(wind[0] * h_hat[0] + wind[1] * h_hat[1])
    return max(0.0, -along_track / max(V_MAX, 1e-9))


def effective_speed(q: Sequence[int], h_hat: Sequence[float], data) -> float:
    """Return ground-relative speed after rain/inflation base speed and wind assistance/resistance."""
    grid = speed_grid(data)
    row, col = int(q[0]), int(q[1])
    base_speed = float(grid[row, col])
    if base_speed <= 0:
        return 0.0

    if not is_layer_map(data):
        return base_speed

    wind = data["W"][row, col]
    along_track = float(wind[0] * h_hat[0] + wind[1] * h_hat[1])
    wind_delta = WIND_SPEED_EFFECT * along_track
    min_speed = NORMAL_SPEED * MIN_SPEED_FACTOR
    max_speed = NORMAL_SPEED * MAX_TAILWIND_SPEED_FACTOR
    return max(min_speed, min(max_speed, base_speed + wind_delta))


def C_rain(q: Sequence[int], layers) -> float:
    rain = float(layers["R"][int(q[0]), int(q[1])])
    return rain ** GAMMA


def C_rain_exposure(q: Sequence[int], layers) -> float:
    rain = float(layers["R"][int(q[0]), int(q[1])])
    return rain ** EXPOSURE_INTENSITY_GAMMA


def C_cruise(q: Sequence[int], layers) -> float:
    return 1.0 - float(layers["V"][int(q[0]), int(q[1])])


def step_length(p: Sequence[float], q: Sequence[float]) -> float:
    return math.hypot(float(q[0]) - float(p[0]), float(q[1]) - float(p[1]))


def _weight(weights, name: str, legacy_index: int | None, default: float) -> float:
    if isinstance(weights, Mapping):
        return float(weights.get(name, default))
    if weights is None:
        return default
    if legacy_index is None:
        return default
    try:
        return float(weights[legacy_index])
    except (IndexError, TypeError):
        return default


def step_cost(p: Sequence[int], q: Sequence[int], layers, weights=MODEL_COST_WEIGHTS) -> float:
    row, col = int(q[0]), int(q[1])
    if layers["O"][row, col]:
        return float("inf")

    w_len = _weight(weights, "length", None, MODEL_COST_WEIGHTS["length"])
    w_inf = _weight(weights, "inflation", 0, MODEL_WEIGHTS[0])
    w_wind = _weight(weights, "wind", 1, MODEL_WEIGHTS[1])
    w_rain = _weight(weights, "rain", 2, MODEL_WEIGHTS[2])
    w_cruise = _weight(weights, "cruise", 3, MODEL_WEIGHTS[3])
    length = step_length(p, q)
    heading = heading_vector(p, q)
    return (
        w_len * length
        + w_inf * length * C_inf_exposure(q, layers)
        + w_rain * length * C_rain_exposure(q, layers)
        + w_cruise * length * C_cruise(q, layers)
        + w_wind * C_wind(q, heading, layers)
    )


def point_cost(row: int, col: int, layers, weights=MODEL_COST_WEIGHTS) -> float:
    """Cell cost used by path evaluators that do not know segment heading."""
    q = (int(row), int(col))
    if layers["O"][q]:
        return 1e6
    w_len = _weight(weights, "length", None, MODEL_COST_WEIGHTS["length"])
    w_inf = _weight(weights, "inflation", 0, MODEL_WEIGHTS[0])
    w_rain = _weight(weights, "rain", 2, MODEL_WEIGHTS[2])
    w_cruise = _weight(weights, "cruise", 3, MODEL_WEIGHTS[3])
    return w_len + w_inf * C_inf_exposure(q, layers) + w_rain * C_rain_exposure(q, layers) + w_cruise * C_cruise(q, layers)


def octile_distance(a: Sequence[int], b: Sequence[int]) -> float:
    dr = abs(int(a[0]) - int(b[0]))
    dc = abs(int(a[1]) - int(b[1]))
    return (dr + dc) + (math.sqrt(2) - 2.0) * min(dr, dc)
