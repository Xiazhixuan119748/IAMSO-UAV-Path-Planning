"""Benchmark-only map generation with protected start-goal connectivity."""

from __future__ import annotations

from typing import Any

import numpy as np

from uav_planner.algorithms.common import iter_moves
from uav_planner.map_generator import (
    _compatible_speed_map,
    _make_cruise_value,
    _make_inflation,
    _make_rain,
    _make_wind,
)
from uav_planner.models import MapLayers
from uav_planner.obstacle_generation import generate_multi_shape_obstacles


def make_protected_obstacles(
    *,
    size: int,
    target_ratio: float,
    seed: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    safety_radius: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    return generate_multi_shape_obstacles(
        size=size,
        target_ratio=target_ratio,
        seed=seed,
        start=start,
        goal=goal,
        safety_radius=safety_radius,
        connectivity_protected=True,
        moves=tuple(iter_moves()),
    )


def generate_benchmark_map(
    *,
    start: tuple[int, int],
    goal: tuple[int, int],
    seed: int,
    size: int,
    target_ratio: float,
    safety_radius: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    obstacles, obstacle_meta = make_protected_obstacles(
        size=size,
        target_ratio=target_ratio,
        seed=seed,
        start=start,
        goal=goal,
        safety_radius=safety_radius,
    )

    inflation = _make_inflation(obstacles)
    wind = _make_wind(rng, size)
    rain = _make_rain(rng, size)
    value, tasks = _make_cruise_value(rng, size)
    speed_map = _compatible_speed_map(inflation, rain)

    layers = MapLayers(
        O=obstacles,
        I=inflation,
        W=wind,
        R=rain,
        V=value,
        tasks=tasks,
        speed_map=speed_map,
    ).as_dict()
    passable = 1 - obstacles
    return layers, passable, obstacle_meta
