"""Map generator for the no-task corner-to-corner benchmark."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from uav_planner.algorithms.common import iter_moves
from uav_planner.geometry import segment_has_collision
from uav_planner.map_generator import _compatible_speed_map, _make_inflation, _make_rain, _make_wind
from uav_planner.models import MapLayers
from uav_planner.obstacle_generation import generate_multi_shape_obstacles, is_connected


def is_strictly_reachable(
    passable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> bool:
    """Check grid reachability without allowing obstacle-corner crossings."""
    if not passable[start] or not passable[goal]:
        return False

    queue = deque([start])
    visited = {start}
    size = int(passable.shape[0])
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for dr, dc, _length in iter_moves():
            neighbor = current[0] + dr, current[1] + dc
            if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size):
                continue
            if neighbor in visited or not passable[neighbor]:
                continue
            if segment_has_collision(passable, current, neighbor):
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return False


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


def generate_no_task_corner_map(
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

    if not is_connected(obstacles, start, goal, tuple(iter_moves())):
        raise RuntimeError("generated no-task corner map is not grid connected")

    inflation = _make_inflation(obstacles)
    wind = _make_wind(rng, size)
    rain = _make_rain(rng, size)
    value = np.zeros((size, size), dtype=float)
    tasks: list[tuple[int, int]] = []
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
    if not is_strictly_reachable(passable, start, goal):
        raise RuntimeError("generated no-task corner map is not strictly reachable")
    return layers, passable, obstacle_meta
