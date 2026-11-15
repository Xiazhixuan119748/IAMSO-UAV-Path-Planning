"""
地图生成模块。

按照数学模型生成 5 层代价地图：
O 障碍层、I 膨胀层、W 风场、R 降雨层、V 巡航价值层。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from uav_planner.config import (
    M_TASKS,
    MOVES,
    NORMAL_SPEED,
    OBSTACLE_TARGET_RATIO,
    R_INF,
    RHO,
    SIGMA,
    SIZE,
    V_MAX,
)
from uav_planner.models import MapLayers
from uav_planner.obstacle_generation import generate_multi_shape_obstacles


MAP_GENERATION_REACHABLE = "reachable"
MAP_GENERATION_RANDOM = "random"
MAP_GENERATION_LABELS = {
    MAP_GENERATION_REACHABLE: "可达地图",
    MAP_GENERATION_RANDOM: "随机压力地图",
}
TERMINAL_SAFETY_RADIUS = {
    30: 2,
    60: 3,
    90: 4,
}


def _smooth_field(field: np.ndarray, passes: int = 8) -> np.ndarray:
    """Small dependency-free smoothing kernel for rain and wind layers."""
    result = field.astype(float, copy=True)
    for _ in range(passes):
        result = (
            result
            + np.roll(result, 1, axis=0)
            + np.roll(result, -1, axis=0)
            + np.roll(result, 1, axis=1)
            + np.roll(result, -1, axis=1)
        ) / 5.0
    return result


def _normalize(field: np.ndarray) -> np.ndarray:
    low = float(field.min())
    high = float(field.max())
    if high - low <= 1e-12:
        return np.zeros_like(field, dtype=float)
    return (field - low) / (high - low)


def _difficulty_value(size: int, table: dict[int, float | int], default):
    matched = default
    for limit in sorted(table):
        if size <= limit:
            return table[limit]
        matched = table[limit]
    return matched


def _target_obstacle_ratio(size: int) -> float:
    return float(_difficulty_value(size, OBSTACLE_TARGET_RATIO, 0.15))


def _terminal_safety_radius(size: int) -> int:
    return int(_difficulty_value(size, TERMINAL_SAFETY_RADIUS, 2))


def _make_obstacles(rng: np.random.Generator, size: int) -> np.ndarray:
    obstacles, _meta = generate_multi_shape_obstacles(
        size=size,
        target_ratio=_target_obstacle_ratio(size),
        moves=MOVES,
        rng=rng,
    )
    return obstacles


def _make_reachable_obstacles(
    rng: np.random.Generator,
    size: int,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> np.ndarray:
    obstacles, _meta = generate_multi_shape_obstacles(
        size=size,
        target_ratio=_target_obstacle_ratio(size),
        moves=MOVES,
        rng=rng,
        start=start,
        goal=goal,
        safety_radius=_terminal_safety_radius(size),
        connectivity_protected=True,
    )
    return obstacles


def _normalize_generation_mode(generation_mode: str | None) -> str:
    if not generation_mode:
        return MAP_GENERATION_REACHABLE

    normalized = str(generation_mode).strip().lower()
    aliases = {
        "reachable": MAP_GENERATION_REACHABLE,
        "protected": MAP_GENERATION_REACHABLE,
        "safe": MAP_GENERATION_REACHABLE,
        "可达": MAP_GENERATION_REACHABLE,
        "可达地图": MAP_GENERATION_REACHABLE,
        "random": MAP_GENERATION_RANDOM,
        "pressure": MAP_GENERATION_RANDOM,
        "stress": MAP_GENERATION_RANDOM,
        "随机": MAP_GENERATION_RANDOM,
        "随机压力": MAP_GENERATION_RANDOM,
        "随机压力地图": MAP_GENERATION_RANDOM,
    }
    if normalized not in aliases:
        raise ValueError(f"未知地图生成模式: {generation_mode}")
    return aliases[normalized]


def _distance_to_obstacles(obstacles: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance from each grid cell to the nearest obstacle."""
    if not np.any(obstacles == 1):
        return np.full(obstacles.shape, float(max(obstacles.shape)), dtype=float)
    return distance_transform_edt(obstacles == 0)


def _make_inflation(obstacles: np.ndarray) -> np.ndarray:
    dist = _distance_to_obstacles(obstacles)
    inflation = np.exp(-RHO * dist)
    inflation[dist > R_INF] = 0.0
    inflation[obstacles == 1] = 1.0
    return np.clip(inflation, 0.0, 1.0)


def _make_wind(rng: np.random.Generator, size: int) -> np.ndarray:
    yy, xx = np.indices((size, size))
    wx = np.zeros((size, size), dtype=float)
    wy = np.zeros((size, size), dtype=float)

    prevailing_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    prevailing_speed = float(rng.uniform(0.2, 0.45))
    wx += prevailing_speed * np.sin(prevailing_angle)
    wy += prevailing_speed * np.cos(prevailing_angle)

    region_count = max(2, size // 35)
    for _ in range(region_count):
        center_r = float(rng.uniform(0, size))
        center_c = float(rng.uniform(0, size))
        sigma = float(rng.uniform(size * 0.18, size * 0.38))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        strength = float(rng.uniform(0.35, 1.0))
        region = np.exp(-((yy - center_r) ** 2 + (xx - center_c) ** 2) / (2 * sigma * sigma))
        wx += strength * np.sin(angle) * region
        wy += strength * np.cos(angle) * region

    wx = _smooth_field(wx, passes=4)
    wy = _smooth_field(wy, passes=4)
    magnitude = np.sqrt(wx * wx + wy * wy)
    max_mag = float(magnitude.max())
    if max_mag > 1e-12:
        wx = wx / max_mag * V_MAX
        wy = wy / max_mag * V_MAX
    return np.stack([wx, wy], axis=-1)


def _make_rain(rng: np.random.Generator, size: int) -> np.ndarray:
    rain = np.zeros((size, size), dtype=float)
    yy, xx = np.indices((size, size))
    storm_count = max(3, size // 45)

    for _ in range(storm_count):
        center_r = float(rng.uniform(0, size))
        center_c = float(rng.uniform(0, size))
        sigma = float(rng.uniform(size * 0.05, size * 0.16))
        strength = float(rng.uniform(0.45, 1.0))
        rain += strength * np.exp(-((yy - center_r) ** 2 + (xx - center_c) ** 2) / (2 * sigma * sigma))
    return _normalize(_smooth_field(rain, passes=4))


def _make_cruise_value(rng: np.random.Generator, size: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    value = np.zeros((size, size), dtype=float)
    tasks: list[tuple[int, int]] = []
    yy, xx = np.indices((size, size))
    scale = max(1.0, size / SIZE)
    task_count = max(1, int(round(M_TASKS * scale)))

    for _ in range(task_count):
        row = int(rng.integers(0, size))
        col = int(rng.integers(0, size))
        tasks.append((row, col))
        value += np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2 * (SIGMA * scale) ** 2))
    return np.clip(_normalize(value), 0.0, 1.0), tasks


def _compatible_speed_map(inflation: np.ndarray, rain: np.ndarray) -> np.ndarray:
    speed = NORMAL_SPEED * (1.0 - 0.55 * inflation) * (1.0 - 0.35 * rain)
    speed = np.clip(speed, NORMAL_SPEED * 0.25, NORMAL_SPEED)
    return speed


def generate_map(start, goal, seed=None, size=SIZE, generation_mode=MAP_GENERATION_REACHABLE):
    """
    生成多层代价地图。

    Returns:
        layers: dict，包含 O/I/W/R/V/tasks/speed_map
        passable: 1 表示可通行，0 表示障碍
    """
    rng = np.random.default_rng(seed)
    generation_mode = _normalize_generation_mode(generation_mode)
    if generation_mode == MAP_GENERATION_REACHABLE:
        obstacles = _make_reachable_obstacles(rng, size, start, goal)
    else:
        obstacles = _make_obstacles(rng, size)

    for point in (start, goal):
        row, col = point
        obstacles[row, col] = 0

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
    return layers, passable
