"""Shared obstacle-generation helpers for runtime maps and fixed benchmarks."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Iterable

import numpy as np


Move = tuple[int, int, float]
Point = tuple[int, int]


def clip_window(size: int, row_min: int, row_max: int, col_min: int, col_max: int) -> tuple[int, int, int, int]:
    return max(0, row_min), min(size, row_max), max(0, col_min), min(size, col_max)


def patch_indices(row_min: int, row_max: int, col_min: int, col_max: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(row_min, row_max, dtype=float)[:, None]
    cols = np.arange(col_min, col_max, dtype=float)[None, :]
    return rows, cols


def span_limits(size: int) -> tuple[int, int]:
    min_span = 2
    max_span = max(4, min(12, int(round(size * 0.12))))
    return min_span, max_span


def bounded_pair(rng: np.random.Generator, size: int, min_span: int | None = None) -> tuple[int, int]:
    low, high = span_limits(size)
    if min_span is not None:
        low = max(low, min_span)
    high = max(high, low)

    first = int(rng.integers(low, high + 1))
    max_second = max(low, min(high, int(round(first * 1.8))))
    min_second = max(low, int(math.ceil(first / 1.8)))
    second = int(rng.integers(min_second, max_second + 1))

    if rng.random() < 0.5:
        return first, second
    return second, first


def _empty_mask(size: int) -> np.ndarray:
    return np.zeros((size, size), dtype=bool)


def rectangle_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    height, width = bounded_pair(rng, size)
    row = int(rng.integers(0, max(1, size - height + 1)))
    col = int(rng.integers(0, max(1, size - width + 1)))
    mask = _empty_mask(size)
    mask[row:row + height, col:col + width] = True
    return mask


def circle_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    _low, high = span_limits(size)
    radius = int(rng.integers(2, max(3, high // 2 + 1)))
    center_r = int(rng.integers(0, size))
    center_c = int(rng.integers(0, size))
    row_min, row_max, col_min, col_max = clip_window(
        size,
        center_r - radius - 1,
        center_r + radius + 2,
        center_c - radius - 1,
        center_c + radius + 2,
    )
    rows, cols = patch_indices(row_min, row_max, col_min, col_max)
    mask = _empty_mask(size)
    mask[row_min:row_max, col_min:col_max] = (rows - center_r) ** 2 + (cols - center_c) ** 2 <= radius ** 2
    return mask


def ellipse_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    axis_r, axis_c = bounded_pair(rng, size)
    axis_r = max(2, axis_r // 2)
    axis_c = max(2, axis_c // 2)
    center_r = int(rng.integers(0, size))
    center_c = int(rng.integers(0, size))
    angle = float(rng.uniform(0.0, math.pi))
    bound = max(axis_r, axis_c) + 2
    row_min, row_max, col_min, col_max = clip_window(
        size,
        center_r - bound,
        center_r + bound + 1,
        center_c - bound,
        center_c + bound + 1,
    )
    rows, cols = patch_indices(row_min, row_max, col_min, col_max)
    x = cols - center_c
    y = rows - center_r
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    xr = x * cos_a + y * sin_a
    yr = -x * sin_a + y * cos_a
    mask = _empty_mask(size)
    mask[row_min:row_max, col_min:col_max] = (
        (xr / max(axis_c, 1e-9)) ** 2 + (yr / max(axis_r, 1e-9)) ** 2 <= 1.0
    )
    return mask


def diamond_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    radius_r, radius_c = bounded_pair(rng, size)
    radius_r = max(2, radius_r // 2)
    radius_c = max(2, radius_c // 2)
    center_r = int(rng.integers(0, size))
    center_c = int(rng.integers(0, size))
    row_min, row_max, col_min, col_max = clip_window(
        size,
        center_r - radius_r - 1,
        center_r + radius_r + 2,
        center_c - radius_c - 1,
        center_c + radius_c + 2,
    )
    rows, cols = patch_indices(row_min, row_max, col_min, col_max)
    mask = _empty_mask(size)
    mask[row_min:row_max, col_min:col_max] = (
        np.abs(rows - center_r) / max(radius_r, 1e-9)
        + np.abs(cols - center_c) / max(radius_c, 1e-9)
        <= 1.0
    )
    return mask


def l_shape_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    height, width = bounded_pair(rng, size, min_span=4)
    thickness = int(rng.integers(2, max(3, min(height, width) // 3 + 1)))
    row = int(rng.integers(0, max(1, size - height + 1)))
    col = int(rng.integers(0, max(1, size - width + 1)))
    orientation = int(rng.integers(0, 4))
    mask = _empty_mask(size)

    if orientation == 0:
        mask[row:row + height, col:col + thickness] = True
        mask[row + height - thickness:row + height, col:col + width] = True
    elif orientation == 1:
        mask[row:row + height, col + width - thickness:col + width] = True
        mask[row + height - thickness:row + height, col:col + width] = True
    elif orientation == 2:
        mask[row:row + height, col:col + thickness] = True
        mask[row:row + thickness, col:col + width] = True
    else:
        mask[row:row + height, col + width - thickness:col + width] = True
        mask[row:row + thickness, col:col + width] = True
    return mask


def t_shape_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    height, width = bounded_pair(rng, size, min_span=5)
    thickness = int(rng.integers(2, max(3, min(height, width) // 3 + 1)))
    row = int(rng.integers(0, max(1, size - height + 1)))
    col = int(rng.integers(0, max(1, size - width + 1)))
    orientation = int(rng.integers(0, 4))
    mask = _empty_mask(size)

    if orientation == 0:
        stem_col = col + (width - thickness) // 2
        mask[row:row + thickness, col:col + width] = True
        mask[row:row + height, stem_col:stem_col + thickness] = True
    elif orientation == 1:
        stem_row = row + (height - thickness) // 2
        mask[row:row + height, col + width - thickness:col + width] = True
        mask[stem_row:stem_row + thickness, col:col + width] = True
    elif orientation == 2:
        stem_col = col + (width - thickness) // 2
        mask[row + height - thickness:row + height, col:col + width] = True
        mask[row:row + height, stem_col:stem_col + thickness] = True
    else:
        stem_row = row + (height - thickness) // 2
        mask[row:row + height, col:col + thickness] = True
        mask[stem_row:stem_row + thickness, col:col + width] = True
    return mask


SHAPE_MASKERS = (
    ("rectangle", rectangle_mask),
    ("circle", circle_mask),
    ("ellipse", ellipse_mask),
    ("diamond", diamond_mask),
    ("l_shape", l_shape_mask),
    ("t_shape", t_shape_mask),
)


def clear_safety_zone(obstacles: np.ndarray, center: Point, radius: int) -> None:
    if radius <= 0:
        obstacles[center] = 0
        return

    size = obstacles.shape[0]
    row, col = center
    row_min, row_max, col_min, col_max = clip_window(size, row - radius, row + radius + 1, col - radius, col + radius + 1)
    rows, cols = patch_indices(row_min, row_max, col_min, col_max)
    disk = (rows - row) ** 2 + (cols - col) ** 2 <= radius ** 2
    obstacles[row_min:row_max, col_min:col_max][disk] = 0


def protect_terminals(obstacles: np.ndarray, start: Point, goal: Point, safety_radius: int) -> None:
    clear_safety_zone(obstacles, start, safety_radius)
    clear_safety_zone(obstacles, goal, safety_radius)


def is_connected(obstacles: np.ndarray, start: Point, goal: Point, moves: Iterable[Move]) -> bool:
    passable = obstacles == 0
    if not passable[start] or not passable[goal]:
        return False

    size = obstacles.shape[0]
    visited = np.zeros_like(passable, dtype=bool)
    queue: deque[Point] = deque([start])
    visited[start] = True

    while queue:
        row, col = queue.popleft()
        if (row, col) == goal:
            return True
        for dr, dc, _length in moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < size and 0 <= nc < size and passable[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return False


def generate_multi_shape_obstacles(
    *,
    size: int,
    target_ratio: float,
    moves: Iterable[Move],
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    start: Point | None = None,
    goal: Point | None = None,
    safety_radius: int = 0,
    connectivity_protected: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    if rng is None:
        rng = np.random.default_rng(seed)

    obstacles = np.zeros((size, size), dtype=np.uint8)
    shape_counts = {name: 0 for name, _masker in SHAPE_MASKERS}
    rejected_counts = {name: 0 for name, _masker in SHAPE_MASKERS}
    accepted = 0
    rejected = 0
    stagnant = 0

    points_protected = start is not None and goal is not None
    if points_protected:
        protect_terminals(obstacles, start, goal, safety_radius)

    max_attempts = max(800, int(size * size * (0.45 + target_ratio * 2.2)))
    move_list = tuple(moves)

    for _attempt in range(max_attempts):
        if float(obstacles.mean()) >= target_ratio:
            break

        name, masker = SHAPE_MASKERS[int(rng.integers(0, len(SHAPE_MASKERS)))]
        mask = masker(rng, size)
        new_cells = int(np.count_nonzero(mask & (obstacles == 0)))
        if new_cells <= 0:
            stagnant += 1
            continue

        candidate = obstacles.copy()
        candidate[mask] = 1
        if points_protected:
            protect_terminals(candidate, start, goal, safety_radius)

        if connectivity_protected and points_protected and not is_connected(candidate, start, goal, move_list):
            rejected_counts[name] += 1
            rejected += 1
            stagnant += 1
        else:
            obstacles = candidate
            shape_counts[name] += 1
            accepted += 1
            stagnant = 0

        if stagnant > max(200, size * 8) and float(obstacles.mean()) >= target_ratio * 0.96:
            break

    if points_protected:
        protect_terminals(obstacles, start, goal, safety_radius)

    meta = {
        "target_obstacle_ratio": float(target_ratio),
        "actual_obstacle_ratio": float(obstacles.mean()),
        "safety_radius": int(safety_radius),
        "accepted_shape_count": int(accepted),
        "rejected_shape_count": int(rejected),
        "shape_mix": shape_counts,
        "rejected_shape_mix": rejected_counts,
        "connectivity_protected": bool(connectivity_protected),
    }
    return obstacles, meta
