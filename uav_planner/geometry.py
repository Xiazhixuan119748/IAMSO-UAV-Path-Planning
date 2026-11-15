"""Geometry helpers for continuous UAV paths."""

from __future__ import annotations

from typing import Sequence

from uav_planner.metaheuristic.encoding import smooth_path


class PathResult(list):
    """List-like path that can carry planner metadata."""

    def __init__(self, path, used_fallback=False):
        super().__init__(path)
        self.used_fallback = used_fallback


def _point_in_map(point: Sequence[float], size: int) -> bool:
    row = float(point[0])
    col = float(point[1])
    return 0.0 <= row <= float(size - 1) and 0.0 <= col <= float(size - 1)


def _segment_intersects_rect(a: Sequence[float], b: Sequence[float], row: int, col: int, margin=0.0) -> bool:
    """Return True when a row/col segment intersects a grid-cell rectangle."""
    r0, c0 = float(a[0]), float(a[1])
    r1, c1 = float(b[0]), float(b[1])
    dr = r1 - r0
    dc = c1 - c0

    r_min = row - 0.5 - margin
    r_max = row + 0.5 + margin
    c_min = col - 0.5 - margin
    c_max = col + 0.5 + margin

    t0 = 0.0
    t1 = 1.0
    for start, delta, low, high in ((r0, dr, r_min, r_max), (c0, dc, c_min, c_max)):
        if abs(delta) <= 1e-12:
            if start < low or start > high:
                return False
            continue
        inv_delta = 1.0 / delta
        enter = (low - start) * inv_delta
        exit_ = (high - start) * inv_delta
        if enter > exit_:
            enter, exit_ = exit_, enter
        t0 = max(t0, enter)
        t1 = min(t1, exit_)
        if t0 > t1:
            return False
    return True


def collision_cells_for_segment(passable, a: Sequence[float], b: Sequence[float], margin=0.0):
    """Return blocked cells intersected by a continuous path segment."""
    size = passable.shape[0]
    row_min = max(0, int(min(float(a[0]), float(b[0])) - 0.5 - margin))
    row_max = min(size - 1, int(max(float(a[0]), float(b[0])) + 0.5 + margin) + 1)
    col_min = max(0, int(min(float(a[1]), float(b[1])) - 0.5 - margin))
    col_max = min(size - 1, int(max(float(a[1]), float(b[1])) + 0.5 + margin) + 1)

    cells = set()
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            if passable[row, col]:
                continue
            if _segment_intersects_rect(a, b, row, col, margin=margin):
                cells.add((row, col))
    return cells


def collision_cells_for_path(passable, path, margin=0.0):
    """Return blocked cells intersected by any segment of the path."""
    if not path:
        return set()

    cells = set()
    if len(path) == 1:
        row = int(round(float(path[0][0])))
        col = int(round(float(path[0][1])))
        if 0 <= row < passable.shape[0] and 0 <= col < passable.shape[1] and not passable[row, col]:
            cells.add((row, col))
        return cells

    for a, b in zip(path[:-1], path[1:]):
        cells.update(collision_cells_for_segment(passable, a, b, margin=margin))
    return cells


def segment_has_collision(passable, a: Sequence[float], b: Sequence[float], margin=0.0) -> bool:
    """Return True when a path segment goes out of map or intersects blocked cells."""
    size = passable.shape[0]
    if not _point_in_map(a, size) or not _point_in_map(b, size):
        return True
    return bool(collision_cells_for_segment(passable, a, b, margin=margin))


def collision_segment_indices_for_path(passable, path, margin=0.0):
    """Return the indices of colliding segments in a path."""
    if not path:
        return []

    size = passable.shape[0]
    if len(path) == 1:
        row = int(round(float(path[0][0])))
        col = int(round(float(path[0][1])))
        if not _point_in_map(path[0], size):
            return [0]
        if 0 <= row < size and 0 <= col < size and not passable[row, col]:
            return [0]
        return []

    indices = []
    for index, (a, b) in enumerate(zip(path[:-1], path[1:])):
        if segment_has_collision(passable, a, b, margin=margin):
            indices.append(index)
    return indices


def collision_segment_count_for_path(passable, path, margin=0.0) -> int:
    """Count how many path segments collide, matching RRT-style edge validation."""
    return len(collision_segment_indices_for_path(passable, path, margin=margin))


def smooth_planner_result(path, passable, used_fallback=False):
    """Return the planner path after visual smoothing without collision repair."""
    if path is None:
        return None

    base_path = [(float(p[0]), float(p[1])) for p in path]
    smoothed = smooth_path(base_path, samples_per_segment=10)
    return PathResult(smoothed, used_fallback=used_fallback)
