"""Depth-first path planner."""

from __future__ import annotations

from uav_planner.geometry import smooth_planner_result

from ..common import is_cell_available, iter_moves, map_size, reconstruct_path


def dfs(passable, speed_map, cost_func, start, goal):
    """Depth-first grid search."""
    stack = [start]
    visited = {start}
    parent = {}
    size = map_size(passable)

    while stack:
        x, y = stack.pop()

        if (x, y) == goal:
            break

        for dx, dy, _length in iter_moves():
            nx, ny = x + dx, y + dy
            if is_cell_available(passable, (nx, ny), size) and (nx, ny) not in visited:
                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)
                stack.append((nx, ny))

    return smooth_planner_result(reconstruct_path(parent, start, goal), passable)
