"""Shared helpers for path-planning algorithms."""

from __future__ import annotations

from uav_planner.config import MOVES


def reconstruct_path(parent, start, goal):
    """Rebuild a path from a parent map."""
    if goal not in parent:
        return None
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)
    path.reverse()
    return path


def map_size(passable):
    return passable.shape[0]


def iter_moves():
    for move in MOVES:
        if len(move) == 3:
            yield move
        else:
            dx, dy = move
            yield dx, dy, (dx * dx + dy * dy) ** 0.5


def is_cell_available(passable, q, size):
    row, col = q
    return 0 <= row < size and 0 <= col < size and passable[row, col]


def edge_cost(cost_func, p, q, cost_data):
    return float(cost_func(p, q, cost_data))


def seed_value(start, goal, offset):
    """Stable random seed so repeated runs are reproducible for one scenario."""
    return (
        int(start[0]) * 73856093
        ^ int(start[1]) * 19349663
        ^ int(goal[0]) * 83492791
        ^ int(goal[1]) * 2654435761
        ^ offset
    ) & 0xFFFFFFFF
