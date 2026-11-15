"""A* path planner."""

from __future__ import annotations

import heapq

from uav_planner.costmap import octile_distance
from uav_planner.geometry import smooth_planner_result

from ..common import edge_cost, is_cell_available, iter_moves, map_size, reconstruct_path


def astar(passable, speed_map, cost_func, start, goal):
    """A* search using octile distance as the heuristic."""
    pq = [(0, start)]
    g = {start: 0}
    parent = {}
    size = map_size(passable)

    while pq:
        _, (x, y) = heapq.heappop(pq)

        if (x, y) == goal:
            break

        for dx, dy, _length in iter_moves():
            nx, ny = x + dx, y + dy
            if is_cell_available(passable, (nx, ny), size):
                c = edge_cost(cost_func, (x, y), (nx, ny), speed_map)
                new_g = g[(x, y)] + c
                if (nx, ny) not in g or new_g < g[(nx, ny)]:
                    g[(nx, ny)] = new_g
                    f = new_g + octile_distance((nx, ny), goal)
                    parent[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (f, (nx, ny)))

    return smooth_planner_result(reconstruct_path(parent, start, goal), passable)
