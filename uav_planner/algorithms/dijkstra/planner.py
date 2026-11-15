"""Dijkstra path planner."""

from __future__ import annotations

import heapq

from uav_planner.geometry import smooth_planner_result

from ..common import edge_cost, is_cell_available, iter_moves, map_size, reconstruct_path


def dijkstra(passable, speed_map, cost_func, start, goal):
    """Dijkstra search on the grid."""
    pq = [(0, start)]
    dist = {start: 0}
    parent = {}
    size = map_size(passable)

    while pq:
        cost, (x, y) = heapq.heappop(pq)

        if (x, y) == goal:
            break

        for dx, dy, _length in iter_moves():
            nx, ny = x + dx, y + dy
            if is_cell_available(passable, (nx, ny), size):
                c = edge_cost(cost_func, (x, y), (nx, ny), speed_map)
                new_cost = cost + c
                if (nx, ny) not in dist or new_cost < dist[(nx, ny)]:
                    dist[(nx, ny)] = new_cost
                    parent[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (new_cost, (nx, ny)))

    return smooth_planner_result(reconstruct_path(parent, start, goal), passable)
