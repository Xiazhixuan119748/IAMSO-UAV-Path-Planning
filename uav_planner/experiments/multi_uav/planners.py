"""Strict-edge A* baseline for the continuous multi-UAV evaluator."""

import heapq

from uav_planner.algorithms.common import edge_cost, iter_moves, reconstruct_path
from uav_planner.costmap import octile_distance
from uav_planner.geometry import segment_has_collision


def safe_astar(passable, layers, cost_func, start, goal):
    queue = [(0.0, start)]
    distance = {start: 0.0}
    parent = {}
    closed = set()
    size = passable.shape[0]
    while queue:
        _, current = heapq.heappop(queue)
        if current in closed:
            continue
        if current == goal:
            return reconstruct_path(parent, start, goal)
        closed.add(current)
        for dr, dc, _ in iter_moves():
            neighbor = (current[0] + dr, current[1] + dc)
            if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size):
                continue
            if not passable[neighbor] or segment_has_collision(passable, current, neighbor):
                continue
            cost = distance[current] + edge_cost(cost_func, current, neighbor, layers)
            if cost < distance.get(neighbor, float("inf")):
                distance[neighbor] = cost
                parent[neighbor] = current
                heapq.heappush(queue, (cost + octile_distance(neighbor, goal), neighbor))
    return None
