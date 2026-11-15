"""RRT path planner."""

from __future__ import annotations

import math
import random

from uav_planner.geometry import smooth_planner_result

from ..common import map_size, seed_value


def _nearest_node_index(nodes, target):
    return min(range(len(nodes)), key=lambda i: math.dist(nodes[i], target))


def _steer(from_node, to_node, step_size):
    distance = math.dist(from_node, to_node)
    if distance <= step_size:
        return float(to_node[0]), float(to_node[1])
    ratio = step_size / max(distance, 1e-9)
    return (
        float(from_node[0] + (to_node[0] - from_node[0]) * ratio),
        float(from_node[1] + (to_node[1] - from_node[1]) * ratio),
    )


def _reconstruct_rrt(nodes, parents, goal_index):
    path = []
    index = goal_index
    while index is not None:
        path.append(nodes[index])
        index = parents[index]
    path.reverse()
    return path


def rrt(passable, speed_map, cost_func, start, goal):
    """Rapidly-exploring random tree planner in continuous grid coordinates."""
    size = map_size(passable)
    if not passable[start] or not passable[goal]:
        return None

    rng = random.Random(seed_value(start, goal, 11))
    start_f = (float(start[0]), float(start[1]))
    goal_f = (float(goal[0]), float(goal[1]))
    nodes = [start_f]
    parents = [None]

    step_size = max(2.0, size / 18.0)
    goal_radius = step_size * 1.5
    max_iterations = max(1200, size * size * 4)
    goal_sample_rate = 0.18

    for _ in range(max_iterations):
        if rng.random() < goal_sample_rate:
            sample = goal_f
        else:
            sample = (rng.uniform(0, size - 1), rng.uniform(0, size - 1))

        nearest_index = _nearest_node_index(nodes, sample)
        new_node = _steer(nodes[nearest_index], sample, step_size)

        row = int(round(new_node[0]))
        col = int(round(new_node[1]))
        if not (0 <= row < size and 0 <= col < size and passable[row, col]):
            continue

        nodes.append(new_node)
        parents.append(nearest_index)
        new_index = len(nodes) - 1

        if math.dist(new_node, goal_f) <= goal_radius:
            nodes.append(goal_f)
            parents.append(new_index)
            return smooth_planner_result(_reconstruct_rrt(nodes, parents, len(nodes) - 1), passable)

    return None
