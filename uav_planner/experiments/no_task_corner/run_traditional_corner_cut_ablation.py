"""Test strict edge validation with the production path smoother."""

from __future__ import annotations

import heapq
import math
import random
import time

from uav_planner.algorithms.common import edge_cost, iter_moves, reconstruct_path, seed_value
from uav_planner.costmap import octile_distance
from uav_planner.costs import no_task_model_cost
from uav_planner.experiments.no_task_corner.dataset import load_map_bundle, no_task_corner_maps_dir
from uav_planner.experiments.no_task_corner.runner import clone_layers, collect_records
from uav_planner.geometry import segment_has_collision, smooth_planner_result
from uav_planner.metrics import calculate_collision_count


def _strict_neighbors(passable, current):
    size = int(passable.shape[0])
    for dr, dc, _ in iter_moves():
        neighbor = current[0] + dr, current[1] + dc
        row, col = neighbor
        if not (0 <= row < size and 0 <= col < size and passable[neighbor]):
            continue
        if segment_has_collision(passable, current, neighbor):
            continue
        yield neighbor


def strict_astar(passable, layers, cost_func, start, goal):
    queue = [(0.0, start)]
    costs = {start: 0.0}
    parent = {}
    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            break
        for neighbor in _strict_neighbors(passable, current):
            candidate = costs[current] + edge_cost(cost_func, current, neighbor, layers)
            if candidate >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = candidate
            parent[neighbor] = current
            heapq.heappush(queue, (candidate + octile_distance(neighbor, goal), neighbor))
    return smooth_planner_result(reconstruct_path(parent, start, goal), passable)


def strict_dijkstra(passable, layers, cost_func, start, goal):
    queue = [(0.0, start)]
    costs = {start: 0.0}
    parent = {}
    while queue:
        current_cost, current = heapq.heappop(queue)
        if current_cost > costs.get(current, math.inf):
            continue
        if current == goal:
            break
        for neighbor in _strict_neighbors(passable, current):
            candidate = current_cost + edge_cost(cost_func, current, neighbor, layers)
            if candidate >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = candidate
            parent[neighbor] = current
            heapq.heappush(queue, (candidate, neighbor))
    return smooth_planner_result(reconstruct_path(parent, start, goal), passable)


def strict_dfs(passable, layers, cost_func, start, goal):
    stack = [start]
    visited = {start}
    parent = {}
    while stack:
        current = stack.pop()
        if current == goal:
            break
        for neighbor in _strict_neighbors(passable, current):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = current
            stack.append(neighbor)
    return smooth_planner_result(reconstruct_path(parent, start, goal), passable)


def _steer(from_node, to_node, step_size):
    distance = math.dist(from_node, to_node)
    if distance <= step_size:
        return float(to_node[0]), float(to_node[1])
    ratio = step_size / max(distance, 1e-9)
    return (
        float(from_node[0] + (to_node[0] - from_node[0]) * ratio),
        float(from_node[1] + (to_node[1] - from_node[1]) * ratio),
    )


def strict_rrt(passable, layers, cost_func, start, goal):
    size = int(passable.shape[0])
    if not passable[start] or not passable[goal]:
        return None
    rng = random.Random(seed_value(start, goal, 11))
    start_f = float(start[0]), float(start[1])
    goal_f = float(goal[0]), float(goal[1])
    nodes = [start_f]
    parents = [None]
    step_size = max(2.0, size / 18.0)
    goal_radius = step_size * 1.5
    max_iterations = max(1200, size * size * 4)
    for _ in range(max_iterations):
        sample = goal_f if rng.random() < 0.18 else (rng.uniform(0, size - 1), rng.uniform(0, size - 1))
        nearest = min(range(len(nodes)), key=lambda index: math.dist(nodes[index], sample))
        new_node = _steer(nodes[nearest], sample, step_size)
        if segment_has_collision(passable, nodes[nearest], new_node):
            continue
        nodes.append(new_node)
        parents.append(nearest)
        new_index = len(nodes) - 1
        if math.dist(new_node, goal_f) <= goal_radius and not segment_has_collision(passable, new_node, goal_f):
            nodes.append(goal_f)
            parents.append(new_index)
            path = []
            index = len(nodes) - 1
            while index is not None:
                path.append(nodes[index])
                index = parents[index]
            path.reverse()
            return smooth_planner_result(path, passable)
    return None


ALGORITHMS = {
    "Dijkstra": strict_dijkstra,
    "A*": strict_astar,
    "DFS": strict_dfs,
    "RRT": strict_rrt,
}


def main():
    records = collect_records(["medium"])[1]
    stats = {name: {"attempts": 0, "path_found": 0, "success": 0, "collisions": 0} for name in ALGORITHMS}
    started = time.perf_counter()
    for record in records:
        layers, passable, _ = load_map_bundle(no_task_corner_maps_dir() / record["file"])
        start = tuple(record["start"])
        goal = tuple(record["goal"])
        for name, algorithm in ALGORITHMS.items():
            bucket = stats[name]
            bucket["attempts"] += 1
            try:
                path = algorithm(passable.copy(), clone_layers(layers), no_task_model_cost, start, goal)
            except Exception:
                path = None
            if not path:
                continue
            bucket["path_found"] += 1
            collision = calculate_collision_count(path, passable)
            bucket["collisions"] += int(collision > 0)
            bucket["success"] += int(collision == 0)

    print("Strict edge validation + production Catmull-Rom smoothing")
    print("Algorithm    Path found    Success    Collision failures    Success rate")
    for name, bucket in stats.items():
        attempts = bucket["attempts"]
        print(
            f"{name:<10} {bucket['path_found']:>10}/{attempts:<4} "
            f"{bucket['success']:>8}/{attempts:<4} {bucket['collisions']:>19} "
            f"{bucket['success'] / attempts if attempts else 0:.2%}"
        )
    print(f"Elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
