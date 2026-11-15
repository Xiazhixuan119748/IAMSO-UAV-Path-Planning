"""Curvature-only ablation for traditional planners on the no-task dataset."""

from __future__ import annotations

from uav_planner.costs import no_task_model_cost
from uav_planner.experiments.no_task_corner.dataset import load_map_bundle, no_task_corner_maps_dir
from uav_planner.experiments.no_task_corner.run_traditional_constraint_ablation import path_constraints
from uav_planner.experiments.no_task_corner.run_traditional_corner_cut_ablation import (
    _strict_neighbors,
    strict_astar,
    strict_dijkstra,
    strict_rrt,
)
from uav_planner.experiments.no_task_corner.runner import clone_layers, collect_records
from uav_planner.geometry import smooth_planner_result
from uav_planner.metrics import calculate_collision_count


CURVATURE_THRESHOLDS = (2.0, 2.05, 2.10, 2.15, 2.17, 2.20, 2.25, 2.30, 2.40, 2.50)


def strict_bfs(passable, layers, cost_func, start, goal):
    queue = [start]
    read_index = 0
    visited = {start}
    parent = {}
    while read_index < len(queue):
        current = queue[read_index]
        read_index += 1
        if current == goal:
            break
        for neighbor in _strict_neighbors(passable, current):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = current
            queue.append(neighbor)

    if goal not in parent:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return smooth_planner_result(path, passable)


EXPERIMENT_ALGORITHMS = {
    "Dijkstra": strict_dijkstra,
    "A*": strict_astar,
    "BFS": strict_bfs,
    "RRT": strict_rrt,
}


def main():
    records = collect_records(["medium"])[1]
    stats = {
        name: {"collision_free": 0, "curvature": {limit: 0 for limit in CURVATURE_THRESHOLDS}}
        for name in EXPERIMENT_ALGORITHMS
    }
    for record in records:
        layers, passable, _ = load_map_bundle(no_task_corner_maps_dir() / record["file"])
        start, goal = tuple(record["start"]), tuple(record["goal"])
        for name, algorithm in EXPERIMENT_ALGORITHMS.items():
            try:
                path = algorithm(passable.copy(), clone_layers(layers), no_task_model_cost, start, goal)
            except Exception:
                path = None
            if not path or calculate_collision_count(path, passable) != 0:
                continue
            stats[name]["collision_free"] += 1
            _, max_curvature = path_constraints(path, passable)
            for limit in CURVATURE_THRESHOLDS:
                stats[name]["curvature"][limit] += int(max_curvature <= limit)

    print("Success = collision-free and max absolute curvature <= threshold")
    print("Algorithm    Collision-free  " + "  ".join(f"k<={limit:.1f}" for limit in CURVATURE_THRESHOLDS))
    for name, bucket in stats.items():
        print(
            f"{name:<10} {bucket['collision_free'] / len(records):>14.2%}  "
            + "  ".join(
                f"{bucket['curvature'][limit] / len(records):>6.2%}" for limit in CURVATURE_THRESHOLDS
            )
        )


if __name__ == "__main__":
    main()
