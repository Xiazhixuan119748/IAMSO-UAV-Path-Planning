"""Ablation test for selective smoothing of traditional grid planners.

This script leaves the production planners untouched. It temporarily disables
their post-processing smoothing, then applies local quadratic corner rounding
only at turns of 135 degrees or more before evaluating the final path.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from uav_planner.experiments.fixed_benchmark.dataset import load_map_bundle
from uav_planner.experiments.fixed_benchmark.runner import clone_layers, collect_records
from uav_planner.metrics import calculate_collision_count
from uav_planner.registry import COST_FUNCTIONS


TRADITIONAL = {
    "A*": "uav_planner.algorithms.astar.planner",
    "Dijkstra": "uav_planner.algorithms.dijkstra.planner",
    "DFS": "uav_planner.algorithms.dfs.planner",
    "RRT": "uav_planner.algorithms.rrt.planner",
}
ANGLE_THRESHOLD_DEG = 135.0
ROUNDING_FRACTION = 0.20
CURVE_SAMPLES = 10


def _identity_smoothing(path, passable, used_fallback=False):
    if path is None:
        return None
    return [(float(point[0]), float(point[1])) for point in path]


def _turn_angle(prev, point, next_point):
    v1 = (float(point[0]) - float(prev[0]), float(point[1]) - float(prev[1]))
    v2 = (float(next_point[0]) - float(point[0]), float(next_point[1]) - float(point[1]))
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 <= 1e-9 or n2 <= 1e-9:
        return 0.0
    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosine))


def selective_smooth_path(path):
    """Round only 135/180-degree corners; preserve all other turns."""
    points = [(float(point[0]), float(point[1])) for point in path or []]
    if len(points) < 3:
        return points

    output = [points[0]]
    current = points[0]
    for index in range(1, len(points) - 1):
        point = points[index]
        if _turn_angle(points[index - 1], point, points[index + 1]) < ANGLE_THRESHOLD_DEG:
            output.append(point)
            current = point
            continue

        previous = points[index - 1]
        following = points[index + 1]
        approach = (
            point[0] + ROUNDING_FRACTION * (previous[0] - point[0]),
            point[1] + ROUNDING_FRACTION * (previous[1] - point[1]),
        )
        departure = (
            point[0] + ROUNDING_FRACTION * (following[0] - point[0]),
            point[1] + ROUNDING_FRACTION * (following[1] - point[1]),
        )
        if math.dist(current, approach) > 1e-9:
            output.append(approach)
        for step in range(1, CURVE_SAMPLES + 1):
            t = step / CURVE_SAMPLES
            omt = 1.0 - t
            output.append(
                (
                    omt * omt * approach[0] + 2.0 * omt * t * point[0] + t * t * departure[0],
                    omt * omt * approach[1] + 2.0 * omt * t * point[1] + t * t * departure[1],
                )
            )
        current = departure

    output.append(points[-1])
    return output


def _patch_production_smoothing():
    modules = []
    for module_name in TRADITIONAL.values():
        module = __import__(module_name, fromlist=["smooth_planner_result"])
        modules.append((module, module.smooth_planner_result))
        module.smooth_planner_result = _identity_smoothing
    return modules


def _restore_production_smoothing(modules):
    for module, original in modules:
        module.smooth_planner_result = original


def run(cost_key="4", difficulties=None, dataset="fixed"):
    if cost_key not in COST_FUNCTIONS:
        raise KeyError(cost_key)
    if dataset == "no_task":
        from uav_planner.experiments.no_task_corner.dataset import (
            load_map_bundle as load_no_task_map_bundle,
            no_task_corner_maps_dir,
        )
        from uav_planner.experiments.no_task_corner.runner import collect_records as collect_no_task_records
        from uav_planner.costs import no_task_model_cost

        records = collect_no_task_records(difficulties)[1]
        map_dir = no_task_corner_maps_dir()
        cost_func = no_task_model_cost
    else:
        load_no_task_map_bundle = None
        records = collect_records(difficulties)[1]
        map_dir = Path(__file__).parent / "maps"
        cost_func = COST_FUNCTIONS[cost_key][1]
    algorithms = {}
    for name, module_name in TRADITIONAL.items():
        module = __import__(module_name, fromlist=["*"])
        function_name = {"A*": "astar", "Dijkstra": "dijkstra", "DFS": "dfs", "RRT": "rrt"}[name]
        algorithms[name] = getattr(module, function_name)

    stats = {name: {"attempts": 0, "path_found": 0, "success": 0, "collision": 0, "runtime": 0.0} for name in algorithms}
    patched = _patch_production_smoothing()
    try:
        for record in records:
            loader = load_no_task_map_bundle or load_map_bundle
            layers, passable, _ = loader(map_dir / record["file"])
            start = tuple(record["start"])
            goal = tuple(record["goal"])
            for name, algo in algorithms.items():
                bucket = stats[name]
                bucket["attempts"] += 1
                started = time.perf_counter()
                try:
                    raw_path = algo(passable.copy(), clone_layers(layers), cost_func, start, goal)
                    path = selective_smooth_path(raw_path) if raw_path else None
                except Exception:
                    path = None
                bucket["runtime"] += time.perf_counter() - started
                if not path:
                    continue
                bucket["path_found"] += 1
                collision = calculate_collision_count(path, passable)
                bucket["collision"] += int(collision > 0)
                if collision == 0:
                    bucket["success"] += 1
    finally:
        _restore_production_smoothing(patched)

    print(f"Selective smoothing threshold: >= {ANGLE_THRESHOLD_DEG:.1f} degrees")
    print("Algorithm    Path found    Success    Collision failures    Success rate")
    for name, bucket in stats.items():
        attempts = bucket["attempts"]
        print(
            f"{name:<10} {bucket['path_found']:>10}/{attempts:<4} "
            f"{bucket['success']:>8}/{attempts:<4} {bucket['collision']:>19} "
            f"{bucket['success'] / attempts if attempts else 0:.2%}"
        )
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-key", default="4", choices=sorted(COST_FUNCTIONS))
    parser.add_argument("--difficulty", nargs="*", choices=["simple", "medium", "hard"])
    parser.add_argument("--dataset", choices=["fixed", "no_task"], default="fixed")
    args = parser.parse_args()
    run(args.cost_key, args.difficulty, args.dataset)


if __name__ == "__main__":
    main()
