"""Production IAMSO planner with NF, BI, and CA improvements."""

from __future__ import annotations

import heapq
import math

from uav_planner.config import MODEL_WEIGHTS, NO_TASK_MODEL_COST_WEIGHTS
from uav_planner.costs import distance_cost, model_cost, no_task_model_cost, safe_cost, time_cost
from uav_planner.geometry import (
    PathResult,
    collision_cells_for_path,
    collision_segment_count_for_path,
    smooth_planner_result,
)
from uav_planner.metaheuristic import iamso_optimize
from uav_planner.metaheuristic.collision_aware import collision_aware_initial_population

from ..astar import astar
from ..common import seed_value
from ..dijkstra import dijkstra
from ..rrt import rrt
from .seeds import build_bidirectional_seed_paths


_CARDINAL_MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NO_TASK_LAYER_WEIGHTS = tuple(
    NO_TASK_MODEL_COST_WEIGHTS[key]
    for key in ("inflation", "wind", "rain", "cruise")
)


def _path_is_collision_free(path, passable) -> bool:
    return bool(path) and not collision_cells_for_path(passable, path) and not collision_segment_count_for_path(
        passable, path
    )


def _cardinal_safe_path(passable, layers, cost_func, start, goal):
    """Find a grid-center path whose segments cannot cut diagonally through obstacle corners."""
    if not passable[start] or not passable[goal]:
        return None

    size = int(passable.shape[0])
    queue = [(0.0, start)]
    distance = {start: 0.0}
    parent = {}

    while queue:
        current_cost, current = heapq.heappop(queue)
        if current_cost > distance.get(current, math.inf):
            continue
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path

        row, col = current
        for dr, dc in _CARDINAL_MOVES:
            neighbor = (row + dr, col + dc)
            nr, nc = neighbor
            if not (0 <= nr < size and 0 <= nc < size) or not passable[neighbor]:
                continue
            step = float(cost_func(current, neighbor, layers))
            if not math.isfinite(step):
                continue
            new_cost = current_cost + max(step, 1e-9)
            if new_cost < distance.get(neighbor, math.inf):
                distance[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(queue, (new_cost, neighbor))
    return None


def _safe_fallback_path(passable, layers, cost_func, start, goal, seed_paths):
    for path in seed_paths:
        if _path_is_collision_free(path, passable):
            return path
    return _cardinal_safe_path(passable, layers, cost_func, start, goal)


def _seed_paths(passable, speed_map, cost_func, start, goal):
    """Build the original plan3.0 seed paths for IAMSO."""
    seed_specs = [
        (astar, cost_func),
        (astar, distance_cost),
        (astar, time_cost),
        (astar, safe_cost),
        (astar, model_cost),
        (dijkstra, model_cost),
        (rrt, model_cost),
    ]
    seed_paths = []
    seen = set()
    for algo_func, cost_func in seed_specs:
        try:
            path = algo_func(passable, speed_map, cost_func, start, goal)
        except Exception:
            path = None
        if not path:
            continue
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            seed_paths.append(path)
    return seed_paths


def iamso_planner(passable, speed_map, cost_func, start, goal):
    """Run the production IAMSO-NF-BI-CA algorithm under the IAMSO name."""
    seed_paths, seed_labels = build_bidirectional_seed_paths(
        passable, speed_map, cost_func, start, goal
    )
    base_weights = (
        _NO_TASK_LAYER_WEIGHTS if cost_func is no_task_model_cost else MODEL_WEIGHTS
    )
    result = iamso_optimize(
        passable,
        speed_map,
        cost_func,
        start,
        goal,
        seed_paths=seed_paths,
        fallback_path=None,
        base_weights=base_weights,
        random_seed=seed_value(start, goal, 17),
        initial_population_builder=collision_aware_initial_population,
    )
    path = result.get("path")
    if path is None:
        return None
    planned_path = smooth_planner_result(
        path,
        passable,
        used_fallback=False,
    )
    planned_path.solution_source = result.get("solution_source", "unknown")
    planned_path.feasible = bool(result.get("feasible", False))
    planned_path.seed_labels = seed_labels
    return planned_path
