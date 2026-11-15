"""AO planner adapter."""

from __future__ import annotations

from uav_planner.costs import no_task_model_cost
from uav_planner.geometry import PathResult, collision_segment_count_for_path, smooth_planner_result
from uav_planner.metaheuristic import ao_optimize

from ..astar import astar
from ..common import seed_value


def ao_planner(passable, speed_map, cost_func, start, goal):
    """AO continuous waypoint planner adapter."""
    improved = cost_func is no_task_model_cost
    seed_path = astar(passable, speed_map, cost_func, start, goal)
    result = ao_optimize(
        passable,
        speed_map,
        cost_func,
        start,
        goal,
        seed_path=seed_path,
        random_seed=seed_value(start, goal, 8),
        feasibility_priority=improved,
        collision_safe_smoothing=improved,
    )
    planned = smooth_planner_result(result["path"], passable, used_fallback=False)
    if improved and collision_segment_count_for_path(passable, planned):
        return PathResult(result["path"], used_fallback=False)
    return planned
