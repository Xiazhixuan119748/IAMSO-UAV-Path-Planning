"""MSO planner adapter."""

from __future__ import annotations

from uav_planner.geometry import smooth_planner_result
from uav_planner.metaheuristic import mso_optimize

from ..astar import astar
from ..common import seed_value


def mso_planner(passable, speed_map, cost_func, start, goal):
    """MSO continuous waypoint planner adapter."""
    seed_path = astar(passable, speed_map, cost_func, start, goal)
    result = mso_optimize(
        passable,
        speed_map,
        cost_func,
        start,
        goal,
        seed_path=seed_path,
        random_seed=seed_value(start, goal, 7),
    )
    return smooth_planner_result(result["path"], passable, used_fallback=False)
