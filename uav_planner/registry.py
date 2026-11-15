"""Algorithm and cost-function registries."""

from __future__ import annotations

from uav_planner.pathfinding import (
    astar,
    dijkstra,
    distance_cost,
    model_cost,
    rrt,
    safe_cost,
    time_cost,
    mso_planner,
    ao_planner,
    iamso_planner,
)


ALGORITHMS = {
    "1": ("Dijkstra", dijkstra),
    "2": ("A*", astar),
    "4": ("RRT", rrt),
    "5": ("MSO", mso_planner),
    "6": ("AO", ao_planner),
    "7": ("IAMSO", iamso_planner),
}

COST_FUNCTIONS = {
    "1": ("最短距离", distance_cost),
    "2": ("最快时间", time_cost),
    "3": ("最安全", safe_cost),
    "4": ("模型综合代价", model_cost),
}


def generate_candidates(passable, speed_map, start, goal):
    """
    使用所有算法+代价函数组合生成候选路径。

    Returns:
        候选路径列表，每条路径只保留一次（去重）
    """
    seen = set()
    candidates = []

    for _, (_aname, afun) in ALGORITHMS.items():
        for _, (_cname, cfun) in COST_FUNCTIONS.items():
            path = afun(passable, speed_map, cfun, start, goal)
            if path is None:
                continue
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                candidates.append(path)

    return candidates
