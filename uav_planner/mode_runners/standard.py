"""Runners for interactive modes 1-4."""

from __future__ import annotations

import time as time_module

from uav_planner.config import MODEL_WEIGHTS
from uav_planner.geometry import collision_cells_for_path
from uav_planner.map_generator import MAP_GENERATION_LABELS, MAP_GENERATION_REACHABLE, generate_map
from uav_planner.metrics import calculate_extended_metrics, calculate_layer_cost_metrics, calculate_path_metrics
from uav_planner.output_paths import create_run_output_paths, difficulty_name_for_size
from uav_planner.pathfinding import make_weighted_cost
from uav_planner.registry import ALGORITHMS, COST_FUNCTIONS

from .common import plot_paths, print_result, print_table


def _fmt_optional(value, precision=4):
    return "-" if value is None else f"{value:.{precision}f}"


def _fmt_optional_percent(value):
    return "-" if value is None else f"{value:.2%}"


def _map_generation_text(map_generation_mode=None, map_generation_label=None):
    if map_generation_label:
        return map_generation_label
    return MAP_GENERATION_LABELS.get(map_generation_mode or MAP_GENERATION_REACHABLE, str(map_generation_mode))


def run_single_algorithm_mode(
    map_size,
    start,
    goal,
    algo_func,
    algo_name,
    cost_func,
    cost_name,
    map_generation_mode=None,
    map_generation_label=None,
    difficulty_name=None,
):
    """Run one algorithm once on one generated map."""
    layers, passable = generate_map(start, goal, size=map_size, generation_mode=map_generation_mode)
    result = _run_once(passable, layers, start, goal, algo_func, algo_name, cost_func, cost_name)
    output_paths = create_run_output_paths(
        algo_name,
        difficulty_name or difficulty_name_for_size(map_size),
    )

    print(f"\n地图生成完成 [{_map_generation_text(map_generation_mode, map_generation_label)} | {algo_name} + {cost_name}]")
    print_result(result)
    plot_paths(
        passable,
        layers,
        [result],
        start,
        goal,
        save_png=True,
        output_paths=output_paths,
    )
    print(f"本次运行生成的图片统一保存到: {output_paths.directory}")


def run_algorithm_comparison_mode(
    map_size,
    start,
    goal,
    cost_func,
    cost_name,
    map_generation_mode=None,
    map_generation_label=None,
):
    """Run every algorithm once on the same generated map."""
    layers, passable = generate_map(start, goal, size=map_size, generation_mode=map_generation_mode)
    results = [
        _run_once(passable, layers, start, goal, algo_func, algo_name, cost_func, cost_name)
        for _key, (algo_name, algo_func) in ALGORITHMS.items()
    ]

    print(f"\n地图生成完成 [{_map_generation_text(map_generation_mode, map_generation_label)} | 代价函数: {cost_name}]")
    print_table(results, "算法")
    plot_paths(passable, layers, results, start, goal)


def run_cost_comparison_mode(
    map_size,
    start,
    goal,
    algo_func,
    algo_name,
    map_generation_mode=None,
    map_generation_label=None,
):
    """Run one algorithm once for every cost function on the same generated map."""
    layers, passable = generate_map(start, goal, size=map_size, generation_mode=map_generation_mode)
    results = [
        _run_once(passable, layers, start, goal, algo_func, algo_name, cost_func, cost_name)
        for _key, (cost_name, cost_func) in COST_FUNCTIONS.items()
    ]

    print(f"\n地图生成完成 [{_map_generation_text(map_generation_mode, map_generation_label)} | 算法: {algo_name}]")
    print_table(results, "代价函数")
    plot_paths(passable, layers, results, start, goal)


def run_weighted_mode(
    map_size,
    start,
    goal,
    algo_func,
    algo_name,
    weights=MODEL_WEIGHTS,
    map_generation_mode=None,
    map_generation_label=None,
):
    """Run weighted optimization once on one generated map."""
    w_inf, w_wind, w_rain, w_cruise = weights
    optimization_start = time_module.perf_counter()
    layers, passable = generate_map(start, goal, size=map_size, generation_mode=map_generation_mode)
    generation_text = _map_generation_text(map_generation_mode, map_generation_label)

    path_dist = algo_func(passable, layers, COST_FUNCTIONS["1"][1], start, goal)
    path_time = algo_func(passable, layers, COST_FUNCTIONS["2"][1], start, goal)
    path_safe = algo_func(passable, layers, COST_FUNCTIONS["3"][1], start, goal)

    min_dist = calculate_path_metrics(path_dist, layers)[0] if path_dist else None
    min_time = calculate_path_metrics(path_time, layers)[1] if path_time else None
    max_safe = calculate_path_metrics(path_safe, layers)[2] if path_safe else None

    candidates = []
    seen = set()
    weighted_cost_func = make_weighted_cost(w_inf, w_wind, w_rain, w_cruise)
    candidate_costs = list(COST_FUNCTIONS.items()) + [("weighted", ("新指标加权综合", weighted_cost_func))]
    for _, (_cost_name, cost_func) in candidate_costs:
        path = algo_func(passable, layers, cost_func, start, goal)
        if path:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                candidates.append(path)

    if not candidates:
        runtime = time_module.perf_counter() - optimization_start
        result = {
            "path": None,
            "label": f"新指标加权最优(w={w_inf},{w_wind},{w_rain},{w_cruise})",
            "status": "未找到路径",
            "collision_points": [],
            "distance": None,
            "time": None,
            "safety": None,
            "collision": None,
            "smoothness": None,
            "runtime": runtime,
            "success_rate": 0.0,
            "task_reward": None,
            "rain_exposure": None,
            "inflation_exposure": None,
            "total_exposure": None,
            "score": None,
        }
        print(
            f"\n地图生成完成 [{generation_text} | 算法: {algo_name}, "
            f"权重: 膨胀={w_inf}, 风场={w_wind}, 降雨={w_rain}, 巡航={w_cruise}]"
        )
        print("该算法在当前地图和权重设置下没有生成候选路径，记为失败。")
        print_result(result)
        return

    scored = []
    for path in candidates:
        metrics = calculate_extended_metrics(path, layers, passable=passable, runtime=0.0)
        if metrics["score"] is None:
            continue
        layer_metrics = calculate_layer_cost_metrics(path, layers, weights)
        scored.append((metrics["score"], path, metrics, layer_metrics))

    if not scored:
        runtime = time_module.perf_counter() - optimization_start
        result = {
            "path": None,
            "label": f"新指标加权最优(w={w_inf},{w_wind},{w_rain},{w_cruise})",
            "status": "未找到可评价路径",
            "collision_points": [],
            "distance": None,
            "time": None,
            "safety": None,
            "collision": None,
            "smoothness": None,
            "runtime": runtime,
            "success_rate": 0.0,
            "task_reward": None,
            "rain_exposure": None,
            "inflation_exposure": None,
            "total_exposure": None,
            "score": None,
        }
        print(
            f"\n地图生成完成 [{generation_text} | 算法: {algo_name}, "
            f"权重: 膨胀={w_inf}, 风场={w_wind}, 降雨={w_rain}, 巡航={w_cruise}]"
        )
        print("该算法生成了候选路径，但没有可用于综合评分的有效路径，记为失败。")
        print_result(result)
        return

    scored.sort(key=lambda x: x[0], reverse=True)
    runtime = time_module.perf_counter() - optimization_start
    _best_score, best_path, _best_metrics, best_layer_metrics = scored[0]
    best_metrics = calculate_extended_metrics(best_path, layers, passable=passable, runtime=runtime)
    dist = best_metrics["distance"]
    travel_time = best_metrics["time"]
    safe = best_metrics["safety"]
    collision = best_metrics["collision"]
    smoothness = best_metrics["smoothness"]
    best_score = best_metrics["score"]
    result = {
        "path": best_path,
        "label": f"新指标加权最优(w={w_inf},{w_wind},{w_rain},{w_cruise})",
        "status": "成功" if collision == 0 else "碰撞失败",
        "collision_points": sorted(collision_cells_for_path(passable, best_path)),
        "distance": dist,
        "time": travel_time,
        "safety": safe,
        "collision": collision,
        "smoothness": smoothness,
        "runtime": runtime,
        "success_rate": 1.0 if collision == 0 else 0.0,
        "task_reward": best_metrics["task_reward"],
        "rain_exposure": best_metrics["rain_exposure"],
        "inflation_exposure": best_metrics["inflation_exposure"],
        "total_exposure": best_metrics["total_exposure"],
        "score": best_score,
    }

    print(
        f"\n地图生成完成 [{generation_text} | 算法: {algo_name}, 权重: "
        f"膨胀={w_inf}, 风场={w_wind}, 降雨={w_rain}, 巡航={w_cruise}]"
    )
    print(f"\n{'=' * 50}")
    print("基准值（参考）：")
    print(
        f"  最短距离 = {_fmt_optional(min_dist)}    "
        f"最短时间 = {_fmt_optional(min_time)}    "
        f"最高安全度 = {_fmt_optional_percent(max_safe)}"
    )
    print(f"{'=' * 50}")
    print("新指标加权最优路径：")
    print(f"  距离 = {dist:.4f}    时间 = {travel_time:.4f}    安全度 = {safe:.2%}    综合评分 = {best_score:.2f}/100")
    print(
        f"  碰撞次数 = {collision}    "
        f"平滑度 = {result['smoothness']:.4f}    "
        f"任务收益 = {result['task_reward']:.4f}    "
        f"运行时间 = {runtime:.4f}s    "
        f"成功率 = {result['success_rate']:.2%}"
    )
    print(
        f"  膨胀 = {best_layer_metrics['inflation']:.4f}    "
        f"风场 = {best_layer_metrics['wind']:.4f}    "
        f"降雨 = {best_layer_metrics['rain']:.4f}    "
        f"巡航损失 = {best_layer_metrics['cruise']:.4f}"
    )
    print(
        f"  连续雨区暴露 = {result['rain_exposure']:.4f}    "
        f"连续障碍膨胀暴露 = {result['inflation_exposure']:.4f}"
    )
    print(f"{'=' * 50}")

    plot_paths(passable, layers, [result], start, goal)


def _run_once(passable, layers, start, goal, algo_func, algo_name, cost_func, cost_name):
    from .common import run_single

    return run_single(passable, layers, start, goal, algo_func, algo_name, cost_func, cost_name)
