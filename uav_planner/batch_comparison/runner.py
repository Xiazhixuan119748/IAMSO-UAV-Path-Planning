"""Batch random-map comparison for mode 5."""

from __future__ import annotations

import random
import time as time_module
from datetime import datetime
from pathlib import Path

from uav_planner.map_generator import MAP_GENERATION_LABELS, MAP_GENERATION_REACHABLE, generate_map
from uav_planner.metrics import calculate_collision_count, calculate_extended_metrics
from uav_planner.registry import ALGORITHMS


def _batch_result_filename(map_mode: str, generation_label: str, timestamp: str) -> str:
    mode_label = str(map_mode).strip().replace("模式", "") or "未命名"
    generation_text = str(generation_label).strip().replace("地图", "") or "未命名生成"
    return f"算法批量比较结果_{mode_label}模式_{generation_text}_{timestamp}.md"


def get_batch_iterations(default=100):
    """获取批量随机地图实验次数。"""
    raw = input(f"请输入随机地图迭代次数（直接回车默认 {default} 次）: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"输入无效，使用默认 {default} 次。")
        return default
    return max(1, value)


def select_batch_map_size():
    """选择批量实验地图规模。"""
    options = {
        "1": ("简单", 30),
        "2": ("中等", 60),
        "3": ("困难", 90),
    }
    print("\n======== 选择随机地图难度 ========")
    print("  1. 简单模式 30×30")
    print("  2. 中等模式 60×60")
    print("  3. 困难模式 90×90")
    print("================================")

    while True:
        choice = input("请选择地图难度 (1/2/3): ").strip()
        if choice in options:
            name, size = options[choice]
            print(f"已选择: {name}模式 {size}×{size}")
            return name, size
        print("无效输入，请重新选择！")


def random_goal_for_batch(start, map_size, seed):
    """为批量实验随机生成终点。"""
    rng = random.Random(seed)
    min_distance = max(1, int(map_size * 0.35))
    while True:
        goal = (rng.randrange(map_size), rng.randrange(map_size))
        distance = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        if goal != start and distance >= min_distance:
            return goal


def _new_batch_stats():
    return {
        "attempts": 0,
        "path_found": 0,
        "success": 0,
        "distance": 0.0,
        "time": 0.0,
        "safety": 0.0,
        "collision": 0.0,
        "smoothness": 0.0,
        "task_reward": 0.0,
        "score": 0.0,
        "score_count": 0,
        "runtime": 0.0,
    }


def _format_metric(value):
    return "-" if value is None else f"{value:.4f}"


def _new_map_stats():
    return {
        "blocked_ratio": 0.0,
        "goal_distance": 0.0,
        "count": 0,
    }


def run_batch_comparison(
    start,
    cost_func,
    iterations=100,
    map_size=30,
    seed_base=20260707,
    generation_mode=MAP_GENERATION_REACHABLE,
):
    """每次生成一张随机地图，并让所有算法在同一张地图上运行。"""
    stats = {name: _new_batch_stats() for _, (name, _) in ALGORITHMS.items()}
    map_stats = _new_map_stats()

    for i in range(iterations):
        goal = random_goal_for_batch(start, map_size, seed_base + i)
        speed_map, passable = generate_map(
            start,
            goal,
            seed=seed_base + i,
            size=map_size,
            generation_mode=generation_mode,
        )
        map_stats["blocked_ratio"] += 1.0 - (passable.sum() / passable.size)
        map_stats["goal_distance"] += abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        map_stats["count"] += 1
        print(f"\r批量实验进度: {i + 1}/{iterations}", end="", flush=True)

        for _, (algo_name, algo_func) in ALGORITHMS.items():
            item = stats[algo_name]
            item["attempts"] += 1

            start_time = time_module.perf_counter()
            try:
                path = algo_func(passable, speed_map, cost_func, start, goal)
            except Exception as exc:
                print(f"\n{algo_name} 第 {i + 1} 轮运行失败: {exc}")
                path = None
            runtime = time_module.perf_counter() - start_time
            item["runtime"] += runtime

            if not path:
                continue

            item["path_found"] += 1
            collision_count = calculate_collision_count(path, passable)
            item["collision"] += 1 if collision_count > 0 else 0
            metrics = calculate_extended_metrics(path, speed_map, passable=passable, runtime=runtime)
            if metrics["score"] is not None:
                item["score"] += metrics["score"]
                item["score_count"] += 1
            if collision_count != 0:
                continue

            item["success"] += 1
            item["distance"] += metrics["distance"]
            item["time"] += metrics["time"]
            item["safety"] += metrics["safety"]
            item["smoothness"] += metrics["smoothness"]
            item["task_reward"] += metrics["task_reward"]

    print()

    rows = []
    for algo_name, item in stats.items():
        attempts = item["attempts"]
        path_found = item["path_found"]
        success = item["success"]
        rows.append({
            "algorithm": algo_name,
            "success_rate": success / attempts if attempts else 0,
            "avg_distance": item["distance"] / success if success else None,
            "avg_time": item["time"] / success if success else None,
            "avg_safety": item["safety"] / success if success else None,
            "avg_collision": item["collision"] / path_found if path_found else None,
            "avg_smoothness": item["smoothness"] / success if success else None,
            "avg_task_reward": item["task_reward"] / success if success else None,
            "avg_score": item["score"] / item["score_count"] if item["score_count"] else None,
            "avg_runtime": item["runtime"] / attempts if attempts else None,
        })
    if map_stats["count"]:
        map_stats["blocked_ratio"] /= map_stats["count"]
        map_stats["goal_distance"] /= map_stats["count"]
    return rows, map_stats


def print_batch_comparison_table(rows):
    """以表格形式打印批量实验结果。"""
    print("\n" + "=" * 158)
    print(
        f"{'算法':<18} {'成功率':>8} {'平均路径长度':>14} {'平均通行时间':>14} "
        f"{'平均安全度':>12} {'平均碰撞次数':>14} {'平均平滑度':>14} "
        f"{'平均任务收益':>14} {'平均综合评分/100':>16} {'平均运行时间/s':>16}"
    )
    print("-" * 158)
    for row in rows:
        safety_text = "-" if row["avg_safety"] is None else f"{row['avg_safety']:.2%}"
        print(
            f"{row['algorithm']:<18} "
            f"{row['success_rate']:>7.2%} "
            f"{_format_metric(row['avg_distance']):>14} "
            f"{_format_metric(row['avg_time']):>14} "
            f"{safety_text:>12} "
            f"{_format_metric(row['avg_collision']):>14} "
            f"{_format_metric(row['avg_smoothness']):>14} "
            f"{_format_metric(row['avg_task_reward']):>14} "
            f"{_format_metric(row['avg_score']):>16} "
            f"{_format_metric(row['avg_runtime']):>16}"
        )
    print("=" * 158)


def save_batch_comparison_markdown(
    rows,
    iterations,
    cost_name,
    map_mode,
    map_size,
    start,
    map_stats=None,
    output_path=None,
    generation_mode=MAP_GENERATION_REACHABLE,
    generation_label=None,
):
    """保存批量实验结果为 Markdown 表格。"""
    generation_label = generation_label or MAP_GENERATION_LABELS.get(generation_mode, str(generation_mode))
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("batch_results") / _batch_result_filename(map_mode, generation_label, timestamp)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 算法批量比较结果\n\n")
        f.write(f"- 随机地图迭代次数：{iterations}\n")
        f.write(f"- 地图模式：{map_mode}模式 {map_size}×{map_size}\n")
        f.write(f"- 地图生成方式：{generation_label}\n")
        f.write(f"- 固定起点：{start}\n")
        f.write("- 终点设置：每轮随机生成\n")
        f.write(f"- 代价函数：{cost_name}\n")
        if map_stats:
            f.write(f"- 平均障碍比例：{map_stats['blocked_ratio']:.2%}\n")
            f.write(f"- 平均起终点曼哈顿距离：{map_stats['goal_distance']:.2f}\n")
        f.write("- 说明：路径长度、通行时间、安全度、平滑度、任务收益为成功路径的平均值；碰撞次数按已规划路径二值统计，发生碰撞记为 1，不发生碰撞记为 0；综合评分为 0-100 分，按已规划路径统计，碰撞路径会降低评分；运行时间为全部迭代的平均值。\n\n")
        f.write("| 算法 | 成功率 | 平均路径长度 | 平均通行时间 | 平均安全度 | 平均碰撞次数 | 平均平滑度 | 平均任务收益 | 平均综合评分/100 | 平均运行时间/s |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            safety = "-" if row["avg_safety"] is None else f"{row['avg_safety']:.2%}"
            f.write(
                f"| {row['algorithm']} | {row['success_rate']:.2%} | "
                f"{_format_metric(row['avg_distance'])} | "
                f"{_format_metric(row['avg_time'])} | "
                f"{safety} | "
                f"{_format_metric(row['avg_collision'])} | "
                f"{_format_metric(row['avg_smoothness'])} | "
                f"{_format_metric(row['avg_task_reward'])} | "
                f"{_format_metric(row['avg_score'])} | "
                f"{_format_metric(row['avg_runtime'])} |\n"
            )
    return str(output_path)
