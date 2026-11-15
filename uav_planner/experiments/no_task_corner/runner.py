"""Benchmark runner for the no-task corner-to-corner dataset."""

from __future__ import annotations

import csv
import heapq
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from uav_planner.algorithms.common import (
    edge_cost,
    is_cell_available,
    iter_moves,
    map_size,
    reconstruct_path,
    seed_value,
)
from uav_planner.costmap import octile_distance
from uav_planner.experiments.no_task_corner.dataset import (
    COST_PROFILE,
    EVALUATION_PROFILE,
    load_index,
    load_map_bundle,
    no_task_corner_maps_dir,
    no_task_corner_results_dir,
)
from uav_planner.config import NO_TASK_EVALUATION_SCORE_WEIGHTS
from uav_planner.costs import no_task_model_cost
from uav_planner.geometry import segment_has_collision, smooth_planner_result
from uav_planner.metrics import calculate_extended_metrics
from uav_planner.registry import ALGORITHMS


TRADITIONAL_CORNER_PENALTY = 0.5


def clone_layers(layers: dict[str, Any]) -> dict[str, Any]:
    """Defensively copy map layers so each algorithm sees an isolated input."""
    cloned: dict[str, Any] = {}
    for key, value in layers.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def _clone_passable(passable: np.ndarray) -> np.ndarray:
    return passable.copy()


def _new_bucket() -> dict[str, Any]:
    return {
        "attempts": 0,
        "path_found": 0,
        "success": 0,
        "no_path": 0,
        "collision_failures": 0,
        "exception_count": 0,
        "distance": 0.0,
        "time": 0.0,
        "safety": 0.0,
        "collision": 0.0,
        "smoothness": 0.0,
        "rain_exposure": 0.0,
        "inflation_exposure": 0.0,
        "total_exposure": 0.0,
        "score": 0.0,
        "score_count": 0,
        "runtime": 0.0,
    }


def _safe_avg(total: float, count: int) -> float | None:
    return (total / count) if count else None


def _corner_risk(passable: np.ndarray, p: tuple[int, int], q: tuple[int, int]) -> int:
    if abs(p[0] - q[0]) != 1 or abs(p[1] - q[1]) != 1:
        return 0
    return int(not bool(passable[p[0], q[1]])) + int(not bool(passable[q[0], p[1]]))


def _penalized_cost(passable: np.ndarray, cost_func):
    def cost(p, q, layers):
        return float(cost_func(p, q, layers)) + TRADITIONAL_CORNER_PENALTY * _corner_risk(passable, p, q)

    return cost


def improved_dijkstra(passable, layers, cost_func, start, goal):
    pq = [(0.0, start)]
    dist = {start: 0.0}
    parent = {}
    size = map_size(passable)
    penalized_cost = _penalized_cost(passable, cost_func)

    while pq:
        cost, (x, y) = heapq.heappop(pq)
        if (x, y) == goal:
            break
        for dx, dy, _length in iter_moves():
            neighbor = (x + dx, y + dy)
            if not is_cell_available(passable, neighbor, size):
                continue
            new_cost = cost + edge_cost(penalized_cost, (x, y), neighbor, layers)
            if neighbor not in dist or new_cost < dist[neighbor]:
                dist[neighbor] = new_cost
                parent[neighbor] = (x, y)
                heapq.heappush(pq, (new_cost, neighbor))

    return smooth_planner_result(reconstruct_path(parent, start, goal), passable)


def improved_astar(passable, layers, cost_func, start, goal):
    pq = [(0.0, start)]
    g = {start: 0.0}
    parent = {}
    size = map_size(passable)
    penalized_cost = _penalized_cost(passable, cost_func)

    while pq:
        _, (x, y) = heapq.heappop(pq)
        if (x, y) == goal:
            break
        for dx, dy, _length in iter_moves():
            neighbor = (x + dx, y + dy)
            if not is_cell_available(passable, neighbor, size):
                continue
            new_g = g[(x, y)] + edge_cost(penalized_cost, (x, y), neighbor, layers)
            if neighbor not in g or new_g < g[neighbor]:
                g[neighbor] = new_g
                parent[neighbor] = (x, y)
                heapq.heappush(pq, (new_g + octile_distance(neighbor, goal), neighbor))

    return smooth_planner_result(reconstruct_path(parent, start, goal), passable)


def _nearest_node_index(nodes, target):
    return min(range(len(nodes)), key=lambda index: math.dist(nodes[index], target))


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


def improved_rrt(passable, layers, cost_func, start, goal):
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
        sample = goal_f if rng.random() < goal_sample_rate else (rng.uniform(0, size - 1), rng.uniform(0, size - 1))
        nearest_index = _nearest_node_index(nodes, sample)
        new_node = _steer(nodes[nearest_index], sample, step_size)

        row = int(round(new_node[0]))
        col = int(round(new_node[1]))
        if not (0 <= row < size and 0 <= col < size and passable[row, col]):
            continue
        if segment_has_collision(passable, nodes[nearest_index], new_node):
            continue

        nodes.append(new_node)
        parents.append(nearest_index)
        new_index = len(nodes) - 1

        if math.dist(new_node, goal_f) <= goal_radius and not segment_has_collision(passable, new_node, goal_f):
            nodes.append(goal_f)
            parents.append(new_index)
            return smooth_planner_result(_reconstruct_rrt(nodes, parents, len(nodes) - 1), passable)

    return None


MODE6_ALGORITHM_OVERRIDES = {
    "Dijkstra": improved_dijkstra,
    "A*": improved_astar,
    "RRT": improved_rrt,
}


def _summarize_bucket(algorithm: str, bucket: dict[str, Any]) -> dict[str, Any]:
    attempts = int(bucket["attempts"])
    path_found = int(bucket["path_found"])
    success = int(bucket["success"])
    return {
        "algorithm": algorithm,
        "attempts": attempts,
        "path_found_rate": (path_found / attempts) if attempts else 0.0,
        "success_rate": (success / attempts) if attempts else 0.0,
        "no_path_count": int(bucket["no_path"]),
        "collision_failure_count": int(bucket["collision_failures"]),
        "exception_count": int(bucket["exception_count"]),
        "avg_distance": _safe_avg(bucket["distance"], success),
        "avg_time": _safe_avg(bucket["time"], success),
        "avg_safety": _safe_avg(bucket["safety"], success),
        "avg_collision": _safe_avg(bucket["collision"], path_found),
        "avg_smoothness": _safe_avg(bucket["smoothness"], success),
        "avg_rain_exposure": _safe_avg(bucket["rain_exposure"], success),
        "avg_inflation_exposure": _safe_avg(bucket["inflation_exposure"], success),
        "avg_total_exposure": _safe_avg(bucket["total_exposure"], success),
        "avg_score": _safe_avg(bucket["score"], int(bucket["score_count"])),
        "avg_runtime": _safe_avg(bucket["runtime"], attempts),
    }


def collect_records(
    difficulties: list[str] | None = None,
    index_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = load_index(index_path)
    wanted = set(difficulties or [])
    records = [
        item
        for item in index["maps"]
        if not wanted or item["difficulty"] in wanted
    ]
    return index, records


def _resolve_algorithms(algorithm_names: list[str] | None = None):
    registered = [(name, MODE6_ALGORITHM_OVERRIDES.get(name, func)) for name, func in ALGORITHMS.values()]
    if not algorithm_names:
        return registered

    wanted = set(algorithm_names)
    selected = [(name, func) for name, func in registered if name in wanted]
    missing = wanted - {name for name, _ in selected}
    if missing:
        raise ValueError(f"Unknown algorithms: {', '.join(sorted(missing))}")
    return selected


def evaluate_dataset(
    difficulties: list[str] | None = None,
    algorithm_names: list[str] | None = None,
    record_ids: set[str] | None = None,
    dataset_dir: str | Path | None = None,
) -> dict[str, Any]:
    maps_root = Path(dataset_dir) if dataset_dir is not None else no_task_corner_maps_dir()
    if dataset_dir is None:
        index, records = collect_records(difficulties)
    else:
        index, records = collect_records(difficulties, maps_root / "index.json")
    if record_ids is not None:
        records = [record for record in records if record["id"] in record_ids]
    selected_difficulties = sorted({item["difficulty"] for item in records})
    selected_algorithms = _resolve_algorithms(algorithm_names)

    overall = {name: _new_bucket() for name, _ in selected_algorithms}
    by_difficulty = {
        difficulty: {name: _new_bucket() for name, _ in selected_algorithms}
        for difficulty in selected_difficulties
    }
    detailed_results: list[dict[str, Any]] = []

    total_runs = len(records) * len(selected_algorithms)
    completed_runs = 0
    started_at = time.perf_counter()

    for map_index, record in enumerate(records, start=1):
        dataset_path = maps_root / record["file"]
        layers, passable, meta = load_map_bundle(dataset_path)
        start = tuple(record["start"])
        goal = tuple(record["goal"])
        difficulty = record["difficulty"]

        if layers.get("tasks"):
            raise RuntimeError(f"{record['id']} is not a no-task map: tasks={layers['tasks']}")

        print(
            f"\n地图 {map_index}/{len(records)}: {record['id']} "
            f"({record['difficulty_label']} {record['map_size']}x{record['map_size']})"
        )

        for algo_name, algo_func in selected_algorithms:
            completed_runs += 1
            algo_layers = clone_layers(layers)
            algo_passable = _clone_passable(passable)

            run_started = time.perf_counter()
            status = "success"
            error_message = ""
            try:
                path = algo_func(algo_passable, algo_layers, no_task_model_cost, start, goal)
            except Exception as exc:
                path = None
                status = "exception"
                error_message = str(exc)
            runtime = time.perf_counter() - run_started

            bucket = overall[algo_name]
            diff_bucket = by_difficulty[difficulty][algo_name]
            bucket["attempts"] += 1
            diff_bucket["attempts"] += 1
            bucket["runtime"] += runtime
            diff_bucket["runtime"] += runtime

            distance = None
            travel_time = None
            safety = None
            smoothness = None
            rain_exposure = None
            inflation_exposure = None
            total_exposure = None
            score = None
            collision_count = None
            used_seed_fallback = False
            solution_source = ""

            if path:
                used_seed_fallback = bool(getattr(path, "used_fallback", False))
                solution_source = str(getattr(path, "solution_source", ""))
                bucket["path_found"] += 1
                diff_bucket["path_found"] += 1
                metrics = calculate_extended_metrics(
                    path,
                    algo_layers,
                    passable=algo_passable,
                    runtime=runtime,
                    include_task_reward=False,
                    score_weights=NO_TASK_EVALUATION_SCORE_WEIGHTS,
                )
                distance = metrics["distance"]
                travel_time = metrics["time"]
                safety = metrics["safety"]
                smoothness = metrics["smoothness"]
                rain_exposure = metrics["rain_exposure"]
                inflation_exposure = metrics["inflation_exposure"]
                total_exposure = metrics["total_exposure"]
                score = metrics["score"]
                collision_count = metrics["collision"]
                collision_flag = 1 if collision_count > 0 else 0
                bucket["collision"] += collision_flag
                diff_bucket["collision"] += collision_flag
                if score is not None:
                    bucket["score"] += float(score)
                    diff_bucket["score"] += float(score)
                    bucket["score_count"] += 1
                    diff_bucket["score_count"] += 1

                if collision_count == 0:
                    bucket["success"] += 1
                    diff_bucket["success"] += 1
                    bucket["distance"] += float(distance or 0.0)
                    diff_bucket["distance"] += float(distance or 0.0)
                    bucket["time"] += float(travel_time or 0.0)
                    diff_bucket["time"] += float(travel_time or 0.0)
                    bucket["safety"] += float(safety or 0.0)
                    diff_bucket["safety"] += float(safety or 0.0)
                    bucket["smoothness"] += float(smoothness or 0.0)
                    diff_bucket["smoothness"] += float(smoothness or 0.0)
                    bucket["rain_exposure"] += float(rain_exposure or 0.0)
                    diff_bucket["rain_exposure"] += float(rain_exposure or 0.0)
                    bucket["inflation_exposure"] += float(inflation_exposure or 0.0)
                    diff_bucket["inflation_exposure"] += float(inflation_exposure or 0.0)
                    bucket["total_exposure"] += float(total_exposure or 0.0)
                    diff_bucket["total_exposure"] += float(total_exposure or 0.0)
                else:
                    status = "collision"
                    bucket["collision_failures"] += 1
                    diff_bucket["collision_failures"] += 1
            else:
                if status == "exception":
                    bucket["exception_count"] += 1
                    diff_bucket["exception_count"] += 1
                else:
                    status = "no_path"
                    bucket["no_path"] += 1
                    diff_bucket["no_path"] += 1

            detailed_results.append(
                {
                    "map_id": record["id"],
                    "file": record["file"],
                    "difficulty": difficulty,
                    "difficulty_label": record["difficulty_label"],
                    "map_size": record["map_size"],
                    "seed": record["seed"],
                    "start": start,
                    "goal": goal,
                    "algorithm": algo_name,
                    "cost_profile": COST_PROFILE,
                    "evaluation_profile": EVALUATION_PROFILE,
                    "status": status,
                    "success": 1 if status == "success" else 0,
                    "runtime": runtime,
                    "distance": distance,
                    "time": travel_time,
                    "safety": safety,
                    "collision_count": collision_count,
                    "used_seed_fallback": used_seed_fallback,
                    "solution_source": solution_source,
                    "smoothness": smoothness,
                    "rain_exposure": rain_exposure,
                    "inflation_exposure": inflation_exposure,
                    "total_exposure": total_exposure,
                    "score": score,
                    "error_message": error_message,
                }
            )

            elapsed = time.perf_counter() - started_at
            collision_text = "-" if collision_count is None else str(collision_count)
            print(
                f"  [{completed_runs:>3}/{total_runs}] {algo_name:<8} "
                f"状态={status:<9} 碰撞={collision_text:<3} 用时={runtime:>7.3f}s "
                f"累计耗时={elapsed:>8.1f}s"
            )

    overall_rows = [_summarize_bucket(name, bucket) for name, bucket in overall.items()]
    difficulty_rows = {
        difficulty: [_summarize_bucket(name, bucket) for name, bucket in diff_stats.items()]
        for difficulty, diff_stats in by_difficulty.items()
    }

    return {
        "dataset_name": index["dataset_name"],
        "dataset_version": index["dataset_version"],
        "dataset_root": index["root"],
        "map_count": len(records),
        "selected_difficulties": selected_difficulties,
        "difficulty_labels": {
            key: index["difficulty_specs"][key]["label"]
            for key in selected_difficulties
        },
        "target_obstacle_ratios": {
            key: index["difficulty_specs"][key]["target_obstacle_ratio"]
            for key in selected_difficulties
        },
        "value_mode": index["value_mode"],
        "start_rule": index["start_rule"],
        "goal_rule": index["goal_rule"],
        "cost_profile": COST_PROFILE,
        "evaluation_profile": EVALUATION_PROFILE,
        "algorithms": [name for name, _ in selected_algorithms],
        "summary": {
            "overall": overall_rows,
            "by_difficulty": difficulty_rows,
        },
        "details": detailed_results,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize object: {type(value)!r}")


def _write_json(path: Path, report: dict[str, Any]) -> None:
    serializable = json.loads(json.dumps(report, ensure_ascii=False, default=_json_default))
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary_csv(path: Path, report: dict[str, Any]) -> None:
    fieldnames = [
        "algorithm",
        "success_rate",
        "avg_distance",
        "avg_time",
        "avg_safety",
        "avg_collision",
        "avg_score",
        "avg_runtime",
        "collision_failure_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["summary"]["overall"]:
            writer.writerow({key: row[key] for key in fieldnames})


def _write_details_csv(path: Path, report: dict[str, Any]) -> None:
    fieldnames = [
        "map_id",
        "file",
        "difficulty",
        "difficulty_label",
        "map_size",
        "seed",
        "start",
        "goal",
        "algorithm",
        "cost_profile",
        "evaluation_profile",
        "status",
        "success",
        "runtime",
        "distance",
        "time",
        "safety",
        "collision_count",
        "used_seed_fallback",
        "solution_source",
        "smoothness",
        "rain_exposure",
        "inflation_exposure",
        "total_exposure",
        "score",
        "error_message",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["details"]:
            writer.writerow(row)


def _format_value(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def print_comparison_table(report: dict[str, Any]) -> None:
    """Print the overall benchmark summary in the same style as mode 5."""
    rows = report["summary"]["overall"]
    print("\n" + "=" * 148)
    print(
        f"{'算法':<18} {'成功率':>8} {'平均路径长度':>14} "
        f"{'平均通行时间':>14} {'平均安全度':>12} {'平均碰撞次数':>14} "
        f"{'平均综合评分/100':>16} {'平均运行时间/s':>16} {'碰撞失败次数':>14}"
    )
    print("-" * 148)
    for row in rows:
        print(
            f"{row['algorithm']:<18} "
            f"{_format_percent(row['success_rate']):>8} "
            f"{_format_value(row['avg_distance']):>14} "
            f"{_format_value(row['avg_time']):>14} "
            f"{_format_percent(row['avg_safety']):>12} "
            f"{_format_value(row['avg_collision']):>14} "
            f"{_format_value(row['avg_score']):>16} "
            f"{_format_value(row['avg_runtime']):>16} "
            f"{row['collision_failure_count']:>14}"
        )
    print("=" * 148)


def _append_markdown_table(lines: list[str], rows: list[dict[str, Any]], title: str) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append(
        "| 算法 | 成功率 | 平均路径长度 | 平均通行时间 | 平均安全度 | "
        "平均碰撞次数 | 平均综合评分/100 | 平均运行时间/s | 碰撞失败次数 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['algorithm']} | {_format_percent(row['success_rate'])} | "
            f"{_format_value(row['avg_distance'])} | {_format_value(row['avg_time'])} | "
            f"{_format_percent(row['avg_safety'])} | {_format_value(row['avg_collision'])} | "
            f"{_format_value(row['avg_score'])} | {_format_value(row['avg_runtime'])} | "
            f"{row['collision_failure_count']} |"
        )
    lines.append("")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    algorithms = list(report["algorithms"])
    title = (
        f"# {algorithms[0]} 单算法固定地图实验结果"
        if len(algorithms) == 1
        else "# 左上到右下无任务价值批量比较结果"
    )
    lines = [
        title,
        "",
        f"- 数据集：`{report['dataset_name']}` v{report['dataset_version']}",
        f"- 算法：{', '.join(algorithms)}",
        f"- 地图数量：{report['map_count']}",
        f"- 难度范围：{', '.join(report['selected_difficulties'])}",
        "- 目标障碍比例：" + ", ".join(
            f"{key}={value:.0%}"
            for key, value in report["target_obstacle_ratios"].items()
        ),
        f"- value_mode：`{report['value_mode']}`",
        f"- start_rule：`{report['start_rule']}`",
        f"- goal_rule：`{report['goal_rule']}`",
        f"- cost_profile：`{report['cost_profile']}`",
        f"- evaluation_profile：`{report['evaluation_profile']}`",
        "- 说明：本基准不含任务价值点，不输出 task_reward，综合评分使用 no-task 权重。",
        f"- 地图目录：`{report['dataset_root']}`",
    ]
    ablation = report.get("ablation")
    if ablation:
        lines.extend(
            [
                f"- 消融配置：`{ablation['name']}`",
                f"- 原始安全路径回退：{'启用' if ablation['fallback_path_enabled'] else '禁用'}",
                f"- 多源种子：{'启用' if ablation['seed_paths_enabled'] else '禁用'}",
            ]
        )
    lines.append("")
    _append_markdown_table(lines, report["summary"]["overall"], "总体结果")
    for difficulty, rows in report["summary"]["by_difficulty"].items():
        label = report["difficulty_labels"].get(difficulty, difficulty)
        _append_markdown_table(lines, rows, f"{label} 难度结果")
    path.write_text("\n".join(lines), encoding="utf-8")


def _result_file_stem(
    selected_difficulties: list[str],
    timestamp: str | None = None,
    algorithm_names: list[str] | None = None,
) -> str:
    difficulty_names = {
        "simple": "简单",
        "medium": "中等",
        "hard": "困难",
    }
    if len(selected_difficulties) == 1:
        difficulty_label = difficulty_names.get(selected_difficulties[0], selected_difficulties[0])
    else:
        difficulty_label = "全部难度"
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if algorithm_names and len(algorithm_names) == 1:
        return f"单算法固定地图实验_{algorithm_names[0]}_{difficulty_label}模式_无任务价值_{timestamp}"
    return f"算法批量比较结果_{difficulty_label}模式_固定地图无任务价值_{timestamp}"


def save_report(report: dict[str, Any]) -> dict[str, str]:
    selected_difficulties = list(report["selected_difficulties"])
    stem = _result_file_stem(
        selected_difficulties,
        algorithm_names=list(report.get("algorithms", [])),
    )
    output_dir = no_task_corner_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / f"{stem}.md"

    _write_markdown(markdown_path, report)

    return {"实验报告 Markdown": str(markdown_path)}
