"""Benchmark evaluation helpers for the fixed test dataset."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from uav_planner.experiments.fixed_benchmark.dataset import (
    benchmark_maps_dir,
    benchmark_results_dir,
    load_index,
    load_map_bundle,
)
from uav_planner.metrics import calculate_extended_metrics
from uav_planner.registry import ALGORITHMS, COST_FUNCTIONS


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
        "task_reward": 0.0,
        "score": 0.0,
        "score_count": 0,
        "runtime": 0.0,
    }


def _safe_avg(total: float, count: int) -> float | None:
    return (total / count) if count else None


def _summarize_bucket(algorithm: str, bucket: dict[str, Any]) -> dict[str, Any]:
    attempts = int(bucket["attempts"])
    path_found = int(bucket["path_found"])
    success = int(bucket["success"])
    return {
        "algorithm": algorithm,
        "attempts": attempts,
        "path_found_rate": (bucket["path_found"] / attempts) if attempts else 0.0,
        "success_rate": (success / attempts) if attempts else 0.0,
        "no_path_count": int(bucket["no_path"]),
        "collision_failure_count": int(bucket["collision_failures"]),
        "exception_count": int(bucket["exception_count"]),
        "avg_distance": _safe_avg(bucket["distance"], success),
        "avg_time": _safe_avg(bucket["time"], success),
        "avg_safety": _safe_avg(bucket["safety"], success),
        "avg_collision": _safe_avg(bucket["collision"], path_found),
        "avg_smoothness": _safe_avg(bucket["smoothness"], success),
        "avg_task_reward": _safe_avg(bucket["task_reward"], success),
        "avg_score": _safe_avg(bucket["score"], int(bucket["score_count"])),
        "avg_runtime": _safe_avg(bucket["runtime"], attempts),
    }


def _format_value(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def _result_file_stem(cost_key: str, selected_difficulties: list[str]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cost_name = COST_FUNCTIONS[cost_key][0]
    cost_slug = {
        "最短距离": "distance",
        "最快时间": "time",
        "最安全": "safe",
        "模型综合代价": "model_cost",
    }.get(cost_name, f"cost_{cost_key}")
    difficulty_slug = "all" if len(selected_difficulties) > 1 else selected_difficulties[0]
    return f"baseline_all_algorithms_{difficulty_slug}_{cost_slug}_{timestamp}"


def _clone_passable(passable: np.ndarray) -> np.ndarray:
    return passable.copy()


def collect_records(difficulties: list[str] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = load_index()
    wanted = set(difficulties or [])
    records = [
        item
        for item in index["maps"]
        if not wanted or item["difficulty"] in wanted
    ]
    return index, records


def evaluate_dataset(
    cost_key: str = "4",
    difficulties: list[str] | None = None,
) -> dict[str, Any]:
    if cost_key not in COST_FUNCTIONS:
        raise KeyError(f"未注册的代价函数编号: {cost_key}")

    index, records = collect_records(difficulties)
    cost_name, cost_func = COST_FUNCTIONS[cost_key]
    selected_difficulties = sorted({item["difficulty"] for item in records})

    overall = {name: _new_bucket() for _, (name, _) in ALGORITHMS.items()}
    by_difficulty = {
        difficulty: {name: _new_bucket() for _, (name, _) in ALGORITHMS.items()}
        for difficulty in selected_difficulties
    }
    detailed_results: list[dict[str, Any]] = []

    total_runs = len(records) * len(ALGORITHMS)
    completed_runs = 0
    started_at = time.perf_counter()

    for map_index, record in enumerate(records, start=1):
        dataset_path = benchmark_maps_dir() / record["file"]
        layers, passable, meta = load_map_bundle(dataset_path)
        start = tuple(record["start"])
        goal = tuple(record["goal"])
        difficulty = record["difficulty"]

        print(
            f"\n地图 {map_index}/{len(records)}: {record['id']} "
            f"({record['difficulty_label']} {record['map_size']}x{record['map_size']})"
        )

        for _, (algo_name, algo_func) in ALGORITHMS.items():
            completed_runs += 1
            algo_layers = clone_layers(layers)
            algo_passable = _clone_passable(passable)

            run_started = time.perf_counter()
            status = "success"
            error_message = ""
            try:
                path = algo_func(algo_passable, algo_layers, cost_func, start, goal)
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
            task_reward = None
            rain_exposure = None
            inflation_exposure = None
            total_exposure = None
            score = None
            collision_count = None

            if path:
                bucket["path_found"] += 1
                diff_bucket["path_found"] += 1
                metrics = calculate_extended_metrics(path, algo_layers, passable=algo_passable, runtime=runtime)
                distance = metrics["distance"]
                travel_time = metrics["time"]
                safety = metrics["safety"]
                smoothness = metrics["smoothness"]
                task_reward = metrics["task_reward"]
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
                    bucket["task_reward"] += float(task_reward or 0.0)
                    diff_bucket["task_reward"] += float(task_reward or 0.0)
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
                    "cost_key": cost_key,
                    "cost_name": cost_name,
                    "status": status,
                    "success": 1 if status == "success" else 0,
                    "runtime": runtime,
                    "distance": distance,
                    "time": travel_time,
                    "safety": safety,
                    "collision_count": collision_count,
                    "smoothness": smoothness,
                    "task_reward": task_reward,
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
        "cost_key": cost_key,
        "cost_name": cost_name,
        "algorithms": [name for _, (name, _) in ALGORITHMS.items()],
        "summary": {
            "overall": overall_rows,
            "by_difficulty": difficulty_rows,
        },
        "details": detailed_results,
    }


def _write_json(path: Path, report: dict[str, Any]) -> None:
    serializable = json.loads(json.dumps(report, ensure_ascii=False, default=_json_default))
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法序列化的对象: {type(value)!r}")


def _write_summary_csv(path: Path, report: dict[str, Any]) -> None:
    fieldnames = [
        "scope",
        "algorithm",
        "attempts",
        "path_found_rate",
        "success_rate",
        "no_path_count",
        "collision_failure_count",
        "exception_count",
        "avg_distance",
        "avg_time",
        "avg_safety",
        "avg_collision",
        "avg_smoothness",
        "avg_task_reward",
        "avg_score",
        "avg_runtime",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["summary"]["overall"]:
            writer.writerow({"scope": "overall", **row})
        for difficulty, rows in report["summary"]["by_difficulty"].items():
            for row in rows:
                writer.writerow({"scope": difficulty, **row})


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
        "cost_key",
        "cost_name",
        "status",
        "success",
        "runtime",
        "distance",
        "time",
        "safety",
        "collision_count",
        "smoothness",
        "task_reward",
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


def _append_markdown_table(lines: list[str], rows: list[dict[str, Any]], title: str) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append(
        "| 算法 | 成功率 | 找到路径率 | 平均路径长度 | 平均通行时间 | 平均安全度 | 平均碰撞次数 | 平均平滑度 | 平均任务收益 | 平均综合评分/100 | 平均运行时间/s | 无路径次数 | 碰撞失败次数 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['algorithm']} | {_format_percent(row['success_rate'])} | {_format_percent(row['path_found_rate'])} | "
            f"{_format_value(row['avg_distance'])} | {_format_value(row['avg_time'])} | {_format_percent(row['avg_safety'])} | "
            f"{_format_value(row['avg_collision'])} | {_format_value(row['avg_smoothness'])} | "
            f"{_format_value(row['avg_task_reward'])} | {_format_value(row['avg_score'])} | {_format_value(row['avg_runtime'])} | "
            f"{row['no_path_count']} | {row['collision_failure_count']} |"
        )
    lines.append("")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 固定测试集全算法基线结果",
        "",
        f"- 数据集：`{report['dataset_name']}` v{report['dataset_version']}",
        f"- 地图数量：{report['map_count']}",
        f"- 难度范围：{', '.join(report['selected_difficulties'])}",
        f"- 代价函数：{report['cost_name']}",
        "- 统计口径：路径长度、通行时间、安全度、平滑度、任务收益按成功路径求平均；已规划路径中发生碰撞记为 1，不发生碰撞记为 0；综合评分为 0-100 分，按已规划路径求平均，碰撞路径会降低评分",
        f"- 地图目录：`{report['dataset_root']}`",
        "",
    ]
    _append_markdown_table(lines, report["summary"]["overall"], "总体结果")
    for difficulty, rows in report["summary"]["by_difficulty"].items():
        label = report["difficulty_labels"].get(difficulty, difficulty)
        _append_markdown_table(lines, rows, f"{label} 难度结果")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_report(report: dict[str, Any]) -> dict[str, str]:
    selected_difficulties = list(report["selected_difficulties"])
    stem = _result_file_stem(report["cost_key"], selected_difficulties)
    output_dir = benchmark_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{stem}.json"
    summary_csv_path = output_dir / f"{stem}_summary.csv"
    details_csv_path = output_dir / f"{stem}_details.csv"
    markdown_path = output_dir / f"{stem}.md"

    _write_json(json_path, report)
    _write_summary_csv(summary_csv_path, report)
    _write_details_csv(details_csv_path, report)
    _write_markdown(markdown_path, report)

    return {
        "json": str(json_path),
        "summary_csv": str(summary_csv_path),
        "details_csv": str(details_csv_path),
        "markdown": str(markdown_path),
    }
