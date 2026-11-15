"""Run a hard-35%-reachable IAMSO collision-weight sweep on a fixed 100-map dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from uav_planner.experiments.fixed_benchmark.dataset import (
    benchmark_results_dir,
    experiment_root,
    fixed_goal_for_size,
    fixed_start_for_size,
    load_map_bundle,
    save_map_bundle,
)
from uav_planner.experiments.fixed_benchmark.map_generator import generate_benchmark_map
from uav_planner.experiments.fixed_benchmark.runner import clone_layers
from uav_planner.algorithms.common import seed_value
from uav_planner.algorithms.iamso.planner import _seed_paths
from uav_planner.costs import model_cost
from uav_planner.geometry import smooth_planner_result
from uav_planner.metaheuristic.iamso import iamso_optimize
from uav_planner.metrics import calculate_extended_metrics


DATASET_NAME = "iamso_collision_weight_hard35_reachable_100"
DATASET_VERSION = "1.0"
DATASET_SIZE = 90
DATASET_COUNT = 100
TARGET_OBSTACLE_RATIO = 0.35
SAFETY_RADIUS = 4
SEED_BASE = 2026073900

SWEEP_CONFIGS = [
    {"name": "baseline_1000", "label": "基准 1000", "collision_weight": 1000.0},
    {"name": "cw_2000", "label": "碰撞惩罚 2000", "collision_weight": 2000.0},
    {"name": "cw_3000", "label": "碰撞惩罚 3000", "collision_weight": 3000.0},
    {"name": "cw_4000", "label": "碰撞惩罚 4000", "collision_weight": 4000.0},
    {
        "name": "cw_strict_1e12",
        "label": "严格避碰 1e12(近似无穷大)",
        "collision_weight": 1e12,
    },
]


def dataset_dir() -> Path:
    return experiment_root() / "datasets" / DATASET_NAME


def dataset_index_path() -> Path:
    return dataset_dir() / "index.json"


def build_record(order: int) -> dict[str, Any]:
    start = fixed_start_for_size(DATASET_SIZE)
    goal = fixed_goal_for_size(DATASET_SIZE)
    return {
        "id": f"hard35_{order:03d}",
        "file": f"hard35_{order:03d}.npz",
        "difficulty": "hard35",
        "difficulty_label": "困难35%可达",
        "map_size": DATASET_SIZE,
        "seed": SEED_BASE + order,
        "start": list(start),
        "goal": list(goal),
        "target_obstacle_ratio": TARGET_OBSTACLE_RATIO,
        "safety_radius": SAFETY_RADIUS,
    }


def dataset_metadata() -> dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(dataset_dir().relative_to(experiment_root().parents[2])).replace("\\", "/"),
        "difficulty": "hard35",
        "difficulty_label": "困难35%可达",
        "map_size": DATASET_SIZE,
        "map_count": DATASET_COUNT,
        "start_rule": "fixed_center",
        "goal_rule": "fixed_upper_right_quadrant_with_margin",
        "target_obstacle_ratio": TARGET_OBSTACLE_RATIO,
        "safety_radius": SAFETY_RADIUS,
        "connectivity_protected": True,
        "maps": [],
    }


def ensure_dataset(force: bool = False) -> dict[str, Any]:
    root = dataset_dir()
    index_path = dataset_index_path()
    if force and root.exists():
        shutil.rmtree(root)

    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        existing = list(root.glob("*.npz"))
        if len(existing) == int(index.get("map_count", 0)) == DATASET_COUNT:
            return index

    root.mkdir(parents=True, exist_ok=True)
    index = dataset_metadata()

    for order in range(1, DATASET_COUNT + 1):
        record = build_record(order)
        layers, passable, obstacle_meta = generate_benchmark_map(
            start=tuple(record["start"]),
            goal=tuple(record["goal"]),
            seed=int(record["seed"]),
            size=int(record["map_size"]),
            target_ratio=float(record["target_obstacle_ratio"]),
            safety_radius=int(record["safety_radius"]),
        )
        record["obstacle_meta"] = obstacle_meta
        summary = save_map_bundle(root / record["file"], layers, passable, record)
        index["maps"].append(summary)
        print(
            f"[dataset] {order:>3}/{DATASET_COUNT} {record['id']} "
            f"障碍占比={summary['blocked_ratio']:.2%}"
        )

    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def _new_bucket(label: str) -> dict[str, Any]:
    return {
        "name": label,
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


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"无法序列化的对象: {type(value)!r}")


def _clone_passable(passable):
    return passable.copy()


def _run_iamso_variant(passable, layers, start, goal, seed_paths, collision_weight: float):
    result = iamso_optimize(
        passable,
        layers,
        model_cost,
        start,
        goal,
        seed_paths=seed_paths,
        random_seed=seed_value(start, goal, 17),
        collision_weight=collision_weight,
    )
    return smooth_planner_result(result["path"], passable, used_fallback=False)


def _evaluate_one_map(record: dict[str, Any]) -> list[dict[str, Any]]:
    layers, passable, meta = load_map_bundle(dataset_dir() / record["file"])
    start = tuple(record["start"])
    goal = tuple(record["goal"])
    seed_paths = _seed_paths(
        _clone_passable(passable), clone_layers(layers), model_cost, start, goal
    )

    rows: list[dict[str, Any]] = []
    for config in SWEEP_CONFIGS:
        run_layers = clone_layers(layers)
        run_passable = _clone_passable(passable)
        runtime = 0.0
        status = "success"
        error_message = ""
        try:
            started = time.perf_counter()
            path = _run_iamso_variant(
                run_passable,
                run_layers,
                start,
                goal,
                seed_paths,
                float(config["collision_weight"]),
            )
            runtime = time.perf_counter() - started
        except Exception as exc:
            path = None
            runtime = time.perf_counter() - started
            status = "exception"
            error_message = str(exc)

        distance = None
        travel_time = None
        safety = None
        collision_count = None
        smoothness = None
        task_reward = None
        rain_exposure = None
        inflation_exposure = None
        total_exposure = None
        score = None

        if path:
            metrics = calculate_extended_metrics(path, run_layers, passable=run_passable, runtime=runtime)
            distance = metrics["distance"]
            travel_time = metrics["time"]
            safety = metrics["safety"]
            collision_count = metrics["collision"]
            smoothness = metrics["smoothness"]
            task_reward = metrics["task_reward"]
            rain_exposure = metrics["rain_exposure"]
            inflation_exposure = metrics["inflation_exposure"]
            total_exposure = metrics["total_exposure"]
            score = metrics["score"]
            if collision_count and collision_count > 0:
                status = "collision"
        elif status != "exception":
            status = "no_path"

        rows.append(
            {
                "map_id": record["id"],
                "file": record["file"],
                "difficulty": meta["difficulty"],
                "difficulty_label": meta["difficulty_label"],
                "map_size": meta["map_size"],
                "seed": meta["seed"],
                "start": start,
                "goal": goal,
                "config_name": str(config["name"]),
                "config_label": str(config["label"]),
                "collision_weight": float(config["collision_weight"]),
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
    return rows


def _summarize(name: str, label: str, collision_weight: float, bucket: dict[str, Any]) -> dict[str, Any]:
    attempts = int(bucket["attempts"])
    path_found = int(bucket["path_found"])
    success = int(bucket["success"])
    return {
        "config_name": name,
        "config_label": label,
        "collision_weight": collision_weight,
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
        "avg_task_reward": _safe_avg(bucket["task_reward"], success),
        "avg_score": _safe_avg(bucket["score"], int(bucket["score_count"])),
        "avg_runtime": _safe_avg(bucket["runtime"], attempts),
    }


def evaluate_sweep(index: dict[str, Any], max_workers: int) -> dict[str, Any]:
    records = list(index["maps"])
    buckets = {
        str(config["name"]): _new_bucket(str(config["label"]))
        for config in SWEEP_CONFIGS
    }
    details: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_evaluate_one_map, record): record["id"]
            for record in records
        }
        completed = 0
        for future in as_completed(futures):
            map_id = futures[future]
            rows = future.result()
            completed += 1
            details.extend(rows)

            for row in rows:
                bucket = buckets[row["config_name"]]
                bucket["attempts"] += 1
                bucket["runtime"] += float(row["runtime"])
                if row["status"] == "exception":
                    bucket["exception_count"] += 1
                    continue
                if row["status"] == "no_path":
                    bucket["no_path"] += 1
                    continue

                bucket["path_found"] += 1
                collision_flag = 1 if (row["collision_count"] or 0) > 0 else 0
                bucket["collision"] += collision_flag
                if row["score"] is not None:
                    bucket["score"] += float(row["score"])
                    bucket["score_count"] += 1

                if row["status"] == "success":
                    bucket["success"] += 1
                    bucket["distance"] += float(row["distance"] or 0.0)
                    bucket["time"] += float(row["time"] or 0.0)
                    bucket["safety"] += float(row["safety"] or 0.0)
                    bucket["smoothness"] += float(row["smoothness"] or 0.0)
                    bucket["task_reward"] += float(row["task_reward"] or 0.0)
                elif row["status"] == "collision":
                    bucket["collision_failures"] += 1

            elapsed = time.perf_counter() - started_at
            print(f"[eval] {completed:>3}/{len(records)} {map_id} 完成，累计耗时 {elapsed:>8.1f}s")

    summary_rows = []
    for config in SWEEP_CONFIGS:
        summary_rows.append(
            _summarize(
                str(config["name"]),
                str(config["label"]),
                float(config["collision_weight"]),
                buckets[str(config["name"])],
            )
        )

    baseline = next(row for row in summary_rows if row["config_name"] == "baseline_1000")
    for row in summary_rows:
        base_success = float(baseline["success_rate"] or 0.0)
        base_score = baseline["avg_score"]
        base_distance = baseline["avg_distance"]
        base_time = baseline["avg_time"]
        row["delta_success_pp"] = (float(row["success_rate"] or 0.0) - base_success) * 100.0
        row["delta_score_pct"] = (
            None if not base_score else ((float(row["avg_score"] or 0.0) - float(base_score)) / float(base_score)) * 100.0
        )
        row["delta_distance_pct"] = (
            None
            if not base_distance or row["avg_distance"] is None
            else ((float(row["avg_distance"]) - float(base_distance)) / float(base_distance)) * 100.0
        )
        row["delta_time_pct"] = (
            None
            if not base_time or row["avg_time"] is None
            else ((float(row["avg_time"]) - float(base_time)) / float(base_time)) * 100.0
        )

    return {
        "dataset_name": index["dataset_name"],
        "dataset_version": index["dataset_version"],
        "dataset_root": index["root"],
        "map_count": len(records),
        "difficulty": index["difficulty"],
        "difficulty_label": index["difficulty_label"],
        "map_size": index["map_size"],
        "target_obstacle_ratio": index["target_obstacle_ratio"],
        "configs": SWEEP_CONFIGS,
        "summary": summary_rows,
        "details": details,
    }


def _format_value(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def _format_delta(value: float | None, digits: int = 2, suffix: str = "%") -> str:
    if value is None:
        return "-"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{digits}f}{suffix}"


def save_report(report: dict[str, Any]) -> dict[str, str]:
    output_dir = benchmark_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"iamso_collision_weight_hard35_reachable100_{timestamp}"

    json_path = output_dir / f"{stem}.json"
    summary_csv_path = output_dir / f"{stem}_summary.csv"
    details_csv_path = output_dir / f"{stem}_details.csv"
    markdown_path = output_dir / f"{stem}.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "config_name",
            "config_label",
            "collision_weight",
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
            "delta_success_pp",
            "delta_score_pct",
            "delta_distance_pct",
            "delta_time_pct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["summary"]:
            writer.writerow(row)

    with details_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "map_id",
            "file",
            "difficulty",
            "difficulty_label",
            "map_size",
            "seed",
            "start",
            "goal",
            "config_name",
            "config_label",
            "collision_weight",
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["details"]:
            writer.writerow(row)

    lines = [
        "# IAMSO 碰撞惩罚专项回归结果",
        "",
        f"- 数据集：`{report['dataset_name']}` v{report['dataset_version']}",
        f"- 地图数量：{report['map_count']}",
        f"- 难度：{report['difficulty_label']} {report['map_size']}x{report['map_size']}",
        f"- 目标障碍占比：{report['target_obstacle_ratio']:.0%}",
        "- 地图条件：固定起点、固定终点、连通性保护、仅保留理论可达地图",
        "- 代价函数：IAMSO + 模型综合代价",
        "- 说明：`1e12` 用作“近似无穷大”碰撞惩罚，避免直接使用数学无穷大导致数值不稳定",
        "",
        "| 配置 | 成功率 | 找到路径率 | 平均综合评分/100 | 平均路径长度 | 平均通行时间 | 平均碰撞次数 | 平均运行时间/s | 成功率变化 | 评分变化 | 距离变化 | 时间变化 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['config_label']} | {_format_percent(row['success_rate'])} | "
            f"{_format_percent(row['path_found_rate'])} | {_format_value(row['avg_score'])} | "
            f"{_format_value(row['avg_distance'])} | {_format_value(row['avg_time'])} | "
            f"{_format_value(row['avg_collision'])} | {_format_value(row['avg_runtime'])} | "
            f"{_format_delta(row['delta_success_pp'], suffix='pp')} | "
            f"{_format_delta(row['delta_score_pct'])} | "
            f"{_format_delta(row['delta_distance_pct'])} | "
            f"{_format_delta(row['delta_time_pct'])} |"
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "json": str(json_path),
        "summary_csv": str(summary_csv_path),
        "details_csv": str(details_csv_path),
        "markdown": str(markdown_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IAMSO 碰撞惩罚专项固定测试集回归")
    parser.add_argument(
        "--force-dataset",
        action="store_true",
        help="强制重建 100 张 hard-35%% 可达地图数据集。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(6, os.cpu_count() or 1)),
        help="并行 worker 数，默认按机器核数自动取不超过 6。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_results_dir().mkdir(parents=True, exist_ok=True)

    print("[setup] 准备 hard-35% 可达 100 图专项数据集")
    index = ensure_dataset(force=bool(args.force_dataset))
    print(
        f"[setup] 数据集就绪: {index['map_count']} 张地图，目录: {dataset_dir()}，"
        f"workers={args.max_workers}"
    )

    report = evaluate_sweep(index, max_workers=max(1, int(args.max_workers)))
    paths = save_report(report)

    print("[done] 专项回归完成")
    print(f"[done] markdown: {paths['markdown']}")
    print(f"[done] summary : {paths['summary_csv']}")
    print(f"[done] details : {paths['details_csv']}")
    print(f"[done] json    : {paths['json']}")


if __name__ == "__main__":
    main()
