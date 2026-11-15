"""Compare MSO/AO variants against IAMSO on the hard35 fixed dataset."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from uav_planner.experiments.fixed_benchmark.dataset import (  # noqa: E402
    benchmark_results_dir,
    experiment_root,
    load_map_bundle,
)
from uav_planner.experiments.fixed_benchmark.runner import clone_layers  # noqa: E402
from uav_planner.algorithms.ao.planner import ao_planner  # noqa: E402
from uav_planner.algorithms.mso.planner import mso_planner  # noqa: E402
from uav_planner.costs import model_cost  # noqa: E402
from uav_planner.metrics import calculate_extended_metrics  # noqa: E402


DATASET_DIR = experiment_root() / "datasets" / "iamso_collision_weight_hard35_reachable_100"
INDEX_PATH = DATASET_DIR / "index.json"
IAMSO_BASELINE_PATH = (
    benchmark_results_dir()
    / "archive"
    / "iamso_collision_weight_hard35_reachable100_20260709_215331_summary.csv"
)


def patch_no_exposure() -> None:
    import uav_planner.metaheuristic.ao as ao_mod
    import uav_planner.metaheuristic.fitness as fitness_mod
    import uav_planner.metaheuristic.mso as mso_mod

    original = fitness_mod.evaluate_vector

    def evaluate_vector_no_exposure(*args, **kwargs):
        kwargs["exposure_weight"] = 0.0
        return original(*args, **kwargs)

    mso_mod.evaluate_vector = evaluate_vector_no_exposure
    ao_mod.evaluate_vector = evaluate_vector_no_exposure


def _clone_passable(passable):
    return passable.copy()


def evaluate_one_map(record: dict[str, Any], disable_exposure: bool) -> list[dict[str, Any]]:
    if disable_exposure:
        patch_no_exposure()
    layers, passable, _ = load_map_bundle(DATASET_DIR / record["file"])
    start = tuple(record["start"])
    goal = tuple(record["goal"])
    rows: list[dict[str, Any]] = []
    suffix = "no_exposure" if disable_exposure else "base"

    for algorithm, planner in (
        (f"MSO_{suffix}", mso_planner),
        (f"AO_{suffix}", ao_planner),
    ):
        algo_layers = clone_layers(layers)
        algo_passable = _clone_passable(passable)
        started = time.perf_counter()
        error_message = ""
        try:
            path = planner(algo_passable, algo_layers, model_cost, start, goal)
        except Exception as exc:
            path = None
            error_message = str(exc)
        runtime = time.perf_counter() - started

        if path:
            metrics = calculate_extended_metrics(path, algo_layers, passable=algo_passable, runtime=runtime)
            status = "success" if (metrics["collision"] or 0) == 0 else "collision"
        else:
            metrics = {
                "distance": None,
                "time": None,
                "safety": None,
                "collision": None,
                "smoothness": None,
                "task_reward": None,
                "score": None,
            }
            status = "exception" if error_message else "no_path"

        rows.append(
            {
                "map_id": record["id"],
                "file": record["file"],
                "algorithm": algorithm,
                "status": status,
                "runtime": runtime,
                "distance": metrics["distance"],
                "time": metrics["time"],
                "safety": metrics["safety"],
                "collision_count": metrics["collision"],
                "smoothness": metrics["smoothness"],
                "task_reward": metrics["task_reward"],
                "score": metrics["score"],
                "error_message": error_message,
            }
        )
    return rows


def safe_avg(total: float, count: int) -> float | None:
    return (total / count) if count else None


def summarize(rows: list[dict[str, Any]], algorithm: str) -> dict[str, Any]:
    algo_rows = [row for row in rows if row["algorithm"] == algorithm]
    attempts = len(algo_rows)
    path_rows = [row for row in algo_rows if row["status"] in {"success", "collision"}]
    success_rows = [row for row in algo_rows if row["status"] == "success"]
    score_rows = [row for row in algo_rows if row["score"] is not None]
    return {
        "algorithm": algorithm,
        "attempts": attempts,
        "path_found_rate": len(path_rows) / attempts if attempts else 0.0,
        "success_rate": len(success_rows) / attempts if attempts else 0.0,
        "collision_failure_count": sum(1 for row in algo_rows if row["status"] == "collision"),
        "no_path_count": sum(1 for row in algo_rows if row["status"] == "no_path"),
        "exception_count": sum(1 for row in algo_rows if row["status"] == "exception"),
        "avg_distance": safe_avg(sum(float(row["distance"] or 0.0) for row in success_rows), len(success_rows)),
        "avg_time": safe_avg(sum(float(row["time"] or 0.0) for row in success_rows), len(success_rows)),
        "avg_safety": safe_avg(sum(float(row["safety"] or 0.0) for row in success_rows), len(success_rows)),
        "avg_collision": safe_avg(
            sum(1 if (row["collision_count"] or 0) > 0 else 0 for row in path_rows),
            len(path_rows),
        ),
        "avg_smoothness": safe_avg(
            sum(float(row["smoothness"] or 0.0) for row in success_rows),
            len(success_rows),
        ),
        "avg_task_reward": safe_avg(
            sum(float(row["task_reward"] or 0.0) for row in success_rows),
            len(success_rows),
        ),
        "avg_score": safe_avg(sum(float(row["score"] or 0.0) for row in score_rows), len(score_rows)),
        "avg_runtime": safe_avg(sum(float(row["runtime"]) for row in algo_rows), attempts),
    }


def load_iamso_baseline() -> dict[str, str]:
    with IAMSO_BASELINE_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return next(row for row in rows if row["config_name"] == "baseline_1000")


def format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def format_value(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MSO/AO 与 IAMSO 的 hard35 固定集对照测试")
    parser.add_argument(
        "--disable-exposure",
        action="store_true",
        help="关闭 MSO/AO 共享适应度中的连续暴露非线性风险项。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    disable_exposure = bool(args.disable_exposure)
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    records = list(index["maps"])
    workers = max(1, min(6, os.cpu_count() or 1))
    all_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    suffix = "no_exposure" if disable_exposure else "base"
    mso_name = f"MSO_{suffix}"
    ao_name = f"AO_{suffix}"

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_one_map, record, disable_exposure): record["id"]
            for record in records
        }
        completed = 0
        for future in as_completed(futures):
            all_rows.extend(future.result())
            completed += 1
            if completed % 10 == 0 or completed == len(records):
                print(f"[progress] {completed}/{len(records)} maps done, elapsed={time.perf_counter() - started:.1f}s")

    summary = [
        summarize(all_rows, mso_name),
        summarize(all_rows, ao_name),
    ]
    iamso_baseline = load_iamso_baseline()

    report = {
        "dataset_name": index["dataset_name"],
        "dataset_version": index["dataset_version"],
        "map_count": len(records),
        "dataset_root": str(DATASET_DIR),
        "iamso_baseline": iamso_baseline,
        "variant": suffix,
        "summary": summary,
        "details": all_rows,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"mso_ao_{suffix}_vs_iamso_hard35_{timestamp}"
    output_dir = benchmark_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# MSO/AO 去连续暴露风险 vs IAMSO 对比",
        "",
        f"- 数据集：`{index['dataset_name']}` v{index['dataset_version']}",
        f"- 地图数量：{len(records)}",
        "- 地图条件：困难35%可达，固定起点终点，连通性保护",
        "- IAMSO 对照组：当前 baseline_1000（含当前实现）",
        (
            "- MSO/AO 试验组：仅去掉连续暴露非线性风险项，其他逻辑不变"
            if disable_exposure
            else "- MSO/AO 试验组：当前原始实现"
        ),
        "",
        "| 算法/配置 | 成功率 | 找到路径率 | 平均综合评分/100 | 平均路径长度 | 平均通行时间 | 平均碰撞次数 | 平均运行时间/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| IAMSO baseline_1000 | {format_percent(float(iamso_baseline['success_rate']))} | "
            f"{format_percent(float(iamso_baseline['path_found_rate']))} | {format_value(float(iamso_baseline['avg_score']))} | "
            f"{format_value(float(iamso_baseline['avg_distance']))} | {format_value(float(iamso_baseline['avg_time']))} | "
            f"{format_value(float(iamso_baseline['avg_collision']))} | {format_value(float(iamso_baseline['avg_runtime']))} |"
        ),
    ]
    for row in summary:
        lines.append(
            f"| {row['algorithm']} | {format_percent(row['success_rate'])} | {format_percent(row['path_found_rate'])} | "
            f"{format_value(row['avg_score'])} | {format_value(row['avg_distance'])} | {format_value(row['avg_time'])} | "
            f"{format_value(row['avg_collision'])} | {format_value(row['avg_runtime'])} |"
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[done] json: {json_path}")
    print(f"[done] md  : {markdown_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("IAMSO baseline:", json.dumps(iamso_baseline, ensure_ascii=False))


if __name__ == "__main__":
    main()
