"""Ablation study: IAMSO-NF-BI-CA (collision-aware) vs IAMSO baseline (no CA) vs MSO."

Creates three algorithm variants that share the same optimizer and fitness
function but differ only in population initialization strategy:

  - IAMSO (no CA): original multi-source seed initialization
  - IAMSO-NF-BI-CA: collision-aware seed encoding (production version)
  - MSO: standard single-seed initialization (baseline)

Does NOT modify any existing code.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from uav_planner.config import MODEL_WEIGHTS, NO_TASK_MODEL_COST_WEIGHTS
from uav_planner.experiments.fixed_benchmark.dataset import (
    benchmark_maps_dir,
    load_index,
    load_map_bundle,
)
from uav_planner.geometry import smooth_planner_result
from uav_planner.metrics import calculate_extended_metrics
from uav_planner.metaheuristic import iamso_optimize, mso_optimize
from uav_planner.metaheuristic.collision_aware import collision_aware_initial_population
from uav_planner.algorithms.iamso.seeds import build_bidirectional_seed_paths
from uav_planner.costs import model_cost as _model_cost, no_task_model_cost


_NO_TASK_LAYER_WEIGHTS = tuple(
    NO_TASK_MODEL_COST_WEIGHTS[key]
    for key in ("inflation", "wind", "rain", "cruise")
)


def _seed_value(start, goal, salt):
    return hash((tuple(start), tuple(goal), salt)) % (2**31)


def iamso_no_ca_planner(passable, speed_map, cost_func, start, goal):
    """IAMSO without collision-aware encoding (baseline)."""
    seed_paths, seed_labels = build_bidirectional_seed_paths(
        passable, speed_map, cost_func, start, goal
    )
    base_weights = (
        _NO_TASK_LAYER_WEIGHTS if cost_func is no_task_model_cost else MODEL_WEIGHTS
    )
    result = iamso_optimize(
        passable, speed_map, cost_func, start, goal,
        seed_paths=seed_paths, fallback_path=None,
        base_weights=base_weights,
        random_seed=_seed_value(start, goal, 17),
        initial_population_builder=None,
    )
    path = result.get("path")
    if path is None:
        return None
    planned_path = smooth_planner_result(path, passable, used_fallback=False)
    planned_path.solution_source = result.get("solution_source", "unknown")
    planned_path.feasible = bool(result.get("feasible", False))
    return planned_path


def iamso_ca_planner(passable, speed_map, cost_func, start, goal):
    """IAMSO-NF-BI-CA (production version with collision-aware encoding)."""
    seed_paths, seed_labels = build_bidirectional_seed_paths(
        passable, speed_map, cost_func, start, goal
    )
    base_weights = (
        _NO_TASK_LAYER_WEIGHTS if cost_func is no_task_model_cost else MODEL_WEIGHTS
    )
    result = iamso_optimize(
        passable, speed_map, cost_func, start, goal,
        seed_paths=seed_paths, fallback_path=None,
        base_weights=base_weights,
        random_seed=_seed_value(start, goal, 17),
        initial_population_builder=collision_aware_initial_population,
    )
    path = result.get("path")
    if path is None:
        return None
    planned_path = smooth_planner_result(path, passable, used_fallback=False)
    planned_path.solution_source = result.get("solution_source", "unknown")
    planned_path.feasible = bool(result.get("feasible", False))
    return planned_path


def mso_baseline_planner(passable, speed_map, cost_func, start, goal):
    """MSO baseline (standard single-seed, no BI, no CA)."""
    from uav_planner.algorithms.astar import astar as _astar
    seed_path = _astar(passable, speed_map, cost_func, start, goal)
    result = mso_optimize(
        passable, speed_map, cost_func, start, goal,
        seed_path=seed_path,
        random_seed=_seed_value(start, goal, 7),
    )
    path = result.get("path")
    if path is None:
        return None
    return smooth_planner_result(path, passable, used_fallback=False)


ALGORITHMS = {
    "iamso_no_ca": ("IAMSO(no-CA)", iamso_no_ca_planner),
    "iamso_ca":    ("IAMSO-NF-BI-CA", iamso_ca_planner),
    "mso":         ("MSO", mso_baseline_planner),
}


def _clone_layers(layers):
    cloned = {}
    for key, value in layers.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def _new_bucket():
    return {
        "attempts": 0, "path_found": 0, "success": 0,
        "no_path": 0, "collision_failures": 0, "exception_count": 0,
        "distance": 0.0, "time": 0.0, "safety": 0.0, "collision": 0.0,
        "smoothness": 0.0, "task_reward": 0.0, "score": 0.0,
        "score_count": 0, "runtime": 0.0,
    }


def _safe_avg(total, count):
    return (total / count) if count else None


def _summarize(bucket):
    attempts = bucket["attempts"]
    success = bucket["success"]
    path_found = bucket["path_found"]
    return {
        "attempts": attempts,
        "success_rate": success / attempts if attempts else 0,
        "path_found_rate": path_found / attempts if attempts else 0,
        "avg_distance": _safe_avg(bucket["distance"], success),
        "avg_time": _safe_avg(bucket["time"], success),
        "avg_safety": _safe_avg(bucket["safety"], success),
        "avg_collision": _safe_avg(bucket["collision"], path_found),
        "avg_smoothness": _safe_avg(bucket["smoothness"], success),
        "avg_task_reward": _safe_avg(bucket["task_reward"], success),
        "avg_score": _safe_avg(bucket["score"], bucket["score_count"]),
        "avg_runtime": _safe_avg(bucket["runtime"], attempts),
        "no_path_count": bucket["no_path"],
        "collision_failures": bucket["collision_failures"],
    }


def run_ablation(difficulties=None):
    """Run the ablation experiment and return structured results."""
    index = load_index()
    wanted = set(difficulties or [])
    records = [
        item for item in index["maps"]
        if not wanted or item["difficulty"] in wanted
    ]
    selected = sorted({r["difficulty"] for r in records})
    difficulty_labels = {
        k: index["difficulty_specs"][k]["label"] for k in selected
    }

    overall = {key: _new_bucket() for key in ALGORITHMS}
    by_diff = {
        d: {key: _new_bucket() for key in ALGORITHMS} for d in selected
    }
    details = []

    total = len(records) * len(ALGORITHMS)
    done = 0
    t0 = time.perf_counter()

    for mi, record in enumerate(records, 1):
        ds_path = benchmark_maps_dir() / record["file"]
        layers, passable, meta = load_map_bundle(ds_path)
        start = tuple(record["start"])
        goal = tuple(record["goal"])
        diff = record["difficulty"]
        label = record["difficulty_label"]
        sz = record["map_size"]

        print(f"\nMap {mi}/{len(records)}: {record['id']} ({label} {sz}x{sz})")

        for key, (name, func) in ALGORITHMS.items():
            done += 1
            algo_layers = _clone_layers(layers)
            algo_passable = passable.copy()

            t_start = time.perf_counter()
            status = "success"
            err = ""
            try:
                path = func(algo_passable, algo_layers, _model_cost, start, goal)
            except Exception as exc:
                path = None
                status = "exception"
                err = str(exc)
            runtime = time.perf_counter() - t_start

            bucket = overall[key]
            dbucket = by_diff[diff][key]
            bucket["attempts"] += 1
            dbucket["attempts"] += 1
            bucket["runtime"] += runtime
            dbucket["runtime"] += runtime

            dist = ts = saf = smth = tr = sc = col = None
            if path:
                bucket["path_found"] += 1
                dbucket["path_found"] += 1
                metrics = calculate_extended_metrics(
                    path, algo_layers, passable=algo_passable, runtime=runtime
                )
                dist = metrics["distance"]
                ts = metrics["time"]
                saf = metrics["safety"]
                smth = metrics["smoothness"]
                tr = metrics["task_reward"]
                col = metrics["collision"]
                sc = metrics["score"]

                cflag = 1 if (col and col > 0) else 0
                bucket["collision"] += cflag
                dbucket["collision"] += cflag
                if sc is not None:
                    bucket["score"] += float(sc)
                    dbucket["score"] += float(sc)
                    bucket["score_count"] += 1
                    dbucket["score_count"] += 1

                if col == 0:
                    bucket["success"] += 1
                    dbucket["success"] += 1
                    bucket["distance"] += float(dist or 0)
                    dbucket["distance"] += float(dist or 0)
                    bucket["time"] += float(ts or 0)
                    dbucket["time"] += float(ts or 0)
                    bucket["safety"] += float(saf or 0)
                    dbucket["safety"] += float(saf or 0)
                    bucket["smoothness"] += float(smth or 0)
                    dbucket["smoothness"] += float(smth or 0)
                    bucket["task_reward"] += float(tr or 0)
                    dbucket["task_reward"] += float(tr or 0)
                else:
                    status = "collision"
                    bucket["collision_failures"] += 1
                    dbucket["collision_failures"] += 1
            else:
                if status == "exception":
                    bucket["exception_count"] += 1
                    dbucket["exception_count"] += 1
                else:
                    status = "no_path"
                    bucket["no_path"] += 1
                    dbucket["no_path"] += 1

            details.append({
                "map_id": record["id"], "difficulty": diff,
                "algorithm": key, "algorithm_name": name,
                "status": status, "success": 1 if status == "success" else 0,
                "runtime": runtime, "distance": dist, "time": ts,
                "safety": saf, "collision_count": col, "smoothness": smth,
                "task_reward": tr, "score": sc, "error": err,
            })

            col_str = "-" if col is None else str(int(col))
            elapsed = time.perf_counter() - t0
            print(
                f"  [{done:>3}/{total}] {name:<15} "
                f"status={status:<9} col={col_str:<3} "
                f"time={runtime:>7.3f}s  elapsed={elapsed:>8.1f}s"
            )

    overall_summary = {
        key: {"algorithm": ALGORITHMS[key][0], **_summarize(bucket)}
        for key, bucket in overall.items()
    }
    diff_summary = {
        d: {
            key: {"algorithm": ALGORITHMS[key][0], **_summarize(bucket)}
            for key, bucket in by_diff[d].items()
        }
        for d in selected
    }

    return {
        "dataset_version": index["dataset_version"],
        "map_count": len(records),
        "selected_difficulties": selected,
        "difficulty_labels": difficulty_labels,
        "algorithms": [ALGORITHMS[k][0] for k in ALGORITHMS],
        "summary": {"overall": overall_summary, "by_difficulty": diff_summary},
        "details": details,
    }


def _output_dir():
    d = Path(__file__).parent / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_json(report):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _output_dir() / f"iamso_ca_ablation_{ts}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _save_csv(report):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _output_dir() / f"iamso_ca_ablation_{ts}.csv"
    fields = [
        "scope", "algorithm", "attempts", "success_rate",
        "path_found_rate", "avg_distance", "avg_time", "avg_safety",
        "avg_collision", "avg_smoothness", "avg_task_reward",
        "avg_score", "avg_runtime", "no_path_count", "collision_failures",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in report["summary"]["overall"].values():
            w.writerow({"scope": "overall", **row})
        for diff, rows in report["summary"]["by_difficulty"].items():
            for row in rows.values():
                w.writerow({"scope": diff, **row})
    return path


def _save_markdown(report):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _output_dir() / f"iamso_ca_ablation_{ts}.md"

    def _pct(v):
        return "-" if v is None else f"{v:.2%}"
    def _val(v, d=4):
        return "-" if v is None else f"{v:.{d}f}"

    lines = [
        "# IAMSO-NF-BI-CA Ablation Study",
        "",
        f"- Dataset v{report['dataset_version']}  |  Maps: {report['map_count']}",
        f"- Difficulties: {', '.join(report['selected_difficulties'])}",
        "",
        "## Overall",
        "",
        "| Algorithm | Success Rate | Path Found | Avg Dist | Avg Time | Avg Safety | Avg Col | Avg Smooth | Avg Score | Avg Runtime | No-Path | Col Fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]["overall"].values():
        lines.append(
            f"| {row['algorithm']} | {_pct(row['success_rate'])} | {_pct(row['path_found_rate'])} "
            f"| {_val(row['avg_distance'])} | {_val(row['avg_time'])} | {_pct(row['avg_safety'])} "
            f"| {_val(row['avg_collision'])} | {_val(row['avg_smoothness'])} | {_val(row['avg_score'])} "
            f"| {_val(row['avg_runtime'])} | {row['no_path_count']} | {row['collision_failures']} |"
        )

    for diff, rows in report["summary"]["by_difficulty"].items():
        label = report["difficulty_labels"].get(diff, diff)
        lines += [
            "", f"## {label} ({diff})", "",
            "| Algorithm | Success Rate | Path Found | Avg Dist | Avg Time | Avg Safety | Avg Col | Avg Smooth | Avg Score | Avg Runtime | No-Path | Col Fail |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows.values():
            lines.append(
                f"| {row['algorithm']} | {_pct(row['success_rate'])} | {_pct(row['path_found_rate'])} "
                f"| {_val(row['avg_distance'])} | {_val(row['avg_time'])} | {_pct(row['avg_safety'])} "
                f"| {_val(row['avg_collision'])} | {_val(row['avg_smoothness'])} | {_val(row['avg_score'])} "
                f"| {_val(row['avg_runtime'])} | {row['no_path_count']} | {row['collision_failures']} |"
            )

    lines += ["", f"*Generated {ts}*"]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="IAMSO-NF-BI-CA ablation study")
    parser.add_argument(
        "--difficulty", nargs="*",
        choices=["simple", "medium", "hard"],
        help="Limit to specific difficulties (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  IAMSO-NF-BI-CA Ablation Study")
    print("  Algorithms: IAMSO(no-CA)  |  IAMSO-NF-BI-CA  |  MSO")
    print("=" * 60)

    report = run_ablation(args.difficulty)

    jp = _save_json(report)
    cp = _save_csv(report)
    mp = _save_markdown(report)

    print("\n" + "=" * 60)
    print("  Results saved:")
    print(f"    JSON:   {jp}")
    print(f"    CSV:    {cp}")
    print(f"    MD:     {mp}")
    print("=" * 60)

    print("\n=== Overall Results ===")
    for key, row in report["summary"]["overall"].items():
        print(
            f"  {row['algorithm']:<15} "
            f"Success={row['success_rate']:.2%}  "
            f"ColFail={row['collision_failures']}  "
            f"AvgRuntime={row['avg_runtime']:.3f}s"
        )

    return report


if __name__ == "__main__":
    main()
