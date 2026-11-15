"""AO P1 safe-smoothing ablation on mode-6 fixed maps.

This script is experiment-only. It does not modify the production AO planner,
registry, or mode-6 runner. The default variant is the practical P0+P1 version:
feasibility-priority selection plus safe internal/final smoothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from uav_planner.algorithms.astar import astar
from uav_planner.algorithms.common import seed_value
from uav_planner.config import NO_TASK_EVALUATION_SCORE_WEIGHTS
from uav_planner.costs import no_task_model_cost
from uav_planner.experiments.no_task_corner.dataset import (
    load_index,
    load_map_bundle,
    no_task_corner_maps_dir,
    no_task_corner_results_dir,
)
from uav_planner.experiments.no_task_corner.runner import clone_layers
from uav_planner.metaheuristic.ao import _initial_population, _levy
from uav_planner.metaheuristic.encoding import clip_vector, decode, smooth_path
from uav_planner.metaheuristic.fitness import FitnessResult, evaluate_path, evaluate_vector
from uav_planner.metrics import calculate_collision_count, calculate_extended_metrics


EXPERIMENT_ROOT = Path(__file__).resolve().parent


SCENARIOS = {
    "medium15": {
        "label": "medium15",
        "difficulty": "medium",
        "maps_root": no_task_corner_maps_dir(),
        "index_path": None,
        "description": "Mode-6 medium 60x60 maps with 15% target obstacle ratio.",
    },
    "medium20": {
        "label": "medium20",
        "difficulty": "medium20",
        "maps_root": EXPERIMENT_ROOT / "datasets" / "medium60_density20_reachable_100",
        "index_path": EXPERIMENT_ROOT / "datasets" / "medium60_density20_reachable_100" / "index.json",
        "description": "Mode-6 medium 60x60 reachable dataset with 20% target obstacle ratio.",
    },
    "hard": {
        "label": "hard",
        "difficulty": "hard",
        "maps_root": no_task_corner_maps_dir(),
        "index_path": None,
        "description": "Mode-6 hard 90x90 maps with 20% target obstacle ratio.",
    },
}


@dataclass(frozen=True)
class Variant:
    name: str
    lexicographic_replacement: bool
    feasible_return: bool
    safe_inner_smoothing: bool
    safe_final_smoothing: bool
    description: str


VARIANTS = {
    "AO-P1-FINAL-SAFE": Variant(
        "AO-P1-FINAL-SAFE",
        False,
        False,
        False,
        True,
        "Current AO search, but final smoothing is accepted only if collision-free.",
    ),
    "AO-P1-INNER-SAFE": Variant(
        "AO-P1-INNER-SAFE",
        False,
        False,
        True,
        False,
        "Current AO replacement, but internal smoothing falls back to the raw waypoint polyline when smoothing collides.",
    ),
    "AO-P1-FULL": Variant(
        "AO-P1-FULL",
        False,
        False,
        True,
        True,
        "Current AO replacement plus safe internal and safe final smoothing.",
    ),
    "AO-P0P1-FULL": Variant(
        "AO-P0P1-FULL",
        True,
        True,
        True,
        True,
        "P0 feasibility-priority selection plus safe internal and safe final smoothing.",
    ),
}

DEFAULT_VARIANTS = ("AO-P0P1-FULL",)


def _eval_key(evaluation: FitnessResult) -> tuple[int, float]:
    return int(evaluation.collision_count), float(evaluation.value)


def _is_better(candidate: FitnessResult, current: FitnessResult, *, lexicographic: bool) -> bool:
    if lexicographic:
        return _eval_key(candidate) < _eval_key(current)
    return float(candidate.value) <= float(current.value)


def _update_best_feasible(best_feasible, vector: np.ndarray, evaluation: FitnessResult):
    if int(evaluation.collision_count) != 0:
        return best_feasible
    if best_feasible is None or float(evaluation.value) < float(best_feasible[1].value):
        return vector.copy(), evaluation
    return best_feasible


def evaluate_vector_p1(
    vector: np.ndarray,
    passable: np.ndarray,
    layers: dict[str, Any],
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    safe_inner_smoothing: bool,
) -> tuple[FitnessResult, str, int, int]:
    """Evaluate a vector with optional safe internal smoothing.

    Current AO always evaluates the Catmull-Rom smoothed path. P1 keeps that
    smoothed path only when it is collision-free; otherwise it evaluates and
    returns the raw decoded waypoint polyline.
    """

    if not safe_inner_smoothing:
        evaluation = evaluate_vector(vector, passable, layers, no_task_model_cost, start, goal)
        return evaluation, "smooth", -1, int(evaluation.collision_count)

    raw_path = decode(vector, start, goal)
    smoothed_path = smooth_path(raw_path)
    raw_eval = evaluate_path(raw_path, passable, layers, no_task_model_cost)
    smooth_eval = evaluate_path(smoothed_path, passable, layers, no_task_model_cost)
    if int(smooth_eval.collision_count) == 0:
        return smooth_eval, "smooth", int(raw_eval.collision_count), int(smooth_eval.collision_count)
    return raw_eval, "raw", int(raw_eval.collision_count), int(smooth_eval.collision_count)


def _finalize_path(
    inner_path: list[tuple[float, float]],
    passable: np.ndarray,
    *,
    safe_final_smoothing: bool,
) -> tuple[list[tuple[float, float]], str, int]:
    smoothed = smooth_path(inner_path, samples_per_segment=10)
    smoothed_collision = int(calculate_collision_count(smoothed, passable))
    if safe_final_smoothing and smoothed_collision > 0:
        return inner_path, "kept_inner", int(calculate_collision_count(inner_path, passable))
    return smoothed, "final_smoothed", smoothed_collision


def ao_variant_optimize(
    passable: np.ndarray,
    layers: dict[str, Any],
    start: tuple[int, int],
    goal: tuple[int, int],
    variant: Variant,
    population_size: int = 30,
    iterations: int = 100,
    waypoint_count: int = 6,
) -> dict[str, Any]:
    seed_path = astar(passable, layers, no_task_model_cost, start, goal)
    rng = np.random.default_rng(seed_value(start, goal, 8))
    size = int(passable.shape[0])
    dim = waypoint_count * 2
    lower = np.zeros(dim)
    upper = np.full(dim, float(size - 1))

    population = _initial_population(
        rng, population_size, waypoint_count, size, start, goal, seed_path
    )
    eval_rows = [
        evaluate_vector_p1(
            ind,
            passable,
            layers,
            start,
            goal,
            safe_inner_smoothing=variant.safe_inner_smoothing,
        )
        for ind in population
    ]
    evaluations = [row[0] for row in eval_rows]
    inner_sources = [row[1] for row in eval_rows]
    raw_collisions = [row[2] for row in eval_rows]
    smooth_collisions = [row[3] for row in eval_rows]
    fitness = np.asarray([evaluation.value for evaluation in evaluations], dtype=float)

    if variant.lexicographic_replacement:
        best_idx = min(range(population_size), key=lambda idx: _eval_key(evaluations[idx]))
    else:
        best_idx = int(np.argmin(fitness))
    best = population[best_idx].copy()
    best_eval = evaluations[best_idx]
    best_source = inner_sources[best_idx]
    best_raw_collision = raw_collisions[best_idx]
    best_smooth_collision = smooth_collisions[best_idx]
    best_feasible = None
    for vector, evaluation in zip(population, evaluations):
        best_feasible = _update_best_feasible(best_feasible, vector, evaluation)

    for iteration in range(1, iterations + 1):
        t = iteration / iterations
        mean = population.mean(axis=0)
        worst = population[int(np.argmax(fitness))]

        for i in range(population_size):
            current = population[i]
            if t <= 2.0 / 3.0:
                if rng.random() < 0.5:
                    candidate = best * (1.0 - t) + (mean - best * rng.random(dim))
                else:
                    peer = population[int(rng.integers(0, population_size))]
                    candidate = best * _levy(rng, dim) + peer + rng.random(dim) * (peer - current)
            else:
                quality = 2.0 * rng.random(dim) - 1.0
                if rng.random() < 0.5:
                    candidate = (best - mean) * quality - rng.random(dim) * ((upper - lower) * t + lower)
                else:
                    levy = _levy(rng, dim)
                    candidate = best + quality * (best - current) + rng.random(dim) * levy * (best - worst)

            if rng.random() < 0.10:
                candidate += rng.normal(0.0, size * 0.02 * (1.0 - t), dim)
            candidate = clip_vector(candidate, size)
            candidate_eval, candidate_source, candidate_raw_collision, candidate_smooth_collision = evaluate_vector_p1(
                candidate,
                passable,
                layers,
                start,
                goal,
                safe_inner_smoothing=variant.safe_inner_smoothing,
            )
            best_feasible = _update_best_feasible(best_feasible, candidate, candidate_eval)

            if _is_better(
                candidate_eval,
                evaluations[i],
                lexicographic=variant.lexicographic_replacement,
            ):
                population[i] = candidate
                evaluations[i] = candidate_eval
                inner_sources[i] = candidate_source
                raw_collisions[i] = candidate_raw_collision
                smooth_collisions[i] = candidate_smooth_collision
                fitness[i] = float(candidate_eval.value)
                if _is_better(
                    candidate_eval,
                    best_eval,
                    lexicographic=variant.lexicographic_replacement,
                ):
                    best = candidate.copy()
                    best_eval = candidate_eval
                    best_source = candidate_source
                    best_raw_collision = candidate_raw_collision
                    best_smooth_collision = candidate_smooth_collision

    selected = best
    selected_eval = best_eval
    selected_source = "search_best"
    inner_source = best_source
    raw_collision_count = best_raw_collision
    smooth_collision_count = best_smooth_collision
    if variant.feasible_return and best_feasible is not None:
        selected, selected_eval = best_feasible
        selected_source = "best_feasible"
        selected_eval, inner_source, raw_collision_count, smooth_collision_count = evaluate_vector_p1(
            selected,
            passable,
            layers,
            start,
            goal,
            safe_inner_smoothing=variant.safe_inner_smoothing,
        )

    inner_path = [(float(row), float(col)) for row, col in selected_eval.path]
    final_path, final_source, final_collision_count = _finalize_path(
        inner_path,
        passable,
        safe_final_smoothing=variant.safe_final_smoothing,
    )
    return {
        "path": final_path,
        "raw_collision_count": raw_collision_count,
        "smooth_collision_count": smooth_collision_count,
        "inner_collision_count": int(selected_eval.collision_count),
        "final_collision_count": final_collision_count,
        "inner_source": inner_source,
        "final_source": final_source,
        "selected_source": selected_source,
        "feasible_candidate_found": best_feasible is not None,
        "selected_fitness": float(selected_eval.value),
    }


def _blank_metrics() -> dict[str, Any]:
    return {
        "distance": None,
        "time": None,
        "safety": None,
        "smoothness": None,
        "rain_exposure": None,
        "inflation_exposure": None,
        "total_exposure": None,
        "score": None,
    }


def run_one(record: dict[str, Any], variant: Variant, maps_root: Path) -> dict[str, Any]:
    layers, passable, _ = load_map_bundle(maps_root / record["file"])
    start = tuple(record["start"])
    goal = tuple(record["goal"])
    started = time.perf_counter()
    metrics = _blank_metrics()
    error_message = ""
    try:
        result = ao_variant_optimize(
            passable.copy(), clone_layers(layers), start, goal, variant
        )
        path = result["path"]
        status = "success" if path and result["final_collision_count"] == 0 else "collision"
        if path:
            metrics = calculate_extended_metrics(
                path,
                clone_layers(layers),
                passable=passable,
                runtime=time.perf_counter() - started,
                include_task_reward=False,
                score_weights=NO_TASK_EVALUATION_SCORE_WEIGHTS,
            )
    except Exception as exc:  # pragma: no cover - experiment diagnostics
        result = {
            "raw_collision_count": None,
            "smooth_collision_count": None,
            "inner_collision_count": None,
            "final_collision_count": None,
            "inner_source": "",
            "final_source": "",
            "selected_source": "",
            "feasible_candidate_found": False,
            "selected_fitness": None,
        }
        status = "exception"
        error_message = str(exc)

    runtime = time.perf_counter() - started
    return {
        "map_id": record["id"],
        "file": record["file"],
        "variant": variant.name,
        "status": status,
        "success": 1 if status == "success" else 0,
        "runtime": runtime,
        "raw_collision_count": result["raw_collision_count"],
        "smooth_collision_count": result["smooth_collision_count"],
        "inner_collision_count": result["inner_collision_count"],
        "final_collision_count": result["final_collision_count"],
        "inner_source": result["inner_source"],
        "final_source": result["final_source"],
        "selected_source": result["selected_source"],
        "feasible_candidate_found": result["feasible_candidate_found"],
        "selected_fitness": result["selected_fitness"],
        "distance": metrics["distance"],
        "time": metrics["time"],
        "safety": metrics["safety"],
        "smoothness": metrics["smoothness"],
        "rain_exposure": metrics["rain_exposure"],
        "inflation_exposure": metrics["inflation_exposure"],
        "total_exposure": metrics["total_exposure"],
        "score": metrics["score"],
        "error_message": error_message,
    }


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(details: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    rows = []
    for variant_name in variants:
        selected = [row for row in details if row["variant"] == variant_name]
        attempts = len(selected)
        path_rows = [row for row in selected if row["status"] in {"success", "collision"}]
        success_rows = [row for row in selected if row["status"] == "success"]
        rows.append(
            {
                "variant": variant_name,
                "attempts": attempts,
                "success": len(success_rows),
                "success_rate": len(success_rows) / attempts if attempts else 0.0,
                "collision_failures": sum(row["status"] == "collision" for row in selected),
                "exceptions": sum(row["status"] == "exception" for row in selected),
                "feasible_candidate_found_rate": sum(bool(row["feasible_candidate_found"]) for row in selected) / attempts if attempts else 0.0,
                "best_feasible_return_count": sum(row["selected_source"] == "best_feasible" for row in selected),
                "inner_raw_count": sum(row["inner_source"] == "raw" for row in selected),
                "final_kept_inner_count": sum(row["final_source"] == "kept_inner" for row in selected),
                "avg_raw_collision_count": _avg([float(row["raw_collision_count"]) for row in path_rows if row["raw_collision_count"] is not None and float(row["raw_collision_count"]) >= 0]),
                "avg_smooth_collision_count": _avg([float(row["smooth_collision_count"]) for row in path_rows if row["smooth_collision_count"] is not None]),
                "avg_inner_collision_count": _avg([float(row["inner_collision_count"]) for row in path_rows if row["inner_collision_count"] is not None]),
                "avg_final_collision_count": _avg([float(row["final_collision_count"]) for row in path_rows if row["final_collision_count"] is not None]),
                "avg_distance_success": _avg([float(row["distance"]) for row in success_rows if row["distance"] is not None]),
                "avg_time_success": _avg([float(row["time"]) for row in success_rows if row["time"] is not None]),
                "avg_safety_success": _avg([float(row["safety"]) for row in success_rows if row["safety"] is not None]),
                "avg_score": _avg([float(row["score"]) for row in path_rows if row["score"] is not None]),
                "avg_runtime": _avg([float(row["runtime"]) for row in selected]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)!r}")


def fmt_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def fmt_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# AO P1 Ablation - Mode 6 {report['scenario']}",
        "",
        f"- Dataset: {report['dataset_name']} v{report['dataset_version']}",
        f"- Scenario: {report['scenario']}",
        f"- Difficulty key: {report['difficulty']}",
        f"- Map count: {report['map_count']}",
        f"- Target obstacle ratio: {report['target_obstacle_ratio']:.0%}",
        f"- Maps root: {report['maps_root']}",
        "- Production AO changed: no",
        "",
        "## Variants",
        "",
    ]
    for variant_name in report["variants"]:
        lines.append(f"- {variant_name}: {VARIANTS[variant_name].description}")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Variant | Success | Collision Fail | Feasible Found | Feasible Return | Inner Raw | Final Kept Inner | Avg Inner Collision | Avg Final Collision | Avg Score | Avg Runtime/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["summary"]:
        lines.append(
            f"| {row['variant']} | {fmt_percent(row['success_rate'])} | "
            f"{row['collision_failures']} | {fmt_percent(row['feasible_candidate_found_rate'])} | "
            f"{row['best_feasible_return_count']} | {row['inner_raw_count']} | "
            f"{row['final_kept_inner_count']} | {fmt_float(row['avg_inner_collision_count'])} | "
            f"{fmt_float(row['avg_final_collision_count'])} | {fmt_float(row['avg_score'])} | "
            f"{fmt_float(row['avg_runtime'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AO P1 safe-smoothing ablation on mode-6 fixed maps.")
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default="medium15",
        help="Dataset scenario. Defaults to medium15 for backward compatibility.",
    )
    parser.add_argument("--variant", nargs="+", choices=tuple(VARIANTS), default=list(DEFAULT_VARIANTS))
    parser.add_argument("--all-variants", action="store_true", help="Run all P1/P0+P1 variants.")
    parser.add_argument("--limit", type=int, help="Run only the first N maps for smoke testing.")
    parser.add_argument("--record-id", nargs="*", help="Run explicit ids such as medium_001, medium20_001, or hard_001.")
    parser.add_argument("--output-dir", type=Path, default=no_task_corner_results_dir())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario]
    maps_root = Path(scenario["maps_root"])
    index = load_index(scenario["index_path"])
    difficulty = str(scenario["difficulty"])
    records = [row for row in index["maps"] if row["difficulty"] == difficulty]
    if args.record_id:
        wanted = set(args.record_id)
        records = [row for row in records if row["id"] in wanted]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"No maps selected for scenario {args.scenario}.")

    variants = list(VARIANTS) if args.all_variants else list(args.variant)
    details = []
    total_runs = len(records) * len(variants)
    started = time.perf_counter()
    completed = 0
    for record in records:
        obstacle_ratio = float(record.get("target_obstacle_ratio", 0.0))
        print(f"\nMap {record['id']} ({record['map_size']}x{record['map_size']}, {obstacle_ratio:.0%})")
        for variant_name in variants:
            completed += 1
            row = run_one(record, VARIANTS[variant_name], maps_root)
            details.append(row)
            final_collision = row["final_collision_count"]
            final_text = "-" if final_collision is None else str(final_collision)
            print(
                f"  [{completed:>3}/{total_runs}] {variant_name:<20} "
                f"status={row['status']:<9} final_collision={final_text:<4} "
                f"inner={row['inner_source']:<6} final={row['final_source']:<14} "
                f"runtime={row['runtime']:>7.2f}s elapsed={time.perf_counter() - started:>8.1f}s",
                flush=True,
            )

    summary = summarize(details, variants)
    report = {
        "dataset_name": index["dataset_name"],
        "dataset_version": index["dataset_version"],
        "map_count": len(records),
        "scenario": args.scenario,
        "scenario_description": scenario["description"],
        "difficulty": difficulty,
        "maps_root": str(maps_root),
        "target_obstacle_ratio": float(index["difficulty_specs"][difficulty]["target_obstacle_ratio"]),
        "variants": variants,
        "summary": summary,
        "details": details,
        "elapsed_seconds": time.perf_counter() - started,
    }

    print("\nAO P1 ablation summary")
    print(f"{'Variant':<22} {'Success':>9} {'CollFail':>9} {'FeasFound':>10} {'InnerRaw':>9} {'KeepInner':>10} {'FinalColl':>10} {'Runtime/s':>10}")
    print("-" * 98)
    for row in summary:
        print(
            f"{row['variant']:<22} {fmt_percent(row['success_rate']):>9} "
            f"{row['collision_failures']:>9} {fmt_percent(row['feasible_candidate_found_rate']):>10} "
            f"{row['inner_raw_count']:>9} {row['final_kept_inner_count']:>10} "
            f"{fmt_float(row['avg_final_collision_count']):>10} {fmt_float(row['avg_runtime']):>10}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"ao_p1_ablation_{args.scenario}_{timestamp}"
    json_path = args.output_dir / f"{stem}.json"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    details_path = args.output_dir / f"{stem}_details.csv"
    markdown_path = args.output_dir / f"{stem}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    write_csv(summary_path, summary)
    write_csv(details_path, details)
    write_markdown(markdown_path, report)

    print("\nAO P1 ablation completed. Output files:")
    print(f"- Markdown: {markdown_path}")
    print(f"- Summary CSV: {summary_path}")
    print(f"- Details CSV: {details_path}")
    print(f"- JSON: {json_path}")


if __name__ == "__main__":
    main()
