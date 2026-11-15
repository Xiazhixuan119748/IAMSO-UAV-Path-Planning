"""Run a non-overlapping IAMSO-NF-BI-CA versus MSO map shard."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import time
from pathlib import Path

from uav_planner.algorithms.common import seed_value
from uav_planner.algorithms.mso.planner import mso_planner
from uav_planner.config import NO_TASK_MODEL_COST_WEIGHTS
from uav_planner.geometry import smooth_planner_result
from uav_planner.metaheuristic import iamso_optimize

from . import runner
from uav_planner.metaheuristic.collision_aware import collision_aware_initial_population
from .dataset import no_task_corner_results_dir
from .seed_ablation import build_bidirectional_seed_paths


ALGORITHM_NAME = "IAMSO-NF-BI-CA"
MSO_NAME = "MSO"
WAYPOINT_COUNT = 6
NO_TASK_LAYER_WEIGHTS = tuple(
    NO_TASK_MODEL_COST_WEIGHTS[key]
    for key in ("inflation", "wind", "rain", "cruise")
)


def iamso_collision_aware_planner(passable, layers, cost_func, start, goal):
    seed_paths, seed_labels = build_bidirectional_seed_paths(
        passable, layers, cost_func, start, goal
    )
    result = iamso_optimize(
        passable,
        layers,
        cost_func,
        start,
        goal,
        seed_paths=seed_paths,
        fallback_path=None,
        base_weights=NO_TASK_LAYER_WEIGHTS,
        random_seed=seed_value(start, goal, 17),
        waypoint_count=WAYPOINT_COUNT,
        initial_population_builder=collision_aware_initial_population,
    )
    path = result.get("path")
    if path is None:
        return None
    planned_path = smooth_planner_result(path, passable, used_fallback=False)
    planned_path.solution_source = result.get("solution_source", "unknown")
    planned_path.feasible = bool(result.get("feasible", False))
    planned_path.seed_labels = seed_labels
    return planned_path


def shard_ids(difficulty: str, start_index: int, end_index: int) -> set[str]:
    return {f"{difficulty}_{index:03d}" for index in range(start_index, end_index + 1)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a CA-vs-MSO fixed-map shard")
    parser.add_argument("--difficulty", choices=("simple", "medium", "hard"), required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument(
        "--run-id",
        help="shared identifier used to isolate all shards from one benchmark run",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.start_index <= args.end_index <= 100:
        parser.error("indices must satisfy 1 <= start <= end <= 100")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        parser.error("run-id may contain only letters, numbers, dot, underscore, and hyphen")

    ids = shard_ids(args.difficulty, args.start_index, args.end_index)
    runner.ALGORITHMS = {
        "mso": (MSO_NAME, mso_planner),
        "iamso_nf_bi_ca": (ALGORITHM_NAME, iamso_collision_aware_planner),
    }
    report = runner.evaluate_dataset(
        difficulties=[args.difficulty],
        algorithm_names=[MSO_NAME, ALGORITHM_NAME],
        record_ids=ids,
    )
    expected_pairs = {
        (map_id, algorithm)
        for map_id in ids
        for algorithm in (MSO_NAME, ALGORITHM_NAME)
    }
    actual_pairs = [
        (detail["map_id"], detail["algorithm"])
        for detail in report["details"]
    ]
    pair_counts = Counter(actual_pairs)
    duplicate_pairs = sorted(pair for pair, count in pair_counts.items() if count != 1)
    if set(actual_pairs) != expected_pairs or duplicate_pairs:
        missing = sorted(expected_pairs - set(actual_pairs))
        unexpected = sorted(set(actual_pairs) - expected_pairs)
        raise RuntimeError(
            "shard result mismatch: "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicate_pairs}"
        )

    report["run_id"] = run_id
    report["ablation"] = {
        "name": "nf_bi_collision_aware_compression_6_waypoints",
        "waypoint_count": WAYPOINT_COUNT,
        "fallback_path_enabled": False,
        "seed_paths_enabled": True,
        "bidirectional_seed_paths_enabled": True,
        "collision_aware_compression_enabled": True,
        "final_smoothing_enabled": True,
        "description": (
            "Keep the IAMSO-NF-BI search budget and six intermediate waypoints; "
            "replace only equal-arc seed encoding with collision-aware encoding."
        ),
    }
    report["optimizer_random_seeds"] = {
        MSO_NAME: seed_value((0, 0), (int(report["details"][0]["map_size"]) - 1,) * 2, 7),
        ALGORITHM_NAME: seed_value(
            (0, 0), (int(report["details"][0]["map_size"]) - 1,) * 2, 17
        ),
    }
    report["shard"] = {
        "difficulty": args.difficulty,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "map_ids": sorted(ids),
    }
    output_dir = no_task_corner_results_dir() / "ablation_shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"IAMSO-NF-BI-CA_vs_MSO_{run_id}_{args.difficulty}_"
        f"{args.start_index:03d}-{args.end_index:03d}"
    )
    output_path = output_dir / f"{stem}.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=runner._json_default),
        encoding="utf-8",
    )
    runner.print_comparison_table(report)
    print(f"Shard JSON: {output_path}")


if __name__ == "__main__":
    main()
