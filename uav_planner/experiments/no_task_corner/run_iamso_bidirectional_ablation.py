"""Run IAMSO-NF with bidirectional grid-search seed variants."""

from __future__ import annotations

import argparse

from uav_planner.algorithms.common import seed_value
from uav_planner.config import NO_TASK_MODEL_COST_WEIGHTS
from uav_planner.geometry import PathResult, smooth_planner_result
from uav_planner.metaheuristic import iamso_optimize

from . import runner
from .seed_ablation import build_bidirectional_seed_paths


ALGORITHM_NAME = "IAMSO-NF-BI"
NO_FINAL_SMOOTHING_ALGORITHM_NAME = "IAMSO-NF-BI-NOFS"
NO_TASK_LAYER_WEIGHTS = tuple(
    NO_TASK_MODEL_COST_WEIGHTS[key]
    for key in ("inflation", "wind", "rain", "cruise")
)


def iamso_bidirectional_ablation_planner(passable, layers, cost_func, start, goal):
    seed_paths, seed_labels = build_bidirectional_seed_paths(passable, layers, cost_func, start, goal)
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
    )
    path = result.get("path")
    if path is None:
        return None
    planned_path = smooth_planner_result(path, passable, used_fallback=False)
    planned_path.solution_source = result.get("solution_source", "unknown")
    planned_path.feasible = bool(result.get("feasible", False))
    planned_path.seed_labels = seed_labels
    return planned_path


def iamso_bidirectional_no_final_smoothing_planner(passable, layers, cost_func, start, goal):
    seed_paths, seed_labels = build_bidirectional_seed_paths(passable, layers, cost_func, start, goal)
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
    )
    path = result.get("path")
    if path is None:
        return None
    planned_path = PathResult(
        [(float(point[0]), float(point[1])) for point in path],
        used_fallback=False,
    )
    planned_path.solution_source = result.get("solution_source", "unknown")
    planned_path.feasible = bool(result.get("feasible", False))
    planned_path.seed_labels = seed_labels
    return planned_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the IAMSO-NF-BI fixed-map ablation")
    parser.add_argument(
        "--difficulty",
        choices=("simple", "medium", "hard"),
        default="medium",
        help="fixed-map difficulty to evaluate (default: medium)",
    )
    parser.add_argument(
        "--skip-final-smoothing",
        action="store_true",
        help="return the IAMSO-validated path without adapter-level smoothing",
    )
    args = parser.parse_args(argv)
    algorithm_name = NO_FINAL_SMOOTHING_ALGORITHM_NAME if args.skip_final_smoothing else ALGORITHM_NAME
    planner_func = (
        iamso_bidirectional_no_final_smoothing_planner
        if args.skip_final_smoothing
        else iamso_bidirectional_ablation_planner
    )
    runner.ALGORITHMS = {"iamso_nf_bi": (algorithm_name, planner_func)}
    report = runner.evaluate_dataset(difficulties=[args.difficulty], algorithm_names=[algorithm_name])
    report["ablation"] = {
        "name": (
            "no_safe_fallback_bidirectional_grid_seeds_no_final_smoothing"
            if args.skip_final_smoothing
            else "no_safe_fallback_bidirectional_grid_seeds"
        ),
        "fallback_path_enabled": False,
        "seed_paths_enabled": True,
        "description": (
            "Return the internally validated IAMSO path without adapter-level smoothing."
            if args.skip_final_smoothing
            else "Bidirectional A*/distance-field A*/Theta*/Dijkstra/dynamic A*; RRT/RRT* unchanged."
        ),
    }
    runner.print_comparison_table(report)
    paths = runner.save_report(report)
    print("\nIAMSO bidirectional seed ablation completed:")
    for label, file_path in paths.items():
        print(f"- {label}: {file_path}")


if __name__ == "__main__":
    main()
