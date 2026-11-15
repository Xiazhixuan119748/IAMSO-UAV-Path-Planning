"""Run IAMSO-NF-BI without RRT/RRT* seed paths."""

from __future__ import annotations

from uav_planner.algorithms.common import seed_value
from uav_planner.config import NO_TASK_MODEL_COST_WEIGHTS
from uav_planner.geometry import smooth_planner_result
from uav_planner.metaheuristic import iamso_optimize

from . import runner
from .seed_ablation import build_bidirectional_grid_only_seed_paths


ALGORITHM_NAME = "IAMSO-NF-NORRT"
NO_TASK_LAYER_WEIGHTS = tuple(
    NO_TASK_MODEL_COST_WEIGHTS[key]
    for key in ("inflation", "wind", "rain", "cruise")
)


def iamso_no_rrt_ablation_planner(passable, layers, cost_func, start, goal):
    seed_paths, seed_labels = build_bidirectional_grid_only_seed_paths(
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
    )
    path = result.get("path")
    if path is None:
        return None
    planned_path = smooth_planner_result(path, passable, used_fallback=False)
    planned_path.solution_source = result.get("solution_source", "unknown")
    planned_path.feasible = bool(result.get("feasible", False))
    planned_path.seed_labels = seed_labels
    return planned_path


def main() -> None:
    runner.ALGORITHMS = {"iamso_no_rrt": (ALGORITHM_NAME, iamso_no_rrt_ablation_planner)}
    report = runner.evaluate_dataset(difficulties=["medium"], algorithm_names=[ALGORITHM_NAME])
    report["ablation"] = {
        "name": "no_safe_fallback_bidirectional_grid_only",
        "fallback_path_enabled": False,
        "seed_paths_enabled": True,
        "description": "Remove RRT/RRT*; retain all bidirectional grid seeds and no task-value weight.",
    }
    runner.print_comparison_table(report)
    paths = runner.save_report(report)
    print("\nIAMSO no-RRT seed ablation completed:")
    for label, file_path in paths.items():
        print(f"- {label}: {file_path}")


if __name__ == "__main__":
    main()
