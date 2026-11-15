"""Run the IAMSO no-fallback ablation on the fixed medium dataset."""

from __future__ import annotations

from uav_planner.algorithms.common import seed_value
from uav_planner.algorithms.iamso.planner import _seed_paths
from uav_planner.geometry import smooth_planner_result
from uav_planner.metaheuristic import iamso_optimize

from . import runner


ALGORITHM_NAME = "IAMSO-NF"


def iamso_no_fallback_planner(passable, layers, cost_func, start, goal):
    """Run IAMSO without preserving a raw safe path for final fallback."""
    seed_paths = _seed_paths(passable, layers, cost_func, start, goal)
    result = iamso_optimize(
        passable,
        layers,
        cost_func,
        start,
        goal,
        seed_paths=seed_paths,
        fallback_path=None,
        random_seed=seed_value(start, goal, 17),
    )
    path = result.get("path")
    if path is None:
        return None

    planned_path = smooth_planner_result(path, passable, used_fallback=False)
    planned_path.solution_source = result.get("solution_source", "unknown")
    planned_path.feasible = bool(result.get("feasible", False))
    return planned_path


def main() -> None:
    # The runner resolves algorithms from this process-local registry reference.
    runner.ALGORITHMS = {"iamso_nf": (ALGORITHM_NAME, iamso_no_fallback_planner)}
    report = runner.evaluate_dataset(
        difficulties=["medium"],
        algorithm_names=[ALGORITHM_NAME],
    )
    report["ablation"] = {
        "name": "no_safe_fallback",
        "fallback_path_enabled": False,
        "seed_paths_enabled": True,
        "description": "Disable only the raw safe-path final fallback; retain multi-source seeds.",
    }
    runner.print_comparison_table(report)
    paths = runner.save_report(report)
    print("\nIAMSO-NF experiment completed:")
    for label, file_path in paths.items():
        print(f"- {label}: {file_path}")


if __name__ == "__main__":
    main()
