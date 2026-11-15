"""Ablate clearance and curvature constraints for traditional planners."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from uav_planner.costs import no_task_model_cost
from uav_planner.experiments.no_task_corner.dataset import load_map_bundle, no_task_corner_maps_dir
from uav_planner.experiments.no_task_corner.run_traditional_corner_cut_ablation import (
    ALGORITHMS as STRICT_ALGORITHMS,
)
from uav_planner.experiments.no_task_corner.runner import clone_layers, collect_records
from uav_planner.metaheuristic.encoding import sample_path
from uav_planner.metrics import calculate_collision_count
from uav_planner.visualization_utils import compute_path_curvature


MAIN_CLEARANCE = 0.25
MAIN_MAX_CURVATURE = 3.0
ALGORITHMS = {name: algorithm for name, algorithm in STRICT_ALGORITHMS.items() if name != "DFS"}
CLEARANCE_THRESHOLDS = (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5)
CURVATURE_THRESHOLDS = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)


def path_constraints(path, passable):
    sampled = sample_path(path, step=0.25)
    points = np.asarray(sampled, dtype=float)
    center_distance = distance_transform_edt(np.asarray(passable, dtype=bool))
    interpolated = map_coordinates(
        center_distance,
        [points[:, 0], points[:, 1]],
        order=1,
        mode="constant",
        cval=0.0,
    )
    min_clearance = max(0.0, float(np.min(interpolated)) - 0.5)
    curvature = compute_path_curvature(sampled)
    max_curvature = float(np.max(np.abs(curvature))) if len(curvature) else 0.0
    return min_clearance, max_curvature


def _rate(count, attempts):
    return count / attempts if attempts else 0.0


def main():
    records = collect_records(["medium"])[1]
    stats = {
        name: {
            "attempts": 0,
            "collision_free": 0,
            "clearance_only": 0,
            "curvature_only": 0,
            "all_constraints": 0,
            "clearance_values": [],
            "curvature_values": [],
            "grid": {(clearance, curvature): 0 for clearance in CLEARANCE_THRESHOLDS for curvature in CURVATURE_THRESHOLDS},
        }
        for name in ALGORITHMS
    }

    for record in records:
        layers, passable, _ = load_map_bundle(no_task_corner_maps_dir() / record["file"])
        start = tuple(record["start"])
        goal = tuple(record["goal"])
        for name, algorithm in ALGORITHMS.items():
            bucket = stats[name]
            bucket["attempts"] += 1
            try:
                path = algorithm(passable.copy(), clone_layers(layers), no_task_model_cost, start, goal)
            except Exception:
                path = None
            if not path or calculate_collision_count(path, passable) != 0:
                continue

            bucket["collision_free"] += 1
            clearance, max_curvature = path_constraints(path, passable)
            bucket["clearance_values"].append(clearance)
            bucket["curvature_values"].append(max_curvature)
            clearance_ok = clearance >= MAIN_CLEARANCE
            curvature_ok = max_curvature <= MAIN_MAX_CURVATURE
            bucket["clearance_only"] += int(clearance_ok)
            bucket["curvature_only"] += int(curvature_ok)
            bucket["all_constraints"] += int(clearance_ok and curvature_ok)
            for clearance_limit in CLEARANCE_THRESHOLDS:
                for curvature_limit in CURVATURE_THRESHOLDS:
                    bucket["grid"][(clearance_limit, curvature_limit)] += int(
                        clearance >= clearance_limit and max_curvature <= curvature_limit
                    )

    print(
        f"Main constraints: clearance >= {MAIN_CLEARANCE:.2f} cells, "
        f"max curvature <= {MAIN_MAX_CURVATURE:.2f}/cell"
    )
    print("Algorithm    Collision-free    +Clearance    +Curvature    All three")
    for name, bucket in stats.items():
        attempts = bucket["attempts"]
        print(
            f"{name:<10} {_rate(bucket['collision_free'], attempts):>14.2%} "
            f"{_rate(bucket['clearance_only'], attempts):>13.2%} "
            f"{_rate(bucket['curvature_only'], attempts):>13.2%} "
            f"{_rate(bucket['all_constraints'], attempts):>12.2%}"
        )

    print("\nCombined-constraint sensitivity")
    for name, bucket in stats.items():
        attempts = bucket["attempts"]
        print(name)
        for clearance_limit in CLEARANCE_THRESHOLDS:
            values = [
                _rate(bucket["grid"][(clearance_limit, curvature_limit)], attempts)
                for curvature_limit in CURVATURE_THRESHOLDS
            ]
            formatted = "  ".join(
                f"k<={limit:.2f}: {value:.2%}" for limit, value in zip(CURVATURE_THRESHOLDS, values)
            )
            print(f"  clearance>={clearance_limit:.2f}: {formatted}")

    print("\nObserved distributions on collision-free paths")
    for name, bucket in stats.items():
        clearance = np.asarray(bucket["clearance_values"], dtype=float)
        curvature = np.asarray(bucket["curvature_values"], dtype=float)
        if not len(clearance):
            print(f"{name}: no collision-free paths")
            continue
        cq = np.quantile(clearance, [0.0, 0.25, 0.5, 0.75, 1.0])
        kq = np.quantile(curvature, [0.0, 0.25, 0.5, 0.75, 1.0])
        print(f"{name}: clearance min/q25/median/q75/max = " + "/".join(f"{v:.3f}" for v in cq))
        print(f"{name}: curvature min/q25/median/q75/max = " + "/".join(f"{v:.3f}" for v in kq))


if __name__ == "__main__":
    main()
