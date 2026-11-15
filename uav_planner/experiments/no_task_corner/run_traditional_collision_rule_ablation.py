"""Compare closed-boundary and obstacle-interior collision rules."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from uav_planner.algorithms import astar, dijkstra, rrt
from uav_planner.costs import no_task_model_cost
from uav_planner.experiments.no_task_corner.dataset import load_map_bundle, no_task_corner_maps_dir
from uav_planner.experiments.no_task_corner.runner import clone_layers, collect_records
from uav_planner.geometry import collision_cells_for_path
from uav_planner.metaheuristic.encoding import smooth_path
from uav_planner.metrics import calculate_collision_count


BOUNDARY_EPSILON = 1e-9
ALGORITHMS = {
    "Dijkstra": dijkstra,
    "A*": astar,
    "RRT": rrt,
}
PLANNER_MODULES = (
    "uav_planner.algorithms.dijkstra.planner",
    "uav_planner.algorithms.astar.planner",
    "uav_planner.algorithms.rrt.planner",
)


def obstacle_interior_cells(passable, path):
    """Ignore boundary contact and return only obstacles whose interior is crossed."""
    return collision_cells_for_path(passable, path, margin=-BOUNDARY_EPSILON)


def _self_check():
    corner = np.ones((2, 2), dtype=bool)
    corner[0, 1] = False
    corner_path = [(0.0, 0.0), (1.0, 1.0)]
    assert collision_cells_for_path(corner, corner_path)
    assert not obstacle_interior_cells(corner, corner_path)

    crossing = np.ones((3, 3), dtype=bool)
    crossing[1, 1] = False
    assert obstacle_interior_cells(crossing, [(1.0, 0.0), (1.0, 2.0)]) == {(1, 1)}


def _identity_smoothing(path, passable, used_fallback=False):
    if path is None:
        return None
    return [(float(row), float(col)) for row, col in path]


def _patch_smoothing():
    patched = []
    for module_name in PLANNER_MODULES:
        module = __import__(module_name, fromlist=["smooth_planner_result"])
        patched.append((module, module.smooth_planner_result))
        module.smooth_planner_result = _identity_smoothing
    return patched


def _restore_smoothing(patched):
    for module, original in patched:
        module.smooth_planner_result = original


def parse_args():
    parser = argparse.ArgumentParser(description="传统算法障碍边界碰撞标准消融")
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--samples-per-segment", nargs="+", type=int, default=[10])
    return parser.parse_args()


def main():
    _self_check()
    args = parse_args()
    maps_root = args.dataset_dir or no_task_corner_maps_dir()
    index_path = maps_root / "index.json" if args.dataset_dir else None
    index, records = collect_records([args.difficulty], index_path)
    if not records:
        raise RuntimeError(f"no maps found for difficulty {args.difficulty!r}")
    obstacle_ratio = float(
        index["difficulty_specs"][args.difficulty]["target_obstacle_ratio"]
    )
    sample_counts = list(dict.fromkeys(args.samples_per_segment))
    if any(samples < 1 for samples in sample_counts):
        raise ValueError("samples per segment must be at least 1")

    stats = {
        samples: {
            name: {
                "path_found": 0,
                "current_success": 0,
                "interior_success": 0,
                "boundary_only": 0,
                "interior_failures": 0,
                "out_of_map": 0,
                "exceptions": 0,
            }
            for name in ALGORITHMS
        }
        for samples in sample_counts
    }
    started = time.perf_counter()

    patched = _patch_smoothing()
    try:
        for record in records:
            layers, passable, _ = load_map_bundle(maps_root / record["file"])
            empty_map = np.ones_like(passable, dtype=bool)
            start = tuple(record["start"])
            goal = tuple(record["goal"])

            for name, algorithm in ALGORITHMS.items():
                try:
                    raw_path = algorithm(
                        passable.copy(),
                        clone_layers(layers),
                        no_task_model_cost,
                        start,
                        goal,
                    )
                except Exception:
                    for samples in sample_counts:
                        stats[samples][name]["exceptions"] += 1
                    continue
                if not raw_path:
                    continue

                for samples in sample_counts:
                    bucket = stats[samples][name]
                    path = smooth_path(raw_path, samples_per_segment=samples)
                    bucket["path_found"] += 1
                    current_collision = calculate_collision_count(path, passable) > 0
                    outside = calculate_collision_count(path, empty_map) > 0
                    interior_collision = bool(obstacle_interior_cells(passable, path))
                    interior_success = not outside and not interior_collision

                    bucket["current_success"] += int(not current_collision)
                    bucket["interior_success"] += int(interior_success)
                    bucket["boundary_only"] += int(current_collision and interior_success)
                    bucket["interior_failures"] += int(interior_collision)
                    bucket["out_of_map"] += int(outside)
    finally:
        _restore_smoothing(patched)

    attempts = len(records)
    print(
        f"Dataset={index['dataset_name']}, difficulty={args.difficulty}, "
        f"maps={attempts}, obstacle ratio={obstacle_ratio:.0%}\n"
        "Search paths are fixed; smoothing density and the result collision rule are ablated."
    )
    for samples in sample_counts:
        print(f"\nSamples per segment: {samples}")
        print(
            "Algorithm    Found    Current success    Interior-only success    "
            "Recovered boundary-only    Interior failures"
        )
        for name, bucket in stats[samples].items():
            print(
                f"{name:<10} {bucket['path_found']:>3}/{attempts:<3} "
                f"{bucket['current_success']:>3}/{attempts:<3} "
                f"({bucket['current_success'] / attempts:>6.2%})   "
                f"{bucket['interior_success']:>3}/{attempts:<3} "
                f"({bucket['interior_success'] / attempts:>6.2%})   "
                f"{bucket['boundary_only']:>8} {bucket['interior_failures']:>20}"
            )
            if bucket["out_of_map"] or bucket["exceptions"]:
                print(
                    f"  out_of_map={bucket['out_of_map']}, exceptions={bucket['exceptions']}"
                )
    print(f"Elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
