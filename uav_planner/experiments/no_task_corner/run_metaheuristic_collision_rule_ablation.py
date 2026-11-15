"""Run MSO and IAMSO with obstacle-boundary contact allowed."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import uav_planner.geometry as geometry
from uav_planner.experiments.no_task_corner.runner import (
    evaluate_dataset,
    print_comparison_table,
)


BOUNDARY_EPSILON = 1e-9
DEFAULT_DATASET = Path(__file__).parent / "datasets" / "medium60_density20_reachable_100"


def _enable_interior_collision_rule():
    original = geometry._segment_intersects_rect

    def intersects_interior(a, b, row, col, margin=0.0):
        return original(a, b, row, col, margin=margin - BOUNDARY_EPSILON)

    geometry._segment_intersects_rect = intersects_interior
    return original


def _self_check():
    corner = np.ones((2, 2), dtype=bool)
    corner[0, 1] = False
    path = [(0.0, 0.0), (1.0, 1.0)]
    assert geometry.collision_cells_for_path(corner, path) == {(0, 1)}

    original = _enable_interior_collision_rule()
    try:
        assert not geometry.collision_cells_for_path(corner, path)
    finally:
        geometry._segment_intersects_rect = original


def parse_args():
    parser = argparse.ArgumentParser(description="MSO/IAMSO 障碍内部碰撞标准消融")
    parser.add_argument("--algorithm", nargs="+", choices=("MSO", "IAMSO"), default=["MSO", "IAMSO"])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--record-id", nargs="*")
    return parser.parse_args()


def main():
    _self_check()
    args = parse_args()
    original = _enable_interior_collision_rule()
    try:
        report = evaluate_dataset(
            difficulties=["medium20"],
            algorithm_names=args.algorithm,
            record_ids=set(args.record_id) if args.record_id else None,
            dataset_dir=args.dataset_dir,
        )
    finally:
        geometry._segment_intersects_rect = original

    print("\nSearch, feasibility checks, and final evaluation all used the interior-only collision rule.")
    print_comparison_table(report)


if __name__ == "__main__":
    main()
