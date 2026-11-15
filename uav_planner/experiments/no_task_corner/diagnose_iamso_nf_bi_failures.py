"""Diagnose where IAMSO-NF-BI loses feasibility on hard fixed maps."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from uav_planner.algorithms.common import seed_value
from uav_planner.config import NO_TASK_MODEL_COST_WEIGHTS
from uav_planner.costs import no_task_model_cost
from uav_planner.geometry import (
    collision_cells_for_path,
    collision_segment_count_for_path,
    smooth_planner_result,
)
from uav_planner.metaheuristic.encoding import decode, encode_from_path, smooth_path
from uav_planner.metaheuristic.feasibility import is_feasible
from uav_planner.metaheuristic.iamso import (
    _evaluate_vector_adaptive,
    _initial_population_multi_source,
    adaptive_weights,
    iamso_optimize,
)

from .dataset import load_map_bundle, no_task_corner_maps_dir, no_task_corner_results_dir
from .seed_ablation import build_bidirectional_seed_paths


FAILED_HARD_IDS = (
    "hard_002", "hard_005", "hard_006", "hard_012", "hard_014", "hard_018",
    "hard_022", "hard_024", "hard_029", "hard_031", "hard_035", "hard_036",
    "hard_047", "hard_048", "hard_050", "hard_055", "hard_057", "hard_059",
    "hard_061", "hard_066", "hard_069", "hard_072", "hard_073", "hard_074",
    "hard_076", "hard_078", "hard_079", "hard_082", "hard_085", "hard_089",
    "hard_094", "hard_097",
)
NO_TASK_LAYER_WEIGHTS = tuple(
    NO_TASK_MODEL_COST_WEIGHTS[key]
    for key in ("inflation", "wind", "rain", "cruise")
)


def _collision_summary(passable, path) -> tuple[int, int]:
    return (
        collision_segment_count_for_path(passable, path),
        len(collision_cells_for_path(passable, path)),
    )


def diagnose_map(map_id: str, *, run_optimizer: bool) -> dict[str, object]:
    layers, passable, meta = load_map_bundle(no_task_corner_maps_dir() / f"{map_id}.npz")
    start = tuple(meta["start"])
    goal = tuple(meta["goal"])
    started = time.perf_counter()
    seed_paths, seed_labels = build_bidirectional_seed_paths(
        passable, layers, no_task_model_cost, start, goal
    )

    seed_rows = []
    for label, path in zip(seed_labels, seed_paths):
        encoded = encode_from_path(path, 6)
        decoded = decode(encoded, start, goal)
        internally_smoothed = smooth_path(decoded)
        raw_segments, raw_cells = _collision_summary(passable, path)
        decoded_segments, decoded_cells = _collision_summary(passable, decoded)
        smooth_segments, smooth_cells = _collision_summary(passable, internally_smoothed)
        seed_rows.append(
            {
                "label": label,
                "raw_safe": raw_segments == 0 and raw_cells == 0,
                "decoded_safe": decoded_segments == 0 and decoded_cells == 0,
                "internal_smooth_safe": smooth_segments == 0 and smooth_cells == 0,
                "decoded_collision_segments": decoded_segments,
                "internal_collision_segments": smooth_segments,
            }
        )

    rng = np.random.default_rng(seed_value(start, goal, 17))
    population = _initial_population_multi_source(
        rng, 24, 6, int(passable.shape[0]), start, goal, seed_paths
    )
    weights = adaptive_weights(layers, passable, NO_TASK_LAYER_WEIGHTS)
    initial_evaluations = [
        _evaluate_vector_adaptive(
            vector,
            passable,
            layers,
            no_task_model_cost,
            start,
            goal,
            weights,
            1000.0,
            2.0,
            1.0,
        )
        for vector in population
    ]
    initial_feasible_count = sum(is_feasible(item) for item in initial_evaluations)

    optimizer_feasible = None
    final_output_safe = None
    solution_source = "not_run"
    if run_optimizer:
        result = iamso_optimize(
            passable,
            layers,
            no_task_model_cost,
            start,
            goal,
            seed_paths=seed_paths,
            fallback_path=None,
            base_weights=NO_TASK_LAYER_WEIGHTS,
            random_seed=seed_value(start, goal, 17),
        )
        optimizer_feasible = bool(result.get("feasible", False))
        solution_source = str(result.get("solution_source", "unknown"))
        final_path = smooth_planner_result(result.get("path"), passable, used_fallback=False)
        final_segments, final_cells = _collision_summary(passable, final_path)
        final_output_safe = final_segments == 0 and final_cells == 0

    raw_safe_count = sum(bool(row["raw_safe"]) for row in seed_rows)
    decoded_safe_count = sum(bool(row["decoded_safe"]) for row in seed_rows)
    internal_safe_count = sum(bool(row["internal_smooth_safe"]) for row in seed_rows)
    if initial_feasible_count > 0:
        failure_stage = "final_output" if final_output_safe is False else "none"
    elif decoded_safe_count == 0:
        failure_stage = "six_waypoint_compression"
    elif internal_safe_count == 0:
        failure_stage = "internal_smoothing"
    else:
        failure_stage = "initial_population_perturbation"
    if run_optimizer and not optimizer_feasible:
        failure_stage = f"{failure_stage}+search_no_recovery"

    return {
        "map_id": map_id,
        "seed_count": len(seed_paths),
        "raw_safe_seed_count": raw_safe_count,
        "decoded_safe_seed_count": decoded_safe_count,
        "internal_safe_seed_count": internal_safe_count,
        "initial_feasible_count": initial_feasible_count,
        "optimizer_feasible": optimizer_feasible,
        "final_output_safe": final_output_safe,
        "solution_source": solution_source,
        "failure_stage": failure_stage,
        "runtime": time.perf_counter() - started,
        "seeds": seed_rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose IAMSO-NF-BI hard-map failures")
    parser.add_argument("--map-id", action="append", dest="map_ids")
    parser.add_argument(
        "--run-optimizer",
        action="store_true",
        help="rerun all 80 IAMSO iterations to distinguish search and output failures",
    )
    args = parser.parse_args(argv)
    map_ids = args.map_ids or list(FAILED_HARD_IDS)
    rows = []
    for index, map_id in enumerate(map_ids, start=1):
        print(f"[{index}/{len(map_ids)}] diagnosing {map_id}", flush=True)
        row = diagnose_map(map_id, run_optimizer=args.run_optimizer)
        rows.append(row)
        print(
            f"  raw={row['raw_safe_seed_count']} decoded={row['decoded_safe_seed_count']} "
            f"internal={row['internal_safe_seed_count']} initial={row['initial_feasible_count']} "
            f"stage={row['failure_stage']} time={row['runtime']:.2f}s",
            flush=True,
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = no_task_corner_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"IAMSO-NF-BI_困难失败阶段诊断_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [key for key in rows[0] if key != "seeds"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
