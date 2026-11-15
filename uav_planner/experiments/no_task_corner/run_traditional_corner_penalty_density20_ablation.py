"""Run the traditional-planner corner-penalty ablation on two 20% datasets.

The production planners are not modified.  Dijkstra and A* use the same soft
diagonal corner penalty as the medium-15% ablation.  RRT keeps its production
sampling, steering, seed and iteration budget, but validates each tree edge
and final goal-connection segment before accepting it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_planner.algorithms.common import map_size, seed_value
from uav_planner.costs import no_task_model_cost
from uav_planner.experiments.no_task_corner.dataset import (
    load_index,
    load_map_bundle,
    no_task_corner_maps_dir,
)
from uav_planner.experiments.no_task_corner.run_traditional_corner_penalty_ablation import (
    _astar_with_penalty,
    _dijkstra_with_penalty,
    _evaluate_path,
    _finish_bucket,
    _new_bucket,
    corner_risk,
)
from uav_planner.experiments.no_task_corner.runner import clone_layers
from uav_planner.geometry import segment_has_collision, smooth_planner_result


DEFAULT_LAMBDAS = (0.2, 0.3, 0.4, 0.5)
ALGORITHM_ORDER = ("Dijkstra", "A*", "RRT-edge-valid")
EXPERIMENT_ROOT = Path(__file__).resolve().parent

DATASETS = {
    "medium20": {
        "root": EXPERIMENT_ROOT / "datasets" / "medium60_density20_reachable_100",
        "difficulty": "medium20",
        "label": "中等20%障碍",
        "size": 60,
        "start": (0, 0),
        "goal": (59, 59),
    },
    "hard20": {
        "root": no_task_corner_maps_dir(),
        "difficulty": "hard",
        "label": "困难20%障碍",
        "size": 90,
        "start": (0, 0),
        "goal": (89, 89),
    },
}


def _nearest_node_index(nodes, target):
    return min(range(len(nodes)), key=lambda index: math.dist(nodes[index], target))


def _steer(from_node, to_node, step_size):
    distance = math.dist(from_node, to_node)
    if distance <= step_size:
        return float(to_node[0]), float(to_node[1])
    ratio = step_size / max(distance, 1e-9)
    return (
        float(from_node[0] + (to_node[0] - from_node[0]) * ratio),
        float(from_node[1] + (to_node[1] - from_node[1]) * ratio),
    )


def _reconstruct_rrt(nodes, parents, goal_index):
    path = []
    index = goal_index
    while index is not None:
        path.append(nodes[index])
        index = parents[index]
    path.reverse()
    return path


def edge_valid_rrt(passable, layers, cost_func, start, goal):
    """Production RRT with segment validity added at both accepted edge types."""

    size = map_size(passable)
    if not passable[start] or not passable[goal]:
        return None

    rng = random.Random(seed_value(start, goal, 11))
    start_f = float(start[0]), float(start[1])
    goal_f = float(goal[0]), float(goal[1])
    nodes = [start_f]
    parents = [None]

    step_size = max(2.0, size / 18.0)
    goal_radius = step_size * 1.5
    max_iterations = max(1200, size * size * 4)
    goal_sample_rate = 0.18

    for _ in range(max_iterations):
        if rng.random() < goal_sample_rate:
            sample = goal_f
        else:
            sample = (rng.uniform(0, size - 1), rng.uniform(0, size - 1))

        nearest_index = _nearest_node_index(nodes, sample)
        nearest_node = nodes[nearest_index]
        new_node = _steer(nearest_node, sample, step_size)

        row = int(round(new_node[0]))
        col = int(round(new_node[1]))
        if not (0 <= row < size and 0 <= col < size and passable[row, col]):
            continue
        if segment_has_collision(passable, nearest_node, new_node):
            continue

        nodes.append(new_node)
        parents.append(nearest_index)
        new_index = len(nodes) - 1

        if (
            math.dist(new_node, goal_f) <= goal_radius
            and not segment_has_collision(passable, new_node, goal_f)
        ):
            nodes.append(goal_f)
            parents.append(new_index)
            raw_path = _reconstruct_rrt(nodes, parents, len(nodes) - 1)
            return smooth_planner_result(raw_path, passable)

    return None


def _self_check() -> None:
    import numpy as np

    passable = np.ones((3, 3), dtype=bool)
    passable[0, 1] = False
    assert corner_risk(passable, (0, 0), (1, 1)) == 1
    assert segment_has_collision(passable, (0.0, 0.0), (1.0, 1.0))


def _validate_dataset(
    key: str,
    index: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    config = DATASETS[key]
    difficulty = config["difficulty"]
    spec = index["difficulty_specs"][difficulty]
    if len(records) != 100:
        raise RuntimeError(f"{key}: expected 100 maps, found {len(records)}")
    if int(spec["size"]) != config["size"]:
        raise RuntimeError(f"{key}: unexpected map size: {spec}")
    if not math.isclose(float(spec["target_obstacle_ratio"]), 0.2):
        raise RuntimeError(f"{key}: unexpected obstacle ratio: {spec}")
    for record in records:
        if tuple(record["start"]) != config["start"]:
            raise RuntimeError(f"{record['id']}: unexpected start")
        if tuple(record["goal"]) != config["goal"]:
            raise RuntimeError(f"{record['id']}: unexpected goal")


def _record_outcome(bucket, status: str, runtime: float) -> None:
    bucket["attempts"] += 1
    bucket["runtime_total"] += runtime
    if status == "exception":
        bucket["exceptions"] += 1
    elif status == "no_path":
        bucket["no_path"] += 1
    elif status == "collision":
        bucket["path_found"] += 1
        bucket["collision_failures"] += 1
    else:
        bucket["path_found"] += 1
        bucket["success"] += 1


def _run_grid_planner(algorithm, passable, layers, start, goal, weight):
    started = time.perf_counter()
    try:
        path = algorithm(
            passable,
            layers,
            no_task_model_cost,
            start,
            goal,
            weight,
        )
        error = ""
    except Exception as exc:
        path = None
        error = repr(exc)
    runtime = time.perf_counter() - started
    if error:
        return path, "exception", None, runtime, error
    status, collision_count = _evaluate_path(path, passable)
    return path, status, collision_count, runtime, error


def _run_rrt(passable, layers, start, goal):
    started = time.perf_counter()
    try:
        path = edge_valid_rrt(
            passable,
            layers,
            no_task_model_cost,
            start,
            goal,
        )
        error = ""
    except Exception as exc:
        path = None
        error = repr(exc)
    runtime = time.perf_counter() - started
    if error:
        return path, "exception", None, runtime, error
    status, collision_count = _evaluate_path(path, passable)
    return path, status, collision_count, runtime, error


def run_dataset(key: str, lambdas: tuple[float, ...], verbose: bool) -> dict[str, Any]:
    config = DATASETS[key]
    maps_root = config["root"]
    index = load_index(maps_root / "index.json")
    records = [
        record
        for record in index["maps"]
        if record["difficulty"] == config["difficulty"]
    ]
    _validate_dataset(key, index, records)

    stats = {
        str(weight): {name: _new_bucket() for name in ALGORITHM_ORDER}
        for weight in lambdas
    }
    rrt_bucket = _new_bucket()
    details = []
    searchers = {
        "Dijkstra": _dijkstra_with_penalty,
        "A*": _astar_with_penalty,
    }
    started = time.perf_counter()

    for number, record in enumerate(records, start=1):
        layers, passable, _ = load_map_bundle(maps_root / record["file"])
        start = tuple(record["start"])
        goal = tuple(record["goal"])
        if verbose:
            print(
                f"[{key}] 地图 {number:>3}/100: {record['id']} ",
                end="",
                flush=True,
            )

        for weight in lambdas:
            for name, algorithm in searchers.items():
                path, status, collisions, runtime, error = _run_grid_planner(
                    algorithm,
                    passable.copy(),
                    clone_layers(layers),
                    start,
                    goal,
                    weight,
                )
                _record_outcome(stats[str(weight)][name], status, runtime)
                details.append(
                    {
                        "map_id": record["id"],
                        "algorithm": name,
                        "lambda": weight,
                        "status": status,
                        "success": int(status == "success"),
                        "collision_count": collisions,
                        "runtime_seconds": runtime,
                        "error": error,
                    }
                )

        path, status, collisions, runtime, error = _run_rrt(
            passable.copy(),
            clone_layers(layers),
            start,
            goal,
        )
        _record_outcome(rrt_bucket, status, runtime)
        details.append(
            {
                "map_id": record["id"],
                "algorithm": "RRT-edge-valid",
                "lambda": None,
                "status": status,
                "success": int(status == "success"),
                "collision_count": collisions,
                "runtime_seconds": runtime,
                "error": error,
            }
        )
        if verbose:
            print("done", flush=True)

    rrt_summary = _finish_bucket(rrt_bucket)
    summary = {}
    for weight in lambdas:
        summary[str(weight)] = {
            name: _finish_bucket(stats[str(weight)][name])
            for name in ("Dijkstra", "A*")
        }
        summary[str(weight)]["RRT-edge-valid"] = dict(rrt_summary)
        summary[str(weight)]["RRT-edge-valid"]["lambda_applicable"] = False

    return {
        "dataset_key": key,
        "dataset_name": index["dataset_name"],
        "dataset_version": index["dataset_version"],
        "dataset_directory": str(maps_root),
        "difficulty": config["difficulty"],
        "label": config["label"],
        "map_count": len(records),
        "map_size": config["size"],
        "target_obstacle_ratio": 0.2,
        "start": list(config["start"]),
        "goal": list(config["goal"]),
        "summary": summary,
        "rrt_control": rrt_summary,
        "details": details,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_experiment(lambdas: tuple[float, ...], verbose: bool = True) -> dict[str, Any]:
    _self_check()
    unique_lambdas = tuple(dict.fromkeys(float(value) for value in lambdas))
    if not unique_lambdas:
        raise ValueError("at least one lambda is required")
    if any(not math.isfinite(value) or value < 0.0 for value in unique_lambdas):
        raise ValueError("lambda values must be finite and non-negative")

    started = time.perf_counter()
    datasets = {
        key: run_dataset(key, unique_lambdas, verbose)
        for key in ("medium20", "hard20")
    }
    return {
        "experiment": "traditional corner penalty on medium20 and hard20",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "lambdas": list(unique_lambdas),
        "algorithms": list(ALGORITHM_ORDER),
        "corner_penalty": (
            "lambda * blocked orthogonal side-cell count for a diagonal grid move; "
            "diagonal remains legal"
        ),
        "rrt_change": (
            "production RRT plus strict segment validation for each tree extension "
            "and final goal connection"
        ),
        "rrt_budget": "production seed, step size, goal radius, sample rate and max iterations",
        "smoothing": "production Catmull-Rom, samples_per_segment=10",
        "collision_rule": "production strict calculate_collision_count, margin=0",
        "datasets": datasets,
        "elapsed_seconds": time.perf_counter() - started,
    }


def print_summary(report: dict[str, Any]) -> None:
    for key in ("medium20", "hard20"):
        dataset = report["datasets"][key]
        print(
            f"\n{dataset['label']} / {dataset['map_size']}x{dataset['map_size']} / "
            f"{dataset['map_count']}张固定地图"
        )
        print("lambda    Dijkstra        A*              RRT-edge-valid")
        print("-" * 66)
        for weight in report["lambdas"]:
            row = dataset["summary"][str(weight)]
            print(
                f"{weight:>6g}    "
                f"{row['Dijkstra']['success']:>3}/100 ({row['Dijkstra']['success_rate']:>6.2%})   "
                f"{row['A*']['success']:>3}/100 ({row['A*']['success_rate']:>6.2%})   "
                f"{row['RRT-edge-valid']['success']:>3}/100 "
                f"({row['RRT-edge-valid']['success_rate']:>6.2%})"
            )
        print("\n详细计数")
        print("lambda  algorithm       found  success  collision  no_path  exception")
        for weight in report["lambdas"]:
            for name in ALGORITHM_ORDER:
                row = dataset["summary"][str(weight)][name]
                print(
                    f"{weight:>6g}  {name:<15} {row['path_found']:>5}  "
                    f"{row['success']:>7}  {row['collision_failures']:>9}  "
                    f"{row['no_path']:>7}  {row['exceptions']:>9}"
                )


def save_report(report: dict[str, Any], output_dir: Path | None = None) -> tuple[Path, Path]:
    target = output_dir or (EXPERIMENT_ROOT / "results")
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = target / f"traditional_corner_penalty_density20_ablation_{stamp}.json"
    md_path = target / f"traditional_corner_penalty_density20_ablation_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Traditional corner-penalty ablation on 20% obstacle maps",
        "",
        "- Grid planners: soft diagonal corner penalty",
        "- RRT: production parameters with tree-edge and goal-edge segment validation",
        "- Smoothing: production Catmull-Rom (`samples_per_segment=10`)",
        "- Final collision rule: production strict rule (`margin=0`)",
    ]
    for key in ("medium20", "hard20"):
        dataset = report["datasets"][key]
        lines.extend(
            [
                "",
                f"## {dataset['label']}",
                "",
                "| lambda | Algorithm | Path found | Success | Collision failures | No path | Exceptions | Avg runtime (s) |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for weight in report["lambdas"]:
            for name in ALGORITHM_ORDER:
                row = dataset["summary"][str(weight)][name]
                lines.append(
                    f"| {weight:g} | {name} | {row['path_found']}/{row['attempts']} "
                    f"({row['path_found_rate']:.2%}) | {row['success']}/{row['attempts']} "
                    f"({row['success_rate']:.2%}) | {row['collision_failures']} | "
                    f"{row['no_path']} | {row['exceptions']} | {row['avg_runtime_seconds']:.6f} |"
                )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lambda",
        dest="lambdas",
        nargs="+",
        type=float,
        default=list(DEFAULT_LAMBDAS),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_experiment(tuple(args.lambdas), verbose=not args.quiet)
    print_summary(report)
    json_path, md_path = save_report(report, args.output_dir)
    print(f"\nJSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
