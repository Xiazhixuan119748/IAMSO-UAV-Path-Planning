"""Ablate a soft diagonal corner penalty for the mode-6 traditional planners.

This is deliberately an experiment-only runner.  It does not patch or modify
the production planners.  For A* and Dijkstra, ``lambda`` is added to the
search edge cost for each blocked orthogonal neighbour of a diagonal move.
The diagonal move remains legal, so this is a soft preference rather than a
new collision rule.  RRT is run unchanged as a control because it has no
grid-edge cost in its production implementation.

Every returned path is smoothed by the production ``smooth_planner_result``
and is judged by the production strict collision metric (margin=0).
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

# Allow both ``python -m ...`` from the project root and direct script use.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_planner.algorithms.astar.planner import astar as production_astar
from uav_planner.algorithms.common import (
    edge_cost,
    is_cell_available,
    iter_moves,
    map_size,
    reconstruct_path,
)
from uav_planner.algorithms.dijkstra.planner import dijkstra as production_dijkstra
from uav_planner.algorithms.rrt.planner import rrt as production_rrt
from uav_planner.costmap import octile_distance
from uav_planner.costs import no_task_model_cost
from uav_planner.experiments.no_task_corner.dataset import (
    load_index,
    load_map_bundle,
    no_task_corner_maps_dir,
)
from uav_planner.experiments.no_task_corner.runner import clone_layers
from uav_planner.geometry import smooth_planner_result
from uav_planner.metrics import calculate_collision_count


DEFAULT_LAMBDAS = (0.0, 1.0, 5.0, 10.0)
ALGORITHM_ORDER = ("Dijkstra", "A*", "RRT")


def corner_risk(passable, p: tuple[int, int], q: tuple[int, int]) -> int:
    """Count blocked side cells for a one-cell diagonal move.

    The endpoints are checked by the search before this function is called,
    so only the two orthogonal side cells can contribute to the risk.
    """

    if abs(p[0] - q[0]) != 1 or abs(p[1] - q[1]) != 1:
        return 0
    return int(not bool(passable[p[0], q[1]])) + int(
        not bool(passable[q[0], p[1]])
    )


def _penalized_edge_cost(
    passable,
    base_cost: Callable[[Any, Any, Any], float],
    weight: float,
) -> Callable[[Any, Any, Any], float]:
    """Wrap the existing cost without changing any map-layer costs."""

    def cost(p, q, layers):
        return float(base_cost(p, q, layers)) + float(weight) * corner_risk(
            passable, p, q
        )

    return cost


def _dijkstra_with_penalty(
    passable,
    layers,
    cost_func,
    start,
    goal,
    weight: float,
):
    """Production Dijkstra loop with only the experimental edge wrapper."""

    if weight == 0.0:
        # Use the exact production control path for lambda=0.
        return production_dijkstra(passable, layers, cost_func, start, goal)

    wrapped_cost = _penalized_edge_cost(passable, cost_func, weight)
    pq = [(0.0, start)]
    distances = {start: 0.0}
    parent = {}
    size = map_size(passable)

    while pq:
        current_cost, current = heapq.heappop(pq)
        if current == goal:
            break

        x, y = current
        for dx, dy, _length in iter_moves():
            neighbor = (x + dx, y + dy)
            if not is_cell_available(passable, neighbor, size):
                continue
            step = edge_cost(wrapped_cost, current, neighbor, layers)
            new_cost = current_cost + step
            if neighbor not in distances or new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))

    return smooth_planner_result(reconstruct_path(parent, start, goal), passable)


def _astar_with_penalty(
    passable,
    layers,
    cost_func,
    start,
    goal,
    weight: float,
):
    """Production A* loop with only the experimental edge wrapper."""

    if weight == 0.0:
        # Use the exact production control path for lambda=0.
        return production_astar(passable, layers, cost_func, start, goal)

    wrapped_cost = _penalized_edge_cost(passable, cost_func, weight)
    pq = [(0.0, start)]
    costs = {start: 0.0}
    parent = {}
    size = map_size(passable)

    while pq:
        _, current = heapq.heappop(pq)
        if current == goal:
            break

        x, y = current
        for dx, dy, _length in iter_moves():
            neighbor = (x + dx, y + dy)
            if not is_cell_available(passable, neighbor, size):
                continue
            step = edge_cost(wrapped_cost, current, neighbor, layers)
            new_cost = costs[current] + step
            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                parent[neighbor] = current
                priority = new_cost + octile_distance(neighbor, goal)
                heapq.heappush(pq, (priority, neighbor))

    return smooth_planner_result(reconstruct_path(parent, start, goal), passable)


def _self_check() -> None:
    """Catch accidental changes to the definition of a soft corner risk."""

    passable = __import__("numpy").ones((3, 3), dtype=bool)
    passable[0, 1] = False
    assert corner_risk(passable, (0, 0), (1, 1)) == 1
    assert corner_risk(passable, (0, 0), (0, 1)) == 0
    passable[1, 0] = False
    assert corner_risk(passable, (0, 0), (1, 1)) == 2


def _new_bucket() -> dict[str, Any]:
    return {
        "attempts": 0,
        "path_found": 0,
        "success": 0,
        "no_path": 0,
        "collision_failures": 0,
        "exceptions": 0,
        "runtime_total": 0.0,
        "changed_from_lambda0": 0,
    }


def _finish_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    attempts = int(bucket["attempts"])
    return {
        "attempts": attempts,
        "path_found": int(bucket["path_found"]),
        "path_found_rate": int(bucket["path_found"]) / attempts if attempts else 0.0,
        "success": int(bucket["success"]),
        "success_rate": int(bucket["success"]) / attempts if attempts else 0.0,
        "no_path": int(bucket["no_path"]),
        "collision_failures": int(bucket["collision_failures"]),
        "exceptions": int(bucket["exceptions"]),
        "avg_runtime_seconds": float(bucket["runtime_total"]) / attempts
        if attempts
        else 0.0,
        "changed_from_lambda0": int(bucket["changed_from_lambda0"]),
    }


def _normalise_path(path: Iterable[Any] | None) -> tuple[tuple[float, float], ...] | None:
    if not path:
        return None
    return tuple((float(point[0]), float(point[1])) for point in path)


def _evaluate_path(path, passable) -> tuple[str, int | None]:
    """Apply the unchanged final mode-6 decision to a returned path."""

    if not path:
        return "no_path", None
    collision_count = int(calculate_collision_count(path, passable))
    return ("success" if collision_count == 0 else "collision"), collision_count


def _run_one(
    algorithm: Callable[..., Any],
    passable,
    layers,
    start,
    goal,
    weight: float,
) -> tuple[Any, str, int | None, float, str]:
    run_started = time.perf_counter()
    error = ""
    try:
        if algorithm is production_rrt:
            path = algorithm(
                passable, layers, no_task_model_cost, start, goal
            )
        else:
            path = algorithm(
                passable, layers, no_task_model_cost, start, goal, weight
            )
    except Exception as exc:  # Keep one bad map from changing the dataset.
        path = None
        error = repr(exc)
    runtime = time.perf_counter() - run_started
    if error:
        return path, "exception", None, runtime, error
    status, collision_count = _evaluate_path(path, passable)
    return path, status, collision_count, runtime, error


def _validate_dataset(index: dict[str, Any], records: list[dict[str, Any]]) -> None:
    spec = index["difficulty_specs"]["medium"]
    if len(records) != 100:
        raise RuntimeError(f"expected 100 medium maps, found {len(records)}")
    if int(spec["size"]) != 60 or float(spec["target_obstacle_ratio"]) != 0.15:
        raise RuntimeError(f"unexpected medium specification: {spec}")
    for record in records:
        if tuple(record["start"]) != (0, 0) or tuple(record["goal"]) != (59, 59):
            raise RuntimeError(f"unexpected fixed endpoints in {record['id']}")


def run_experiment(
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    dataset_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the requested lambda values on the unchanged medium fixed set."""

    _self_check()
    if any(not math.isfinite(value) or value < 0.0 for value in lambdas):
        raise ValueError("lambda values must be finite and non-negative")
    if not lambdas:
        raise ValueError("at least one lambda value is required")

    maps_root = Path(dataset_dir) if dataset_dir is not None else no_task_corner_maps_dir()
    index = load_index(maps_root / "index.json")
    records = [record for record in index["maps"] if record["difficulty"] == "medium"]
    _validate_dataset(index, records)

    # Keep first occurrence order while rejecting duplicate rows.
    unique_lambdas = tuple(dict.fromkeys(float(value) for value in lambdas))
    searchers = {
        "Dijkstra": _dijkstra_with_penalty,
        "A*": _astar_with_penalty,
    }
    stats = {
        str(weight): {name: _new_bucket() for name in ALGORITHM_ORDER}
        for weight in unique_lambdas
    }
    details: list[dict[str, Any]] = []
    baseline_paths: dict[tuple[str, str], tuple[tuple[float, float], ...] | None] = {}
    started = time.perf_counter()

    for map_number, record in enumerate(records, start=1):
        layers, passable, _ = load_map_bundle(maps_root / record["file"])
        start = tuple(record["start"])
        goal = tuple(record["goal"])
        if verbose:
            print(
                f"地图 {map_number:>3}/100: {record['id']} ",
                end="",
                flush=True,
            )

        for weight in unique_lambdas:
            weight_key = str(weight)
            for name in ("Dijkstra", "A*"):
                bucket = stats[weight_key][name]
                bucket["attempts"] += 1
                algo_passable = passable.copy()
                algo_layers = clone_layers(layers)
                path, status, collision_count, runtime, error = _run_one(
                    searchers[name],
                    algo_passable,
                    algo_layers,
                    start,
                    goal,
                    weight,
                )
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

                normalised = _normalise_path(path)
                baseline_key = (record["id"], name)
                if weight == 0.0:
                    baseline_paths[baseline_key] = normalised
                elif normalised != baseline_paths.get(baseline_key):
                    bucket["changed_from_lambda0"] += 1

                details.append(
                    {
                        "map_id": record["id"],
                        "algorithm": name,
                        "lambda": weight,
                        "status": status,
                        "success": int(status == "success"),
                        "path_found": int(status in {"success", "collision"}),
                        "collision_count": collision_count,
                        "runtime_seconds": runtime,
                        "path_changed_from_lambda0": bool(
                            weight != 0.0 and normalised != baseline_paths.get(baseline_key)
                        ),
                        "error": error,
                    }
                )

        # RRT has no lambda-dependent edge cost; run the unchanged control once.
        rrt_bucket = stats[str(unique_lambdas[0])]["RRT"]
        rrt_bucket["attempts"] += 1
        rrt_path, rrt_status, rrt_collision, rrt_runtime, rrt_error = _run_one(
            production_rrt,
            passable.copy(),
            clone_layers(layers),
            start,
            goal,
            0.0,
        )
        rrt_bucket["runtime_total"] += rrt_runtime
        if rrt_status == "exception":
            rrt_bucket["exceptions"] += 1
        elif rrt_status == "no_path":
            rrt_bucket["no_path"] += 1
        elif rrt_status == "collision":
            rrt_bucket["path_found"] += 1
            rrt_bucket["collision_failures"] += 1
        else:
            rrt_bucket["path_found"] += 1
            rrt_bucket["success"] += 1

        details.append(
            {
                "map_id": record["id"],
                "algorithm": "RRT",
                "lambda": None,
                "status": rrt_status,
                "success": int(rrt_status == "success"),
                "path_found": int(rrt_status in {"success", "collision"}),
                "collision_count": rrt_collision,
                "runtime_seconds": rrt_runtime,
                "path_changed_from_lambda0": False,
                "error": rrt_error,
                "control_note": "unchanged production RRT; lambda not applicable",
            }
        )
        if verbose:
            print(" done", flush=True)

    # RRT control is copied into each lambda row for a compact comparison table.
    rrt_summary = _finish_bucket(stats[str(unique_lambdas[0])]["RRT"])
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for weight in unique_lambdas:
        weight_key = str(weight)
        summaries[weight_key] = {
            name: _finish_bucket(stats[weight_key][name])
            for name in ("Dijkstra", "A*")
        }
        summaries[weight_key]["RRT"] = dict(rrt_summary)
        summaries[weight_key]["RRT"]["lambda_applicable"] = False

    return {
        "experiment": "traditional soft corner penalty ablation",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "name": index["dataset_name"],
            "version": index["dataset_version"],
            "directory": str(maps_root),
            "difficulty": "medium",
            "map_count": len(records),
            "map_size": 60,
            "target_obstacle_ratio": 0.15,
            "start": [0, 0],
            "goal": [59, 59],
        },
        "lambdas": list(unique_lambdas),
        "corner_penalty": (
            "lambda * (blocked orthogonal side cells) for each diagonal grid move; "
            "diagonal remains legal"
        ),
        "smoothing": "production smooth_planner_result / Catmull-Rom / samples_per_segment=10",
        "collision_rule": "production calculate_collision_count / collision_cells_for_path margin=0",
        "runtime_budget": "production planner loops and RRT iteration budget unchanged",
        "algorithms": list(ALGORITHM_ORDER),
        "rrt_control": rrt_summary,
        "summary": summaries,
        "details": details,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _format_rate(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def print_summary(report: dict[str, Any]) -> None:
    print("\n传统算法软切角代价消融（模式6 / 中等15% / 100张固定地图）")
    print("判定：严格 margin=0；平滑：生产 Catmull-Rom，samples_per_segment=10")
    print("\nlambda    Dijkstra 成功       A* 成功            RRT 对照")
    print("".join(["-" for _ in range(68)]))
    for weight in report["lambdas"]:
        row = report["summary"][str(weight)]
        print(
            f"{weight:>6g}    "
            f"{row['Dijkstra']['success']:>3}/100 ({_format_rate(row['Dijkstra']['success_rate'])})   "
            f"{row['A*']['success']:>3}/100 ({_format_rate(row['A*']['success_rate'])})   "
            f"{row['RRT']['success']:>3}/100 ({_format_rate(row['RRT']['success_rate'])})"
        )
    print("\n详细计数（每行按 λ）：")
    print("lambda  algorithm  found  success  collision  no_path  changed_from_0")
    for weight in report["lambdas"]:
        row = report["summary"][str(weight)]
        for name in ALGORITHM_ORDER:
            value = row[name]
            print(
                f"{weight:>6g}  {name:<9}  {value['path_found']:>5}  "
                f"{value['success']:>7}  {value['collision_failures']:>9}  "
                f"{value['no_path']:>7}  {value['changed_from_lambda0']:>14}"
            )


def save_report(report: dict[str, Any], output_dir: Path | None = None) -> tuple[Path, Path]:
    target = output_dir or (Path(__file__).resolve().parent / "results")
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = target / f"traditional_corner_penalty_ablation_{stamp}.json"
    md_path = target / f"traditional_corner_penalty_ablation_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Traditional soft corner-penalty ablation",
        "",
        "- Dataset: mode 6 medium, target obstacle ratio 15%, 100 fixed maps",
        "- Smoothing: production Catmull-Rom (`samples_per_segment=10`)",
        "- Final collision rule: production strict rule (`margin=0`)",
        "- RRT: unchanged control; lambda is not applicable",
        "",
        "| lambda | Algorithm | Path found | Success | Collision failures | No path | Changed from lambda=0 | Avg runtime (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for weight in report["lambdas"]:
        for name in ALGORITHM_ORDER:
            row = report["summary"][str(weight)][name]
            lines.append(
                f"| {weight:g} | {name} | {row['path_found']}/{row['attempts']} "
                f"({row['path_found_rate']:.2%}) | {row['success']}/{row['attempts']} "
                f"({row['success_rate']:.2%}) | {row['collision_failures']} | {row['no_path']} | "
                f"{row['changed_from_lambda0']} | {row['avg_runtime_seconds']:.6f} |"
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
        help="non-negative soft corner weights (default: 0 1 5 10)",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_experiment(
        tuple(args.lambdas),
        dataset_dir=args.dataset_dir,
        verbose=not args.quiet,
    )
    print_summary(report)
    json_path, md_path = save_report(report, args.output_dir)
    print(f"\nJSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
