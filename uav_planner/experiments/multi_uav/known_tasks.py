"""Four-corner known-task benchmark using the original registered planners."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import heapq
import json
import math
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

from uav_planner.registry import ALGORITHMS
from .simulation import Config, Simulation


def scenario(size, seed, task_count):
    from uav_planner.map_generator import generate_map

    if size < 8 or task_count < 1:
        raise ValueError("size >= 8 and tasks >= 1 required")
    corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
    center = (size // 2, size // 2)
    stations = corners + [center]
    layers, _ = generate_map(corners[0], corners[-1], seed=seed, size=size)
    # Protect exact requested locations; do not shift launches to nearby free cells.
    for row, col in stations:
        layers["O"][max(0, row - 1):row + 2, max(0, col - 1):col + 2] = 0
    from uav_planner.map_generator import _make_inflation, _compatible_speed_map
    layers["I"] = _make_inflation(layers["O"])
    layers["speed_map"] = _compatible_speed_map(layers["I"], layers["R"])
    components, _ = label(1 - layers["O"])
    if len({int(components[p]) for p in stations}) != 1:
        raise ValueError("stations disconnected on this seed; select another seed (no hidden corridor repair)")
    mask = components == components[corners[0]]
    cells = [tuple(map(int, p)) for p in np.argwhere(mask) if tuple(p) not in stations]
    if len(cells) < task_count:
        raise ValueError("not enough task cells")
    rng = np.random.default_rng(seed + 10000)
    tasks = [cells[i] for i in rng.choice(len(cells), task_count, replace=False)]
    layers = dict(layers, V=np.zeros_like(layers["V"]), tasks=list(tasks))
    return layers, mask, stations, tasks


def run_one(size, seed, layers, mask, stations, tasks, name, planner, wall_budget):
    """Greedy nearest feasible dispatch; exact event times, distance-limited sorties."""
    random.seed(seed)
    np.random.seed(seed)
    config = Config(size=size, tasks=len(tasks), wall_budget=wall_budget or 600)
    engine = Simulation(config, layers, mask, stations, tasks, planner)
    engine.deadline = time.perf_counter() + wall_budget if wall_budget else math.inf
    began = time.perf_counter()
    pending = set(range(len(tasks)))
    completed = set()
    vehicles = [dict(position=p, remaining=float(size), distance=0.0, charges=0,
                     max_sortie=0.0, visited={p}, status="idle") for p in stations[:4]]
    queue = [(0.0, i, None, None) for i in range(4)]
    heapq.heapify(queue)
    events, flights = [], []
    now = 0.0
    budget_hit = False

    def leg(a, b):
        value = engine.plan(a, b)
        if value is None:
            return None
        length = sum(math.dist(x, y) for x, y in zip(value.points, value.points[1:]))
        return value, length

    def returning(point, available):
        # Shortlist stations by geometric lower bound; stop at the first feasible return.
        for station in sorted(stations, key=lambda p: (math.dist(point, p), p)):
            value = leg(point, station)
            if value is not None and value[1] <= available + 1e-8:
                return station, value
        return None

    while queue:
        now, index, arrival_task, arrival_mode = heapq.heappop(queue)
        u = vehicles[index]
        if arrival_task is not None:
            completed.add(arrival_task)
            u["visited"] = set()
            events.append(dict(time=now, uav=index, kind="task_completed", task=arrival_task))
        if arrival_mode == "charge":
            u["charges"] += 1
            u["remaining"] = float(size)
            u["visited"].add(u["position"])
            events.append(dict(time=now, uav=index, kind="charged"))
        if time.perf_counter() >= engine.deadline:
            budget_hit = True
            break
        choice = None
        for task in sorted(pending, key=lambda j: (math.dist(u["position"], tasks[j]), j)):
            if math.dist(u["position"], tasks[task]) > u["remaining"] + 1e-8:
                continue
            outbound = leg(u["position"], tasks[task])
            if outbound is None or outbound[1] > u["remaining"] + 1e-8:
                continue
            back = returning(tasks[task], u["remaining"] - outbound[1])
            if back is not None and outbound[1] + back[1][1] <= u["remaining"] + 1e-8:
                choice = (tasks[task], outbound, task, "task")
                pending.remove(task)
                break
        if choice is None:
            if not pending:
                if u["position"] in stations:
                    u["status"] = "finished"
                    continue
                back = returning(u["position"], u["remaining"])
                if back and back[1][1] <= u["remaining"] + 1e-8:
                    choice = (back[0], back[1], None, "final_return")
            else:
                # Prevent endless zero-time charges or station-to-station cycles.
                for station in sorted(stations, key=lambda p: (math.dist(u["position"], p), p)):
                    if station in u["visited"]:
                        continue
                    value = leg(u["position"], station)
                    if value is not None and value[1] <= u["remaining"] + 1e-8:
                        choice = (station, value, None, "charge")
                        break
        if time.perf_counter() >= engine.deadline:
            budget_hit = True
            break
        if choice is None:
            u["status"] = "blocked" if pending or u["position"] not in stations else "finished"
            continue
        goal, (path, length), task, mode = choice
        if length > u["remaining"] + 1e-7:
            raise RuntimeError("sortie range exceeded")
        flights.append(dict(uav=index, start_time=now, end_time=now + path.duration,
                            mode=mode, task=task, start=u["position"], goal=goal,
                            distance=length, points=path.points))
        u["remaining"] -= length
        u["distance"] += length
        u["max_sortie"] = max(u["max_sortie"], size - u["remaining"])
        u["position"] = goal
        u["status"] = "flying"
        heapq.heappush(queue, (now + path.duration, index, task, "charge" if mode == "charge" else mode))

    # Budget-censored runs are not assigned misleading partial-flight totals.
    success = len(completed) == len(tasks) and not budget_hit and all(u["position"] in stations for u in vehicles)
    result = dict(algorithm=name, seed=seed, size=size, task_count=len(tasks), uavs=4,
                  actual_obstacle_ratio=float(np.mean(layers["O"])),
                  station_count=5, range_limit=float(size), success=success,
                  task_success_rate=len(completed) / len(tasks), completed=len(completed),
                  charging_count=sum(u["charges"] for u in vehicles),
                  total_distance=None if budget_hit else sum(u["distance"] for u in vehicles),
                  total_runtime=None if budget_hit else now,
                  termination="planning_budget" if budget_hit else "success" if success else "blocked",
                  max_sortie_distance=max(u["max_sortie"] for u in vehicles),
                  planning_calls=engine.planning_calls, planning_failures=engine.planning_failures,
                  planning_seconds=engine.planning_seconds, wall_seconds=time.perf_counter() - began)
    return result, flights, events, vehicles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wall-budget", type=float, default=0, help="Planning seconds; 0 means unlimited (default)")
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.wall_budget) or args.wall_budget < 0:
        parser.error("--wall-budget must be finite and nonnegative; 0 means unlimited")
    planners = dict(ALGORITHMS.values())
    names = args.algorithms or list(planners)
    if any(name not in planners for name in names):
        parser.error(f"algorithms must be from {list(planners)}")
    layers, mask, stations, tasks = scenario(args.size, args.seed, args.tasks)
    output = args.output or Path("outputs/multi_uav") / f"known_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "scenario.npz", **layers, stations=stations, searchable=mask)
    results = []
    for name in names:
        print(f"Running {name}: {args.size}x{args.size}, 4 UAVs, {len(tasks)} known tasks", flush=True)
        result, flights, events, vehicles = run_one(args.size, args.seed, layers, mask, stations, tasks,
                                                  name, planners[name], args.wall_budget)
        results.append(result)
        slug = name.replace("*", "Star")
        (output / f"{slug}.json").write_text(json.dumps(dict(summary=result, flights=flights,
                events=events, vehicles=[dict(u, visited=list(u["visited"])) for u in vehicles]), indent=2), encoding="utf-8")
        with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(layers["O"], cmap="gray_r", vmin=0, vmax=1)
        ax.scatter([p[1] for p in tasks], [p[0] for p in tasks], c="red", marker="x", label="Known tasks")
        ax.scatter([p[1] for p in stations], [p[0] for p in stations], c="black", marker="s", label="Stations / launches")
        for i in range(4):
            segments = [f for f in flights if f["uav"] == i]
            for j, flight in enumerate(segments):
                ax.plot([p[1] for p in flight["points"]], [p[0] for p in flight["points"]],
                        color=f"C{i}", linewidth=1, label=f"UAV {i + 1}" if j == 0 else None)
        ax.set(title=f"{name}: {result['completed']}/{len(tasks)} | {result['termination']}", xlabel="Column", ylabel="Row")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"{slug}.png", dpi=140)
        plt.close(fig)
        print(json.dumps(result), flush=True)
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
