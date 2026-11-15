"""Run one reproducible multi-UAV scenario and export auditable results."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from uav_planner.algorithms.iamso import iamso_planner
from .planners import safe_astar
from .simulation import Config, Simulation, make_scenario


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def export(sim, result, output, animate=False):
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "events.jsonl").open("w", encoding="utf-8") as stream:
        for event in sim.events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    write_csv(output / "trajectories.csv", [dict(uav=u.id, **row) for u in sim.uavs for row in u.trajectory])
    write_csv(output / "progress.csv", sim.frames)
    np.savez_compressed(output / "scenario.npz", O=sim.layers["O"], I=sim.layers["I"],
                        W=sim.layers["W"], R=sim.layers["R"], speed_map=sim.layers["speed_map"],
                        searchable=sim.searchable, truth=sim._truth, stations=sim.stations,
                        covered=sim.covered)
    fig, (ax, progress) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax.imshow(sim.layers["R"], cmap="Blues", vmin=0, vmax=1, alpha=0.55)
    obstacles = np.ma.masked_where(sim.layers["O"] == 0, sim.layers["O"])
    ax.imshow(obstacles, cmap="gray_r", vmin=0, vmax=1)
    covered = np.ma.masked_where(~sim.covered, sim.covered)
    ax.imshow(covered, cmap="Greens", alpha=0.25, vmin=0, vmax=1)
    ax.scatter(sim._truth[:, 1], sim._truth[:, 0], marker="x", c="crimson", label="Task truth (evaluation only)")
    ax.scatter([p[1] for p in sim.stations], [p[0] for p in sim.stations], marker="s", c="black", s=65, label="Charging station")
    for u in sim.uavs:
        ax.plot([p["col"] for p in u.trajectory], [p["row"] for p in u.trajectory], linewidth=1.3, label=f"UAV {u.id}")
    ax.set(title=f"{result['planner']} | {result['termination']}", xlabel="Column", ylabel="Row")
    ax.legend(fontsize=7, loc="upper right")
    progress.plot([p["time"] for p in sim.frames], [p["discovered"] for p in sim.frames], label="Discovered")
    progress.plot([p["time"] for p in sim.frames], [p["completed"] for p in sim.frames], label="Completed")
    progress.axhline(sim.config.tasks, linestyle=":", color="gray", label="Total tasks")
    progress.set(xlabel="Simulation time (relative units)", ylabel="Tasks", ylim=(0, sim.config.tasks + 0.5))
    progress.legend()
    fig.tight_layout()
    fig.savefig(output / "overview.png", dpi=160)
    plt.close(fig)
    if animate:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(sim.layers["R"], cmap="Blues", alpha=0.5, vmin=0, vmax=1)
        ax.imshow(obstacles, cmap="gray_r", vmin=0, vmax=1)
        ax.scatter([p[1] for p in sim.stations], [p[0] for p in sim.stations], marker="s", c="black")
        tasks = ax.scatter([], [], c="crimson", marker="x")
        lines = [ax.plot([], [], label=f"UAV {u.id}")[0] for u in sim.uavs]
        dots = [ax.plot([], [], "o", color=line.get_color())[0] for line in lines]
        ax.legend(loc="upper right", fontsize=7)
        frames = np.unique(np.linspace(0, len(sim.frames) - 1, min(150, len(sim.frames))).astype(int))

        def update(index):
            now = sim.frames[index]["time"]
            visible = [info["position"] for info in sim.known.values() if info["discovered"] <= now]
            tasks.set_offsets(np.asarray([(p[1], p[0]) for p in visible]).reshape(-1, 2))
            for u, line, dot in zip(sim.uavs, lines, dots):
                history = u.trajectory[:index + 1]
                line.set_data([p["col"] for p in history], [p["row"] for p in history])
                dot.set_data([history[-1]["col"]], [history[-1]["row"]])
            ax.set_title(f"t={now:.1f} | completed={sim.frames[index]['completed']}/{sim.config.tasks}")

        animation = FuncAnimation(fig, update, frames=frames, interval=100)
        animation.save(output / "replay.gif", writer=PillowWriter(fps=10))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", choices=("iamso", "astar"), default="iamso")
    for name, default, kind in (("size", 60, int), ("seed", 42, int), ("uavs", 3, int),
                                ("tasks", 10, int), ("sensor-radius", 5.0, float),
                                ("battery", 65.0, float), ("reserve", 5.0, float),
                                ("max-time", 1500.0, float), ("wall-budget", 300.0, float),
                                ("candidates", 2, int)):
        parser.add_argument(f"--{name}", type=kind, default=default)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = Config(**{name: getattr(args, name) for name in (
        "size", "seed", "uavs", "tasks", "sensor_radius", "battery", "reserve", "max_time", "wall_budget", "candidates")})
    layers, reachable, stations, truth = make_scenario(config)
    sim = Simulation(config, layers, reachable, stations, truth,
                     iamso_planner if args.planner == "iamso" else safe_astar)
    print(f"Running {args.planner}: {config.uavs} UAVs, {config.tasks} hidden tasks, seed={config.seed}", flush=True)
    result = dict(planner=args.planner, strategy="rolling_true_cost_idle_dispatch", **sim.run())
    output = args.output or Path("outputs/multi_uav") / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.planner}_seed{config.seed}"
    export(sim, result, output, args.gif)
    print(json.dumps({k: v for k, v in result.items() if k not in ("config", "tasks", "final_uavs")}, indent=2))
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
