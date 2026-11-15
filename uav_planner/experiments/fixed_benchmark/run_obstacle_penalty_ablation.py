"""Ablation: obstacle-distance penalty in A*, Dijkstra, RRT.

For each algorithm, compares the original version against a version with
obstacle distance penalty added to the edge cost.  No existing code is
modified.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from uav_planner.costs import model_cost, no_task_model_cost
from uav_planner.costmap import octile_distance
from uav_planner.experiments.no_task_corner.dataset import (
    load_index, load_map_bundle,
    no_task_corner_maps_dir as maps_dir,
)
from uav_planner.geometry import smooth_planner_result
from uav_planner.metrics import calculate_extended_metrics
from uav_planner.algorithms.common import (
    is_cell_available, iter_moves, map_size, reconstruct_path,
)

from uav_planner.algorithms.dijkstra.planner import dijkstra
from uav_planner.algorithms.astar.planner import astar
from uav_planner.algorithms.rrt.planner import rrt

PENALTY_W = 10.0


def _obs_dist(passable):
    return distance_transform_edt((1 - passable).astype(np.uint8))


def _penalty_cost(dist_field, weight):
    def cost(p, q, data):
        base = float(model_cost(p, q, data))
        r, c = int(q[0]), int(q[1])
        return base + weight * max(0.0, 1.0 - float(dist_field[r, c]))
    return cost


def dijkstra_pen(passable, speed_map, cost_func, start, goal):
    dist = _obs_dist(passable)
    pc = _penalty_cost(dist, PENALTY_W)
    pq = [(0.0, start)]
    g = {start: 0.0}
    parent = {}
    size = map_size(passable)
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            break
        x, y = cur
        for dx, dy, _ in iter_moves():
            nx, ny = x + dx, y + dy
            if is_cell_available(passable, (nx, ny), size):
                c = pc(cur, (nx, ny), speed_map)
                ng = g[cur] + c
                if (nx, ny) not in g or ng < g[(nx, ny)]:
                    g[(nx, ny)] = ng
                    parent[(nx, ny)] = cur
                    heapq.heappush(pq, (ng, (nx, ny)))
    return smooth_planner_result(reconstruct_path(parent, start, goal), passable)


def astar_pen(passable, speed_map, cost_func, start, goal):
    dist = _obs_dist(passable)
    pc = _penalty_cost(dist, PENALTY_W)
    pq = [(0, start)]
    g = {start: 0}
    parent = {}
    size = map_size(passable)
    while pq:
        _, (x, y) = heapq.heappop(pq)
        if (x, y) == goal:
            break
        for dx, dy, _ in iter_moves():
            nx, ny = x + dx, y + dy
            if is_cell_available(passable, (nx, ny), size):
                c = pc((x, y), (nx, ny), speed_map)
                ng = g[(x, y)] + c
                if (nx, ny) not in g or ng < g[(nx, ny)]:
                    g[(nx, ny)] = ng
                    parent[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (ng + octile_distance((nx, ny), goal), (nx, ny)))
    return smooth_planner_result(reconstruct_path(parent, start, goal), passable)


ALGOS = {
    "Dijkstra":     dijkstra,
    "A*":           astar,
    "RRT":          rrt,
    "Dijkstra+Pen": dijkstra_pen,
    "A*+Pen":       astar_pen,
}


def _cloned(l):
    return {k: v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v for k, v in l.items()}

def _bucket():
    return {"attempts":0,"path_found":0,"success":0,"no_path":0,"collision_failures":0,
            "distance":0.,"time":0.,"safety":0.,"smoothness":0.,"task_reward":0.,"runtime":0.,"score":0.,"sc":0}

def _avg(t, c):
    return t / c if c else None

def _sum(b):
    a, s, p = b["attempts"], b["success"], b["path_found"]
    return {"attempts":a,"success_rate":s/a if a else 0,"path_found_rate":p/a if a else 0,
            "avg_distance":_avg(b["distance"],s),"avg_time":_avg(b["time"],s),
            "avg_safety":_avg(b["safety"],s),"avg_smoothness":_avg(b["smoothness"],s),
            "avg_score":_avg(b["score"],b["sc"]),"avg_runtime":_avg(b["runtime"],a),
            "no_path_count":b["no_path"],"collision_failures":b["collision_failures"]}


def run(difficulties=None, weight=10.0):
    global PENALTY_W
    PENALTY_W = weight

    idx = load_index()
    want = set(difficulties or [])
    recs = [r for r in idx["maps"] if not want or r["difficulty"] in want]
    sel = sorted({r["difficulty"] for r in recs})
    dl = {k: idx["difficulty_specs"][k]["label"] for k in sel}

    ov = {n: _bucket() for n in ALGOS}
    bd = {d: {n: _bucket() for n in ALGOS} for d in sel}
    details = []
    total = len(recs)
    done = 0
    t0 = time.perf_counter()

    for mi, rec in enumerate(recs, 1):
        layers, passable, _ = load_map_bundle(maps_dir() / rec["file"])
        start, goal = tuple(rec["start"]), tuple(rec["goal"])
        d, lb, sz = rec["difficulty"], rec["difficulty_label"], rec["map_size"]
        print(f"\nMap {mi}/{total}: {rec['id']} ({lb} {sz}x{sz})")

        for nm, fn in ALGOS.items():
            done += 1
            al = _cloned(layers); ap = passable.copy()
            t1 = time.perf_counter()
            st = "success"; err = ""
            try:
                path = fn(ap, al, no_task_model_cost, start, goal)
            except Exception as ex:
                path = None; st = "exception"; err = str(ex)
            rt = time.perf_counter() - t1

            b = ov[nm]; bb = bd[d][nm]
            b["attempts"] += 1; bb["attempts"] += 1; b["runtime"] += rt; bb["runtime"] += rt

            dv = ts = sf = sm = tr = sc = co = None
            if path:
                b["path_found"] += 1; bb["path_found"] += 1
                m = calculate_extended_metrics(path, al, passable=ap, runtime=rt)
                dv, ts, sf, sm, tr, co, sc = m["distance"],m["time"],m["safety"],m["smoothness"],m["task_reward"],m["collision"],m["score"]
                if co and co > 0:
                    b["collision_failures"] += 1; bb["collision_failures"] += 1; st = "collision"
                else:
                    b["success"] += 1; bb["success"] += 1
                    for k, v in [("distance",dv),("time",ts),("safety",sf),("smoothness",sm),("task_reward",tr)]:
                        b[k] += float(v or 0); bb[k] += float(v or 0)
                if sc is not None:
                    b["score"] += float(sc); bb["score"] += float(sc); b["sc"] += 1; bb["sc"] += 1
            else:
                st = "no_path"; b["no_path"] += 1; bb["no_path"] += 1

            details.append({"map_id":rec["id"],"difficulty":d,"algorithm":nm,"status":st,
                            "success":1 if st=="success" else 0,"runtime":rt,"collision_count":co,
                            "distance":dv,"time":ts,"safety":sf,"smoothness":sm,"score":sc,"error":err})

            cs = "-" if co is None else str(int(co))
            el = time.perf_counter() - t0
            print(f"  [{done:>3}/{total*len(ALGOS):>3}] {nm:<14} status={st:<9} col={cs:<3} time={rt:>6.3f}s  elapsed={el:>7.1f}s")

    ov_s = {n: {"algorithm":n, **_sum(b)} for n, b in ov.items()}
    bd_s = {d: {n: {"algorithm":n, **_sum(b)} for n, b in bd[d].items()} for d in sel}
    return {"dataset_version":idx["dataset_version"],"map_count":total,"selected_difficulties":sel,
            "difficulty_labels":dl,"penalty_weight":weight,
            "summary":{"overall":ov_s,"by_difficulty":bd_s},"details":details}


def save(report):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    od = Path(__file__).parent / "results"; od.mkdir(parents=True, exist_ok=True)
    (od / f"obs_penalty_ablation_{ts}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _p(v): return "-" if v is None else f"{v:.2%}"
    def _v(v): return "-" if v is None else f"{v:.4f}"

    lines = [f"# Obstacle Penalty Ablation (w={report['penalty_weight']})", "",
             f"Maps: {report['map_count']}  |  Difficulties: {', '.join(report['selected_difficulties'])}", "",
             "| Algorithm | Success | Path Found | Avg Dist | Avg Time | Avg Safety | Avg Smooth | Avg Score | Runtime | No-Path | ColFail |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summary"]["overall"].values():
        lines.append(f"| {row['algorithm']} | {_p(row['success_rate'])} | {_p(row['path_found_rate'])} "
                     f"| {_v(row['avg_distance'])} | {_v(row['avg_time'])} | {_p(row['avg_safety'])} "
                     f"| {_v(row['avg_smoothness'])} | {_v(row['avg_score'])} | {_v(row['avg_runtime'])} "
                     f"| {row['no_path_count']} | {row['collision_failures']} |")
    for diff, rows in report["summary"]["by_difficulty"].items():
        lbl = report["difficulty_labels"].get(diff, diff)
        lines += ["", f"## {lbl} ({diff})", "",
            "| Algorithm | Success | Path Found | Avg Dist | Avg Time | Avg Safety | Avg Smooth | Avg Score | Runtime | No-Path | ColFail |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows.values():
            lines.append(f"| {row['algorithm']} | {_p(row['success_rate'])} | {_p(row['path_found_rate'])} "
                         f"| {_v(row['avg_distance'])} | {_v(row['avg_time'])} | {_p(row['avg_safety'])} "
                         f"| {_v(row['avg_smoothness'])} | {_v(row['avg_score'])} | {_v(row['avg_runtime'])} "
                         f"| {row['no_path_count']} | {row['collision_failures']} |")
    (od / f"obs_penalty_ablation_{ts}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults saved to {od}")


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--difficulty", nargs="*", choices=["simple","medium","hard"])
    pa.add_argument("--penalty-weight", type=float, default=10.0)
    a = pa.parse_args()

    print(f"{'='*60}\n  Obstacle Penalty Ablation  w={a.penalty_weight}\n  Dijkstra|A*|RRT vs Dijkstra+Pen|A*+Pen\n{'='*60}")
    rpt = run(a.difficulty, a.penalty_weight)
    save(rpt)
    print("\n=== Success Rates ===")
    for n, r in rpt["summary"]["overall"].items():
        print(f"  {n:<14} Success={r['success_rate']:.2%}  ColFail={r['collision_failures']}  Runtime={r['avg_runtime']:.3f}s")

