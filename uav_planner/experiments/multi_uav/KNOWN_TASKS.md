# Four-Corner Known-Task Comparison

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uav_planner.experiments.multi_uav.known_tasks --size 30 --tasks 10 --seed 42 --wall-budget 600
```

The original six registry planners are used without a safe-A* replacement:
Dijkstra, A*, RRT, MSO, AO, IAMSO. All use `no_task_model_cost` and identical
continuous collision checks. Public task positions are known to dispatch from
time zero. The value layer is zero to avoid additional task-attraction rewards.
The path adapter receives no task list because its contract is point-to-point;
the dispatcher receives all task coordinates directly.

## Scenario

- Four UAVs start at (0,0), (0,n-1), (n-1,0), (n-1,n-1).
- Five stations: these four corners plus (n//2,n//2). For an even grid, this is
  the lower-right member of the central four cells.
- Exact station cells and their one-cell neighborhoods are cleared before
  recomputing inflation and speed. No connecting corridors are carved. A seed
  with disconnected stations is rejected explicitly.
- Ten distinct known tasks are sampled from the stations' connected component,
  excluding station locations. Scene arrays and task coordinates are saved.
- A full charge permits cumulative polyline travel of at most n grid-distance
  units, not merely each individual leg. Thus n=30 means a range limit of 30;
  corner coordinate separation on an axis is 29. Initial charge is not counted.
- Charging is instantaneous and not queued. Service time is zero for this
  initial routing comparison. Weather affects travel speed and path cost, not
  the distance-based range allowance. No aircraft-to-aircraft collision model.

## Dispatch And Timing

All planners share the same greedy dispatcher: at each arrival, sort remaining
tasks by straight-line distance; accept the first task whose planned flight and
a successfully planned return fit remaining range. Stations are tried in
geometric-distance order until a feasible return is found; this is not a global
route optimizer or an exhaustive shortest-return search. If no task fits, move
to another reachable station and recharge. Per-UAV station visit sets prevent
endless charging cycles until that UAV next completes a task. Aircraft that
exhaust these attempts remain blocked; they are not awakened by another UAV's
completion because task availability only decreases in this model.

An event queue advances exact flight arrival times. All four UAVs start at t=0;
their flights overlap. Runtime is fleet makespan through final return to any
station, not the sum of four flight times. Each segment uses existing weather-
adjusted speed with 0.5-distance path sampling. Planning wall time and charging
time are excluded. No discretized waiting time is added.

Every accepted task reserves a feasible return within remaining range. The
experiment requires all tasks completed and all UAVs back at a station for
whole-run success. Terminal returns are included in distance/runtime but do not
count as charging when the aircraft will no longer fly. A charge is counted only
when the dispatcher explicitly requests a refill for further attempts.

## Metrics And Interpretation

- `task_success_rate`: completed tasks / all tasks; single-scene task completion
  fraction, NOT an empirical probability of success over independent runs.
- `success`: binary whole-run success, including final return. Formal success
  rate requires averaging this over many paired scenes/repeats.
- `charging_count`: sum of requested completed refills across all four UAVs.
- `total_distance`: sum of all executed flight lengths, including station travel.
- `total_runtime`: fleet makespan in relative simulated time units, excluding
  planning, charging and service time.
- Failed runs report partial-work distance/runtime, which cannot be ranked as
  better than complete successful runs solely because they are smaller.
- `max_sortie_distance` checks the cumulative between-charge distance limit.
- Planner calls, rejected paths and planning wall seconds are diagnostic only.

The default planning wall budget is now unlimited (`--wall-budget 0`). The
explicit 600-second command above reproduces the earlier capped pilot. A positive
per-algorithm wall budget is checked between calls. A call is not forcibly
interrupted and can overrun. Budget-censored runs retain completion counts, but
distance/runtime are null because committed in-flight legs were not fully
simulated. Output includes `summary.csv`, per-algorithm JSON (flights/events),
per-algorithm PNG and `scenario.npz`. Output directories must not already exist.

Only one seed is an initial functionality/performance sample, not a statistical
comparison. The original optimizers retain their original search budgets and
endpoint-derived randomness. No optimizer budget reduction, silent fallback,
path repair or selective task resampling is applied.
