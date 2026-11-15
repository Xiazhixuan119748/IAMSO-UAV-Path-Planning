# Multi-UAV Cooperative Search Prototype

This independent experiment leaves modes 1-6 unchanged. It uses known static
obstacle/weather layers, a known total task count, and private unknown task
positions. Stations and tasks are sampled in the launch station's four-connected
component. The public task list is empty and the public value layer is zero.

## Run

Run from the repository root:

```powershell
# Fast closed-loop validation: 60x60, 3 UAVs, 10 tasks, 2 stations.
.\.venv\Scripts\python.exe -m uav_planner.experiments.multi_uav.run_experiment --planner astar --gif

# Production IAMSO integration smoke test; start small because each leg is costly.
.\.venv\Scripts\python.exe -m uav_planner.experiments.multi_uav.run_experiment --planner iamso --size 15 --tasks 3 --sensor-radius 3 --wall-budget 90

# Full-size IAMSO experiment, with an explicit wall-clock planning budget.
.\.venv\Scripts\python.exe -m uav_planner.experiments.multi_uav.run_experiment --planner iamso --wall-budget 600

.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_multi_uav.py -v
```

Default output: `outputs/multi_uav/<timestamp>_<planner>_seed<seed>/`.
An explicit `--output` must be a new directory; existing runs are not overwritten.
`--seed`, `--uavs`, `--tasks`, `--battery`, `--reserve`, `--sensor-radius`,
`--candidates`, `--max-time`, and `--wall-budget` can be varied independently.

## Model And Decisions

- One simulation clock advances all UAVs at dt=0.5 relative time units. Sensing
  checks swept disks along every traversed segment, including return flights.
- A task is discovered once, reserved for one UAV, serviced, then completed once.
  Idle aircraft compete using normalized actual leg time and energy. The search
  stage ranks observation cells by uncovered area/distance, then checks the best
  two candidates with the selected planner, expanding to at most 12 when none
  is feasible. Reserved destination disks prevent
  some duplicate search. All searchable cells are potential observation nodes.
- Every accepted search/task decision reserves its actual flight energy, task
  service energy, the cheapest successfully planned directed return to a station,
  and the safety margin. Return costs are never copied from the outbound leg.
- Relative energy = 0.35*length + 0.2*time + 0.2*rain exposure + 0.2*headwind
  exposure; service power=0.2, service time=3. Default battery=65, reserve=5.
  These are uncalibrated simulation values, not physical Wh or measured seconds.
- Charging takes 15 time units, then restores a full battery. No station queue,
  communication latency, sensor occlusion, or inter-UAV collision model is used.
- `astar` here means an experiment-local A* with continuous collision checking
  on each edge. It does not change the original production A* or silently repair
  rejected IAMSO paths. IAMSO uses the existing production implementation and
  budget. Invalid/failed paths are logged and never count as feasible.
- Static directed legs, including failures, are cached within a run. Wall-clock
  limits are checked between planner calls; one planner call cannot be interrupted
  and may overrun the limit. Planning wall time is reported separately and does
  not advance simulated flight time.

## Outputs And Metrics

- `summary.json`: configuration, algorithm, termination, task fractions, coverage,
  time, relative energy, charge counts/duration, battery minima and planner cost.
- `events.jsonl`: discoveries, unique assignments/completions, arrivals, recharge,
  planning failures and termination.
- `trajectories.csv`: synchronized position, battery and mode histories.
- `progress.csv`: discovery, completion and coverage over simulation time.
- `scenario.npz`: static layers, private truth, stations, searchable/final coverage
  masks, for evaluation and exact replay reconstruction. Do not feed truth back
  to the scheduler.
- `overview.png`: final trajectories and task progress. Truth is shown only in
  this post-run evaluation figure. `--gif` also exports an approximate 150-frame
  replay that shows tasks only after discovery.

Success requires all N tasks completed. Time/budget/no-feasible-action terminations
retain partial metrics with `finish_time=null`. Repeat search fraction counts
unique cells per UAV per assigned leg (including transit), not raw timestep hits;
this definition avoids timestep-dependent repeated stationary observations.
Simulation events are timestamped at tick boundaries, with timing error up to dt
per transition. Stop at the last task completion, without a post-mission return;
energy includes only travel/service actually executed up to that stop.

## Prototype Boundaries

This is a closed-loop integration baseline, not the final claimed research method.
Committed flights are not interrupted when a task is discovered. Thus the current
executor choice compares available idle aircraft; it does not yet compare every
in-flight discoverer/alternative's remaining mission completion time. The score
uses actual local leg time/energy, not a globally optimized makespan prediction.
Add interruption-aware task dispatch and static-partition/nearest-node baselines
before a formal strategy comparison. Greedy top-k selection can stall even when
another plan exists; `no_feasible_action` is not a proof of global infeasibility.
No weather hard-safety threshold is invented: obstacle checks are hard, weather
is a cost/energy model until explicit mission thresholds are supplied.

Run paired scenes across seeds with separate planning budgets and multiple
independent optimizer repetitions before drawing performance conclusions. The
production IAMSO currently derives its random seed from endpoints; changing only
the scene seed is not an independent optimizer-repeat control.

## Initial Verification (2026-10-22)

Seed 42, strict A*, 60x60, 3 UAVs, 10 tasks: all tasks completed at simulation
time 240.5, 8 completed charges, minimum battery 5.1922 (reserve 5), no stranded
UAVs. Total relative energy 475.2927, distance 972.6245, coverage 95.36%.
517 planning calls took 20.56 wall seconds; total simulation wall time 21.53 s
(excluding figure/GIF export). Results: `outputs/multi_uav/validated_astar/`.

Production IAMSO smoke test, seed 42, 8x8, 1 UAV, 1 task, radius 1: task completed
at simulation time 7.5, no recharge required, 15 planning calls, 152.08 planning
wall seconds. Results: `outputs/multi_uav/validated_iamso/`. The 15x15, 3-UAV run
hit its 90-second planning budget before flight began (one call overran the
limit; recorded wall time 99.78 s). No claim of full-size IAMSO success is made.

12 focused tests passed. The broader 53-test suite before adding the final three
focused cases had 2 failures in existing mode-6 forwarding assertions: the current
implementation adds `dataset_dir=None`, while those tests expect no such keyword.
Those pre-existing modified modules were left unchanged.
