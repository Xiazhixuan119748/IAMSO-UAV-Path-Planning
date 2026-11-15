# IAMSO UAV Path Planning

An experimental Python project for comparing Dijkstra, A*, DFS, RRT, MSO, AO, and IAMSO on two-dimensional UAV global path-planning problems.

This repository is a snapshot of the local project code and map assets. It excludes private presentation and patent documents, local environments, and generated experiment results. The included datasets have not been verified against the exact final experimental configuration reported in the manuscript; this repository alone is not a complete archive of the reported numerical results.

## Quick Start

Python 3.11 or newer is required.

```powershell
uv sync
.\.venv\Scripts\python.exe .\main.py
```

The runtime dependencies declared in `pyproject.toml` are NumPy, SciPy, and Matplotlib.

## Runtime Modes

| Mode | Purpose | Main output |
|---:|---|---|
| 1 | Run one algorithm | Path image, layer image, optional GIF |
| 2 | Compare all algorithms with one cost function | Comparison table and plots |
| 3 | Compare all cost functions with one algorithm | Comparison table and plots |
| 4 | Run with custom weights | Path and metrics |
| 5 | Batch comparison on generated maps | `batch_results/` |
| 6 | Fixed no-task-value benchmark | Markdown under `无价值固定地图批量生成结果/` |

## Repository Layout

```text
main.py
uav_planner/
  algorithms/                 Planner adapters
  metaheuristic/              MSO, AO, and IAMSO implementations
  batch_comparison/           Mode 5 batch runner
  mode_runners/               Interactive mode dispatch
  experiments/
    fixed_benchmark/          Fixed task-value benchmark and regressions
    no_task_corner/           Mode 6 fixed benchmark
tests/                        Automated regression tests
docs/algorithm/               IAMSO implementation notes
outputs/                      Rendered images and animations
batch_results/                Mode 5 reports
无价值固定地图批量生成结果/     Mode 6 Markdown reports
```

## Fixed Benchmark

The task-value benchmark is stored under `uav_planner/experiments/fixed_benchmark/`. Dataset version `fixed_benchmark_maps v3.0` contains 30 maps for each difficulty:

| Difficulty | Size | Target obstacle ratio |
|---|---:|---:|
| simple | 30x30 | 15% |
| medium | 60x60 | 25% |
| hard | 90x90 | 35% |

```powershell
.\.venv\Scripts\python.exe -m uav_planner.experiments.fixed_benchmark.run_benchmark
.\.venv\Scripts\python.exe -m uav_planner.experiments.fixed_benchmark.run_benchmark --difficulty simple
.\.venv\Scripts\python.exe -m uav_planner.experiments.fixed_benchmark.generate_maps --force
```

The fixed benchmark is separate from the active three-difficulty Mode 6 dataset.

## Mode 6 Benchmark

Mode 6 uses the maps under `uav_planner/experiments/no_task_corner/maps/` with fixed corner-to-corner endpoints and no task-value points. It currently writes Markdown only.

```powershell
.\.venv\Scripts\python.exe -m uav_planner.experiments.no_task_corner.run_benchmark --difficulty simple
.\.venv\Scripts\python.exe -m uav_planner.experiments.no_task_corner.generate_maps --force
```

## Tests

Publication check: 19 focused tests were run using the existing local Python environment. Seventeen passed. Two tests in `tests/test_no_task_corner_mode.py` failed because their expected mocked calls omit the current `dataset_dir=None` keyword argument. The source code and tests are preserved without algorithm changes. The complete test suite and the full manuscript experiments have not been rerun.

```powershell
$env:MPLBACKEND='Agg'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
```

## Output Directories

- `outputs/`: rendered paths, map layers, and animations from modes 1-4.
- `batch_results/`: Mode 5 batch reports.
- `无价值固定地图批量生成结果/`: Mode 6 Markdown reports.
- `uav_planner/experiments/fixed_benchmark/results/`: new benchmark output and versioned history under `archive/`.

Generated runtime output directories are ignored by Git. Benchmark maps remain versioned experiment assets. Historical result files are not included in this publication snapshot.
