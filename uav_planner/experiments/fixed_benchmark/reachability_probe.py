"""Small-sample reachability probe for multi-shape, no-wall obstacle maps."""

from __future__ import annotations

import csv
import json
import math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from uav_planner.experiments.fixed_benchmark.dataset import (
    benchmark_results_dir,
    fixed_goal_for_size,
    fixed_start_for_size,
)
from uav_planner.algorithms.common import iter_moves

PROBE_CASES = [
    {
        "name": "simple",
        "label": "简单",
        "size": 30,
        "target_ratio": 0.30,
        "samples": 100,
        "seed_base": 2026071000,
    },
    {
        "name": "medium",
        "label": "中等",
        "size": 60,
        "target_ratio": 0.45,
        "samples": 100,
        "seed_base": 2026072000,
    },
    {
        "name": "hard",
        "label": "困难",
        "size": 90,
        "target_ratio": 0.60,
        "samples": 100,
        "seed_base": 2026073000,
    },
]


def _clip_window(size: int, row_min: int, row_max: int, col_min: int, col_max: int) -> tuple[int, int, int, int]:
    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(size, row_max)
    col_max = min(size, col_max)
    return row_min, row_max, col_min, col_max


def _patch_indices(row_min: int, row_max: int, col_min: int, col_max: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(row_min, row_max, dtype=float)[:, None]
    cols = np.arange(col_min, col_max, dtype=float)[None, :]
    return rows, cols


def _rand_span(rng: np.random.Generator, size: int, low_ratio: float, high_ratio: float, minimum: int = 2) -> int:
    low = max(minimum, int(round(size * low_ratio)))
    high = max(low, int(round(size * high_ratio)))
    return int(rng.integers(low, high + 1))


def _stamp_rectangle(rng: np.random.Generator, obstacles: np.ndarray) -> None:
    size = obstacles.shape[0]
    height = _rand_span(rng, size, 0.06, 0.24)
    width = _rand_span(rng, size, 0.06, 0.24)
    row = int(rng.integers(0, max(1, size - height + 1)))
    col = int(rng.integers(0, max(1, size - width + 1)))
    obstacles[row:row + height, col:col + width] = 1


def _stamp_circle(rng: np.random.Generator, obstacles: np.ndarray) -> None:
    size = obstacles.shape[0]
    radius = _rand_span(rng, size, 0.05, 0.12, minimum=2)
    center_r = int(rng.integers(0, size))
    center_c = int(rng.integers(0, size))
    row_min, row_max, col_min, col_max = _clip_window(
        size,
        center_r - radius - 1,
        center_r + radius + 2,
        center_c - radius - 1,
        center_c + radius + 2,
    )
    rows, cols = _patch_indices(row_min, row_max, col_min, col_max)
    mask = (rows - center_r) ** 2 + (cols - center_c) ** 2 <= radius ** 2
    obstacles[row_min:row_max, col_min:col_max][mask] = 1


def _stamp_ellipse(rng: np.random.Generator, obstacles: np.ndarray) -> None:
    size = obstacles.shape[0]
    axis_r = _rand_span(rng, size, 0.05, 0.14, minimum=2)
    axis_c = _rand_span(rng, size, 0.05, 0.18, minimum=3)
    center_r = int(rng.integers(0, size))
    center_c = int(rng.integers(0, size))
    angle = float(rng.uniform(0.0, math.pi))
    bound = max(axis_r, axis_c) + 2
    row_min, row_max, col_min, col_max = _clip_window(
        size,
        center_r - bound,
        center_r + bound + 1,
        center_c - bound,
        center_c + bound + 1,
    )
    rows, cols = _patch_indices(row_min, row_max, col_min, col_max)
    x = cols - center_c
    y = rows - center_r
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    xr = x * cos_a + y * sin_a
    yr = -x * sin_a + y * cos_a
    mask = (xr / max(axis_c, 1e-9)) ** 2 + (yr / max(axis_r, 1e-9)) ** 2 <= 1.0
    obstacles[row_min:row_max, col_min:col_max][mask] = 1


def _stamp_diamond(rng: np.random.Generator, obstacles: np.ndarray) -> None:
    size = obstacles.shape[0]
    radius_r = _rand_span(rng, size, 0.05, 0.12, minimum=2)
    radius_c = _rand_span(rng, size, 0.05, 0.16, minimum=2)
    center_r = int(rng.integers(0, size))
    center_c = int(rng.integers(0, size))
    row_min, row_max, col_min, col_max = _clip_window(
        size,
        center_r - radius_r - 1,
        center_r + radius_r + 2,
        center_c - radius_c - 1,
        center_c + radius_c + 2,
    )
    rows, cols = _patch_indices(row_min, row_max, col_min, col_max)
    mask = np.abs(rows - center_r) / max(radius_r, 1e-9) + np.abs(cols - center_c) / max(radius_c, 1e-9) <= 1.0
    obstacles[row_min:row_max, col_min:col_max][mask] = 1


def _stamp_l_shape(rng: np.random.Generator, obstacles: np.ndarray) -> None:
    size = obstacles.shape[0]
    height = _rand_span(rng, size, 0.10, 0.28, minimum=4)
    width = _rand_span(rng, size, 0.10, 0.28, minimum=4)
    thickness = min(height, width, _rand_span(rng, size, 0.03, 0.08, minimum=2))
    row = int(rng.integers(0, max(1, size - height + 1)))
    col = int(rng.integers(0, max(1, size - width + 1)))
    orientation = int(rng.integers(0, 4))

    if orientation == 0:
        obstacles[row:row + height, col:col + thickness] = 1
        obstacles[row + height - thickness:row + height, col:col + width] = 1
    elif orientation == 1:
        obstacles[row:row + height, col + width - thickness:col + width] = 1
        obstacles[row + height - thickness:row + height, col:col + width] = 1
    elif orientation == 2:
        obstacles[row:row + height, col:col + thickness] = 1
        obstacles[row:row + thickness, col:col + width] = 1
    else:
        obstacles[row:row + height, col + width - thickness:col + width] = 1
        obstacles[row:row + thickness, col:col + width] = 1


def _stamp_t_shape(rng: np.random.Generator, obstacles: np.ndarray) -> None:
    size = obstacles.shape[0]
    height = _rand_span(rng, size, 0.10, 0.26, minimum=5)
    width = _rand_span(rng, size, 0.12, 0.30, minimum=5)
    thickness = min(height, width, _rand_span(rng, size, 0.03, 0.08, minimum=2))
    row = int(rng.integers(0, max(1, size - height + 1)))
    col = int(rng.integers(0, max(1, size - width + 1)))
    orientation = int(rng.integers(0, 4))

    if orientation == 0:
        stem_col = col + (width - thickness) // 2
        obstacles[row:row + thickness, col:col + width] = 1
        obstacles[row:row + height, stem_col:stem_col + thickness] = 1
    elif orientation == 1:
        stem_row = row + (height - thickness) // 2
        obstacles[row:row + height, col + width - thickness:col + width] = 1
        obstacles[stem_row:stem_row + thickness, col:col + width] = 1
    elif orientation == 2:
        stem_col = col + (width - thickness) // 2
        obstacles[row + height - thickness:row + height, col:col + width] = 1
        obstacles[row:row + height, stem_col:stem_col + thickness] = 1
    else:
        stem_row = row + (height - thickness) // 2
        obstacles[row:row + height, col:col + thickness] = 1
        obstacles[stem_row:stem_row + thickness, col:col + width] = 1


SHAPE_STAMPERS = [
    ("rectangle", _stamp_rectangle),
    ("circle", _stamp_circle),
    ("ellipse", _stamp_ellipse),
    ("diamond", _stamp_diamond),
    ("l_shape", _stamp_l_shape),
    ("t_shape", _stamp_t_shape),
]


def generate_multishape_obstacles(
    size: int,
    target_ratio: float,
    seed: int,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[np.ndarray, dict[str, int]]:
    """Generate a mixed-shape obstacle map without wall obstacles."""
    rng = np.random.default_rng(seed)
    obstacles = np.zeros((size, size), dtype=np.uint8)
    shape_counts = {name: 0 for name, _ in SHAPE_STAMPERS}

    max_attempts = max(120, int(size * (1.5 + target_ratio * 12.0)))
    while float(obstacles.mean()) < target_ratio and sum(shape_counts.values()) < max_attempts:
        name, stamper = SHAPE_STAMPERS[int(rng.integers(0, len(SHAPE_STAMPERS)))]
        stamper(rng, obstacles)
        obstacles[start] = 0
        obstacles[goal] = 0
        shape_counts[name] += 1

    obstacles[start] = 0
    obstacles[goal] = 0
    return obstacles, shape_counts


def analyze_reachability(passable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> dict[str, Any]:
    """Reachability under the same 8-neighbor movement model used by grid planners."""
    size = int(passable.shape[0])
    free_cells = int(passable.sum())
    if free_cells <= 0 or not passable[start] or not passable[goal]:
        return {
            "reachable": False,
            "free_cells": free_cells,
            "component_cells": 0,
            "component_ratio": 0.0,
        }

    visited = np.zeros_like(passable, dtype=bool)
    queue: deque[tuple[int, int]] = deque([start])
    visited[start] = True
    component_cells = 0
    reachable = False

    while queue:
        row, col = queue.popleft()
        component_cells += 1
        if (row, col) == goal:
            reachable = True

        for dr, dc, _length in iter_moves():
            nr, nc = row + dr, col + dc
            if 0 <= nr < size and 0 <= nc < size and passable[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

    return {
        "reachable": reachable,
        "free_cells": free_cells,
        "component_cells": component_cells,
        "component_ratio": (component_cells / free_cells) if free_cells else 0.0,
    }


def _summarize_rows(case: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    reachable_count = sum(1 for row in rows if row["reachable"])
    return {
        "name": case["name"],
        "label": case["label"],
        "size": case["size"],
        "target_ratio": case["target_ratio"],
        "samples": len(rows),
        "reachable_count": reachable_count,
        "unreachable_count": len(rows) - reachable_count,
        "reachable_rate": reachable_count / len(rows) if rows else 0.0,
        "avg_blocked_ratio": float(np.mean([row["blocked_ratio"] for row in rows])) if rows else 0.0,
        "avg_component_ratio": float(np.mean([row["component_ratio"] for row in rows])) if rows else 0.0,
        "avg_shape_count": float(np.mean([row["shape_count"] for row in rows])) if rows else 0.0,
    }


def run_reachability_probe(cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cases = cases or PROBE_CASES
    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    total_maps = sum(int(case["samples"]) for case in cases)
    completed = 0

    for case in cases:
        rows: list[dict[str, Any]] = []
        start = fixed_start_for_size(int(case["size"]))
        goal = fixed_goal_for_size(int(case["size"]))
        print(
            f"\n评估 {case['label']}：尺寸 {case['size']}x{case['size']}，"
            f"目标障碍占比 {case['target_ratio']:.0%}，样本数 {case['samples']}"
        )
        for index in range(1, int(case["samples"]) + 1):
            seed = int(case["seed_base"]) + index
            obstacles, shape_counts = generate_multishape_obstacles(
                size=int(case["size"]),
                target_ratio=float(case["target_ratio"]),
                seed=seed,
                start=start,
                goal=goal,
            )
            passable = 1 - obstacles
            reach = analyze_reachability(passable, start, goal)
            row = {
                "case_name": case["name"],
                "case_label": case["label"],
                "map_size": int(case["size"]),
                "target_ratio": float(case["target_ratio"]),
                "sample_index": index,
                "seed": seed,
                "start": start,
                "goal": goal,
                "blocked_ratio": float(1.0 - passable.mean()),
                "free_cells": reach["free_cells"],
                "component_cells": reach["component_cells"],
                "component_ratio": float(reach["component_ratio"]),
                "reachable": bool(reach["reachable"]),
                "shape_count": int(sum(shape_counts.values())),
                "shape_mix": dict(shape_counts),
            }
            rows.append(row)
            details.append(row)
            completed += 1
            print(
                f"\r  进度 {index:>3}/{case['samples']} | "
                f"总进度 {completed:>3}/{total_maps} | "
                f"理论可达={'是' if row['reachable'] else '否'} | "
                f"实际占比={row['blocked_ratio']:.2%}",
                end="",
                flush=True,
            )
        print()
        summaries.append(_summarize_rows(case, rows))

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "assumption": {
            "movement_model": "与当前栅格算法一致的8邻域可通行判定",
            "start": "地图中心",
            "goal": "右上象限固定终点",
            "map_rule": "多形状障碍 + 去墙状障碍，仅评估理论连通率",
        },
        "summaries": summaries,
        "details": details,
    }


def _result_stem() -> str:
    return f"multishape_reachability_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法序列化的对象: {type(value)!r}")


def save_reachability_report(report: dict[str, Any]) -> dict[str, str]:
    output_dir = benchmark_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _result_stem()

    json_path = output_dir / f"{stem}.json"
    summary_csv_path = output_dir / f"{stem}_summary.csv"
    details_csv_path = output_dir / f"{stem}_details.csv"
    markdown_path = output_dir / f"{stem}.md"

    json_path.write_text(
        json.dumps(json.loads(json.dumps(report, ensure_ascii=False, default=_json_default)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "label",
                "size",
                "target_ratio",
                "samples",
                "reachable_count",
                "unreachable_count",
                "reachable_rate",
                "avg_blocked_ratio",
                "avg_component_ratio",
                "avg_shape_count",
            ],
        )
        writer.writeheader()
        writer.writerows(report["summaries"])

    with details_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_name",
                "case_label",
                "map_size",
                "target_ratio",
                "sample_index",
                "seed",
                "start",
                "goal",
                "blocked_ratio",
                "free_cells",
                "component_cells",
                "component_ratio",
                "reachable",
                "shape_count",
                "shape_mix",
            ],
        )
        writer.writeheader()
        writer.writerows(report["details"])

    lines = [
        "# 多形状无墙地图小样本理论可达率评估",
        "",
        "- 说明：本评估不使用现有主地图生成器，只在固定基准实验目录下做独立试验。",
        "- 地图规则：多形状障碍 + 去墙状障碍。",
        "- 可达判定：按当前项目栅格算法一致的 8 邻域连通性判断。",
        "- 起点：地图中心。",
        "- 终点：右上象限固定终点。",
        "",
        "| 场景 | 尺寸 | 目标障碍占比 | 样本数 | 理论可达张数 | 天然无解张数 | 理论可达率 | 平均实际障碍占比 | 起点连通分量平均占比 | 平均形状数量 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summaries"]:
        lines.append(
            f"| {row['label']} | {row['size']}x{row['size']} | {row['target_ratio']:.0%} | {row['samples']} | "
            f"{row['reachable_count']} | {row['unreachable_count']} | {row['reachable_rate']:.2%} | "
            f"{row['avg_blocked_ratio']:.2%} | {row['avg_component_ratio']:.2%} | {row['avg_shape_count']:.2f} |"
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "json": str(json_path),
        "summary_csv": str(summary_csv_path),
        "details_csv": str(details_csv_path),
        "markdown": str(markdown_path),
    }
