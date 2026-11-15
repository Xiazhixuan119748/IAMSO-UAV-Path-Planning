"""Helpers for building and loading a fixed benchmark-map dataset."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

DATASET_NAME = "fixed_benchmark_maps"
DATASET_VERSION = "3.0"

DIFFICULTY_SPECS: dict[str, dict[str, Any]] = {
    "simple": {
        "label": "简单",
        "size": 30,
        "count": 30,
        "seed_base": 2026070900,
        "target_obstacle_ratio": 0.15,
        "safety_radius": 2,
    },
    "medium": {
        "label": "中等",
        "size": 60,
        "count": 30,
        "seed_base": 2026071900,
        "target_obstacle_ratio": 0.25,
        "safety_radius": 3,
    },
    "hard": {
        "label": "困难",
        "size": 90,
        "count": 30,
        "seed_base": 2026072900,
        "target_obstacle_ratio": 0.35,
        "safety_radius": 4,
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def experiment_root() -> Path:
    return Path(__file__).resolve().parent


def benchmark_maps_dir() -> Path:
    return experiment_root() / "maps"


def benchmark_results_dir() -> Path:
    return experiment_root() / "results"


def fixed_start_for_size(size: int) -> tuple[int, int]:
    center = size // 2
    return center, center


def fixed_goal_for_size(size: int) -> tuple[int, int]:
    margin = max(3, size // 10)
    row = max(margin, size // 4)
    col = min(size - margin - 1, (size * 3) // 4)
    return row, col


def record_id(difficulty: str, order: int) -> str:
    return f"{difficulty}_{order:03d}"


def record_filename(difficulty: str, order: int) -> str:
    return f"{record_id(difficulty, order)}.npz"


def build_record(difficulty: str, order: int) -> dict[str, Any]:
    spec = DIFFICULTY_SPECS[difficulty]
    size = int(spec["size"])
    start = fixed_start_for_size(size)
    goal = fixed_goal_for_size(size)
    seed = int(spec["seed_base"]) + order
    return {
        "id": record_id(difficulty, order),
        "file": record_filename(difficulty, order),
        "difficulty": difficulty,
        "difficulty_label": spec["label"],
        "map_size": size,
        "seed": seed,
        "start": list(start),
        "goal": list(goal),
        "target_obstacle_ratio": float(spec["target_obstacle_ratio"]),
        "safety_radius": int(spec["safety_radius"]),
    }


def dataset_metadata() -> dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(benchmark_maps_dir().relative_to(project_root())).replace("\\", "/"),
        "generation_rule": {
            "map_sizes": [30, 60, 90],
            "difficulty_counts": {
                key: int(value["count"])
                for key, value in DIFFICULTY_SPECS.items()
            },
            "start_rule": "fixed_center",
            "goal_rule": "fixed_upper_right_quadrant_with_margin",
            "obstacle_mode": "multi_shape",
            "wall_obstacles_enabled": False,
            "connectivity_protected": True,
            "shape_limits": "bounded size, thickness, and aspect ratio",
            "terminal_safety_radius": {
                str(value["size"]): int(value["safety_radius"])
                for value in DIFFICULTY_SPECS.values()
            },
            "target_obstacle_ratio": {
                str(value["size"]): float(value["target_obstacle_ratio"])
                for value in DIFFICULTY_SPECS.values()
            },
        },
        "difficulty_specs": {
            key: {
                "label": value["label"],
                "size": int(value["size"]),
                "count": int(value["count"]),
                "seed_base": int(value["seed_base"]),
                "target_obstacle_ratio": float(value["target_obstacle_ratio"]),
                "safety_radius": int(value["safety_radius"]),
            }
            for key, value in DIFFICULTY_SPECS.items()
        },
    }


def save_map_bundle(
    output_path: Path,
    layers: dict[str, Any],
    passable: np.ndarray,
    record: dict[str, Any],
) -> dict[str, Any]:
    tasks = np.asarray(layers.get("tasks", []), dtype=np.int16)
    np.savez_compressed(
        output_path,
        O=np.asarray(layers["O"]),
        I=np.asarray(layers["I"]),
        W=np.asarray(layers["W"]),
        R=np.asarray(layers["R"]),
        V=np.asarray(layers["V"]),
        speed_map=np.asarray(layers["speed_map"]),
        passable=np.asarray(passable),
        tasks=tasks,
        start=np.asarray(record["start"], dtype=np.int16),
        goal=np.asarray(record["goal"], dtype=np.int16),
        difficulty=np.asarray(record["difficulty"]),
        difficulty_label=np.asarray(record["difficulty_label"]),
        map_size=np.asarray(record["map_size"], dtype=np.int16),
        seed=np.asarray(record["seed"], dtype=np.int64),
        target_obstacle_ratio=np.asarray(record["target_obstacle_ratio"], dtype=float),
        safety_radius=np.asarray(record["safety_radius"], dtype=np.int16),
    )

    bundle_summary = dict(record)
    bundle_summary["blocked_ratio"] = round(float(1.0 - passable.mean()), 6)
    bundle_summary["task_count"] = int(tasks.shape[0])
    if "obstacle_meta" in record:
        bundle_summary["obstacle_meta"] = record["obstacle_meta"]
    bundle_summary["goal_manhattan_distance"] = int(
        abs(record["goal"][0] - record["start"][0]) + abs(record["goal"][1] - record["start"][1])
    )
    bundle_summary["goal_euclidean_distance"] = round(
        float(np.linalg.norm(np.asarray(record["goal"], dtype=float) - np.asarray(record["start"], dtype=float))),
        6,
    )
    return bundle_summary


def write_index(index: dict[str, Any], index_path: Path | None = None) -> Path:
    target = index_path or (benchmark_maps_dir() / "index.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_index(index_path: Path | None = None) -> dict[str, Any]:
    target = index_path or (benchmark_maps_dir() / "index.json")
    return json.loads(target.read_text(encoding="utf-8"))


def load_map_bundle(file_path: str | Path) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    path = Path(file_path)
    with np.load(path, allow_pickle=False) as data:
        layers = {
            "O": data["O"],
            "I": data["I"],
            "W": data["W"],
            "R": data["R"],
            "V": data["V"],
            "tasks": [tuple(item) for item in data["tasks"].tolist()],
            "speed_map": data["speed_map"],
        }
        passable = data["passable"]
        meta = {
            "start": tuple(int(x) for x in data["start"].tolist()),
            "goal": tuple(int(x) for x in data["goal"].tolist()),
            "difficulty": str(data["difficulty"].item()),
            "difficulty_label": str(data["difficulty_label"].item()),
            "map_size": int(data["map_size"].item()),
            "seed": int(data["seed"].item()),
            "file": path.name,
        }
    return layers, passable, meta
