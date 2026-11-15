"""Dataset helpers for the no-task corner-to-corner benchmark."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DATASET_NAME = "no_task_corner"
DATASET_VERSION = "1.2"
VALUE_MODE = "no_task"
START_RULE = "fixed_upper_left"
GOAL_RULE = "fixed_lower_right"
COST_PROFILE = "no_task_model_cost"
EVALUATION_PROFILE = "no_task"

DIFFICULTY_SPECS: dict[str, dict[str, Any]] = {
    "simple": {
        "label": "简单",
        "size": 30,
        "count": 100,
        "seed_base": 2026074100,
        "target_obstacle_ratio": 0.15,
        "safety_radius": 2,
    },
    "medium": {
        "label": "中等",
        "size": 60,
        "count": 100,
        "seed_base": 2026075100,
        "target_obstacle_ratio": 0.15,
        "safety_radius": 3,
    },
    "hard": {
        "label": "困难",
        "size": 90,
        "count": 100,
        "seed_base": 2026076100,
        "target_obstacle_ratio": 0.20,
        "safety_radius": 4,
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def experiment_root() -> Path:
    return Path(__file__).resolve().parent


def no_task_corner_maps_dir() -> Path:
    return experiment_root() / "maps"


def no_task_corner_results_dir() -> Path:
    return project_root() / "无价值固定地图批量生成结果"


def fixed_start_for_size(size: int) -> tuple[int, int]:
    return 0, 0


def fixed_goal_for_size(size: int) -> tuple[int, int]:
    return size - 1, size - 1


def record_id(difficulty: str, order: int) -> str:
    return f"{difficulty}_{order:03d}"


def record_filename(difficulty: str, order: int) -> str:
    return f"{record_id(difficulty, order)}.npz"


def build_record(difficulty: str, order: int, seed: int | None = None) -> dict[str, Any]:
    spec = DIFFICULTY_SPECS[difficulty]
    size = int(spec["size"])
    start = fixed_start_for_size(size)
    goal = fixed_goal_for_size(size)
    actual_seed = int(spec["seed_base"]) + order if seed is None else int(seed)
    return {
        "id": record_id(difficulty, order),
        "file": record_filename(difficulty, order),
        "difficulty": difficulty,
        "difficulty_label": spec["label"],
        "map_size": size,
        "seed": actual_seed,
        "start": list(start),
        "goal": list(goal),
        "target_obstacle_ratio": float(spec["target_obstacle_ratio"]),
        "safety_radius": int(spec["safety_radius"]),
        "value_mode": VALUE_MODE,
        "start_rule": START_RULE,
        "goal_rule": GOAL_RULE,
        "cost_profile": COST_PROFILE,
        "evaluation_profile": EVALUATION_PROFILE,
    }


def dataset_metadata(count_per_difficulty: int | None = None) -> dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(no_task_corner_maps_dir().relative_to(project_root())).replace("\\", "/"),
        "value_mode": VALUE_MODE,
        "start_rule": START_RULE,
        "goal_rule": GOAL_RULE,
        "cost_profile": COST_PROFILE,
        "evaluation_profile": EVALUATION_PROFILE,
        "generation_rule": {
            "map_sizes": [30, 60, 90],
            "difficulty_counts": {
                key: int(count_per_difficulty or value["count"])
                for key, value in DIFFICULTY_SPECS.items()
            },
            "start_rule": START_RULE,
            "goal_rule": GOAL_RULE,
            "obstacle_mode": "multi_shape",
            "wall_obstacles_enabled": False,
            "connectivity_protected": True,
            "connectivity_rule": "collision_safe_8_neighbor",
            "value_mode": VALUE_MODE,
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
                "count": int(count_per_difficulty or value["count"]),
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
    tasks = np.asarray(layers.get("tasks", []), dtype=np.int16).reshape(-1, 2)
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
        value_mode=np.asarray(record["value_mode"]),
        start_rule=np.asarray(record["start_rule"]),
        goal_rule=np.asarray(record["goal_rule"]),
        cost_profile=np.asarray(record["cost_profile"]),
        evaluation_profile=np.asarray(record["evaluation_profile"]),
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
    target = index_path or (no_task_corner_maps_dir() / "index.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_index(index_path: Path | None = None) -> dict[str, Any]:
    target = index_path or (no_task_corner_maps_dir() / "index.json")
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
            "value_mode": str(data["value_mode"].item()),
            "start_rule": str(data["start_rule"].item()),
            "goal_rule": str(data["goal_rule"].item()),
            "cost_profile": str(data["cost_profile"].item()),
            "evaluation_profile": str(data["evaluation_profile"].item()),
            "file": path.name,
        }
    return layers, passable, meta
