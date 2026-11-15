"""Generate the 100-map, 60x60, 20% reachable no-task benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from uav_planner.experiments.no_task_corner.dataset import (
    COST_PROFILE,
    EVALUATION_PROFILE,
    GOAL_RULE,
    START_RULE,
    VALUE_MODE,
    project_root,
    save_map_bundle,
)
from uav_planner.experiments.no_task_corner.map_generator import generate_no_task_corner_map


DATASET_NAME = "medium60_density20_reachable_100"
DATASET_VERSION = "1.0"
MAP_SIZE = 60
MAP_COUNT = 100
TARGET_OBSTACLE_RATIO = 0.20
SAFETY_RADIUS = 3
SEED_BASE = 2026077100


def dataset_dir() -> Path:
    return Path(__file__).resolve().parent / "datasets" / DATASET_NAME


def generate_dataset(*, force: bool = False, max_retries: int = 20) -> dict:
    max_retries = max(1, max_retries)
    root = dataset_dir()
    root.mkdir(parents=True, exist_ok=True)
    existing = list(root.glob("*.npz"))
    index_path = root / "index.json"
    if (existing or index_path.exists()) and not force:
        raise FileExistsError(f"{root} already contains a dataset; use --force to rebuild it")
    if force:
        for path in existing:
            path.unlink()
        index_path.unlink(missing_ok=True)

    start = (0, 0)
    goal = (MAP_SIZE - 1, MAP_SIZE - 1)
    records = []
    for order in range(1, MAP_COUNT + 1):
        for retry in range(max_retries):
            seed = SEED_BASE + order + retry * 100000
            try:
                layers, passable, obstacle_meta = generate_no_task_corner_map(
                    start=start,
                    goal=goal,
                    seed=seed,
                    size=MAP_SIZE,
                    target_ratio=TARGET_OBSTACLE_RATIO,
                    safety_radius=SAFETY_RADIUS,
                )
                break
            except RuntimeError:
                if retry == max_retries - 1:
                    raise

        record = {
            "id": f"medium20_{order:03d}",
            "file": f"medium20_{order:03d}.npz",
            "difficulty": "medium20",
            "difficulty_label": "中等20%障碍",
            "map_size": MAP_SIZE,
            "seed": seed,
            "start": list(start),
            "goal": list(goal),
            "target_obstacle_ratio": TARGET_OBSTACLE_RATIO,
            "safety_radius": SAFETY_RADIUS,
            "value_mode": VALUE_MODE,
            "start_rule": START_RULE,
            "goal_rule": GOAL_RULE,
            "cost_profile": COST_PROFILE,
            "evaluation_profile": EVALUATION_PROFILE,
            "obstacle_meta": obstacle_meta,
            "retry_count": retry,
        }
        records.append(save_map_bundle(root / record["file"], layers, passable, record))
        print(f"\r生成新测试地图: {order}/{MAP_COUNT}", end="", flush=True)

    index = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(root.relative_to(project_root())).replace("\\", "/"),
        "value_mode": VALUE_MODE,
        "start_rule": START_RULE,
        "goal_rule": GOAL_RULE,
        "cost_profile": COST_PROFILE,
        "evaluation_profile": EVALUATION_PROFILE,
        "generation_rule": {
            "map_sizes": [MAP_SIZE],
            "difficulty_counts": {"medium20": MAP_COUNT},
            "start_rule": START_RULE,
            "goal_rule": GOAL_RULE,
            "obstacle_mode": "multi_shape",
            "wall_obstacles_enabled": False,
            "connectivity_protected": True,
            "connectivity_rule": "collision_safe_8_neighbor",
            "value_mode": VALUE_MODE,
            "terminal_safety_radius": {str(MAP_SIZE): SAFETY_RADIUS},
            "target_obstacle_ratio": {str(MAP_SIZE): TARGET_OBSTACLE_RATIO},
        },
        "difficulty_specs": {
            "medium20": {
                "label": "中等20%障碍",
                "size": MAP_SIZE,
                "count": MAP_COUNT,
                "seed_base": SEED_BASE,
                "target_obstacle_ratio": TARGET_OBSTACLE_RATIO,
                "safety_radius": SAFETY_RADIUS,
            }
        },
        "map_count": MAP_COUNT,
        "maps": records,
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-retries", type=int, default=20)
    args = parser.parse_args()
    index = generate_dataset(force=args.force, max_retries=args.max_retries)
    print(f"已生成 {index['map_count']} 张地图: {dataset_dir()}")


if __name__ == "__main__":
    main()
