"""Generate the no-task corner-to-corner benchmark dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from uav_planner.algorithms.common import iter_moves
from uav_planner.experiments.no_task_corner.dataset import (
    DIFFICULTY_SPECS,
    build_record,
    dataset_metadata,
    no_task_corner_maps_dir,
    save_map_bundle,
    write_index,
)
from uav_planner.experiments.no_task_corner.map_generator import (
    generate_no_task_corner_map,
    is_strictly_reachable,
)
from uav_planner.obstacle_generation import is_connected


def _prepare_directories(dataset_dir: Path, force: bool) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    existing_maps = list(dataset_dir.glob("*.npz"))
    index_path = dataset_dir / "index.json"
    if (existing_maps or index_path.exists()) and not force:
        raise FileExistsError(
            f"{dataset_dir} already contains no-task corner maps. Rebuild with --force."
        )
    if force:
        for path in existing_maps:
            path.unlink()
        if index_path.exists():
            index_path.unlink()


def _generate_one_record(difficulty: str, order: int, max_retries: int) -> tuple[dict, dict, object]:
    spec = DIFFICULTY_SPECS[difficulty]
    base_seed = int(spec["seed_base"]) + order
    last_error: Exception | None = None

    for retry in range(max(1, max_retries)):
        seed = base_seed + retry * 100000
        record = build_record(difficulty, order, seed=seed)
        try:
            layers, passable, obstacle_meta = generate_no_task_corner_map(
                start=tuple(record["start"]),
                goal=tuple(record["goal"]),
                seed=record["seed"],
                size=record["map_size"],
                target_ratio=float(spec["target_obstacle_ratio"]),
                safety_radius=int(spec["safety_radius"]),
            )
            if layers.get("tasks"):
                raise RuntimeError("no-task map unexpectedly contains tasks")
            if not is_connected(layers["O"], tuple(record["start"]), tuple(record["goal"]), tuple(iter_moves())):
                raise RuntimeError("no connected route after map generation")
            if not is_strictly_reachable(
                passable, tuple(record["start"]), tuple(record["goal"])
            ):
                raise RuntimeError("no strictly reachable route after map generation")
            record["obstacle_meta"] = obstacle_meta
            record["retry_count"] = retry
            return record, layers, passable
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"failed to generate reachable no-task map for {difficulty}_{order:03d}"
    ) from last_error


def generate_dataset(
    *,
    force: bool = False,
    count_per_difficulty: int | None = None,
    max_retries: int = 20,
) -> dict:
    dataset_dir = no_task_corner_maps_dir()
    _prepare_directories(dataset_dir, force=force)

    index = dataset_metadata(count_per_difficulty=count_per_difficulty)
    index["maps"] = []

    total = sum(int(count_per_difficulty or spec["count"]) for spec in DIFFICULTY_SPECS.values())
    current = 0
    for difficulty, spec in DIFFICULTY_SPECS.items():
        count = int(count_per_difficulty or spec["count"])
        for order in range(1, count + 1):
            record, layers, passable = _generate_one_record(difficulty, order, max_retries=max_retries)
            summary = save_map_bundle(dataset_dir / record["file"], layers, passable, record)
            index["maps"].append(summary)
            current += 1
            print(
                f"\r生成 no-task corner 地图: {current}/{total} "
                f"({record['difficulty_label']} {order}/{count})",
                end="",
                flush=True,
            )

    index["map_count"] = len(index["maps"])
    print()
    write_index(index, dataset_dir / "index.json")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成左上到右下、无任务价值的固定测试地图集")
    parser.add_argument(
        "--force",
        action="store_true",
        help="如果 no_task_corner_maps 已存在，则先清理后重建。",
    )
    parser.add_argument(
        "--count-per-difficulty",
        type=int,
        default=None,
        help="每种难度生成多少张图；默认使用完整规格 100 张。",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=20,
        help="单张地图连通性失败时最多换 seed 重试次数。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = generate_dataset(
        force=bool(args.force),
        count_per_difficulty=args.count_per_difficulty,
        max_retries=int(args.max_retries),
    )
    print(
        f"no-task corner 地图集已生成: {index['map_count']} 张地图，目录: {no_task_corner_maps_dir()}"
    )


if __name__ == "__main__":
    main()
