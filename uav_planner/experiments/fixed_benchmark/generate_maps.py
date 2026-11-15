"""Generate a fixed benchmark-map dataset without changing planner logic."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from uav_planner.experiments.fixed_benchmark.dataset import (  # noqa: E402
    DIFFICULTY_SPECS,
    benchmark_maps_dir,
    benchmark_results_dir,
    build_record,
    dataset_metadata,
    save_map_bundle,
    write_index,
)
from uav_planner.experiments.fixed_benchmark.map_generator import generate_benchmark_map  # noqa: E402


def _prepare_directories(dataset_dir: Path, force: bool) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    benchmark_results_dir().mkdir(parents=True, exist_ok=True)
    existing_maps = list(dataset_dir.glob("*.npz"))
    index_path = dataset_dir / "index.json"
    if (existing_maps or index_path.exists()) and not force:
        raise FileExistsError(
            f"{dataset_dir} 已存在固定测试集文件。如需重建，请加 --force。"
        )
    if force:
        for path in existing_maps:
            path.unlink()
        if index_path.exists():
            index_path.unlink()


def generate_dataset(force: bool = False) -> dict:
    dataset_dir = benchmark_maps_dir()
    _prepare_directories(dataset_dir, force=force)

    index = dataset_metadata()
    index["maps"] = []

    total = sum(int(spec["count"]) for spec in DIFFICULTY_SPECS.values())
    current = 0
    for difficulty, spec in DIFFICULTY_SPECS.items():
        count = int(spec["count"])
        for order in range(1, count + 1):
            record = build_record(difficulty, order)
            start = tuple(record["start"])
            goal = tuple(record["goal"])
            layers, passable, obstacle_meta = generate_benchmark_map(
                start=start,
                goal=goal,
                seed=record["seed"],
                size=record["map_size"],
                target_ratio=float(spec["target_obstacle_ratio"]),
                safety_radius=int(spec["safety_radius"]),
            )
            record["obstacle_meta"] = obstacle_meta
            summary = save_map_bundle(dataset_dir / record["file"], layers, passable, record)
            index["maps"].append(summary)
            current += 1
            print(
                f"\r生成固定测试地图: {current}/{total} "
                f"({record['difficulty_label']} {order}/{count})",
                end="",
                flush=True,
            )

    index["map_count"] = len(index["maps"])
    print()
    write_index(index, dataset_dir / "index.json")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="生成固定测试地图集")
    parser.add_argument(
        "--force",
        action="store_true",
        help="若已存在 benchmark_maps，则先清理后重建。",
    )
    args = parser.parse_args()
    index = generate_dataset(force=args.force)
    print(
        f"固定测试地图集已生成: {index['map_count']} 张地图，目录: {benchmark_maps_dir()}"
    )


if __name__ == "__main__":
    main()
