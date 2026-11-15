"""Run all registered algorithms on the no-task corner benchmark dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from uav_planner.experiments.no_task_corner.runner import (
    evaluate_dataset,
    print_comparison_table,
    save_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行左上到右下、无任务价值固定测试集全算法基线")
    parser.add_argument(
        "--difficulty",
        nargs="*",
        choices=["simple", "medium", "hard"],
        help="可选：只跑指定难度；默认跑全部 no_task_corner 地图。",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="可选：运行包含 index.json 的独立 no-task 数据集目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_dataset(difficulties=args.difficulty, dataset_dir=args.dataset_dir)
    print_comparison_table(report)
    paths = save_report(report)

    print("\nno-task corner 基线运行完成，结果文件如下：")
    for label, file_path in paths.items():
        print(f"- {label}: {file_path}")


if __name__ == "__main__":
    main()
