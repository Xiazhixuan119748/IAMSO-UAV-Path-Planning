"""Run all registered algorithms on the fixed benchmark map dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from uav_planner.experiments.fixed_benchmark.runner import evaluate_dataset, save_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行固定测试集全算法基线")
    parser.add_argument(
        "--cost-key",
        default="4",
        choices=["1", "2", "3", "4"],
        help="代价函数编号，默认 4（模型综合代价）。",
    )
    parser.add_argument(
        "--difficulty",
        nargs="*",
        choices=["simple", "medium", "hard"],
        help="可选：只跑指定难度；默认跑全部固定测试图。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_dataset(cost_key=args.cost_key, difficulties=args.difficulty)
    paths = save_report(report)

    print("\n基线运行完成，结果文件如下：")
    for label, file_path in paths.items():
        print(f"- {label}: {file_path}")


if __name__ == "__main__":
    main()
