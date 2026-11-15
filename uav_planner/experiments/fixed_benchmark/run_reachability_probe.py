"""Run the small-sample reachability probe for multi-shape obstacle maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from uav_planner.experiments.fixed_benchmark.reachability_probe import (  # noqa: E402
    run_reachability_probe,
    save_reachability_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估多形状障碍地图的理论可达率")
    return parser.parse_args()


def main() -> None:
    parse_args()
    report = run_reachability_probe()
    paths = save_reachability_report(report)
    print("\n评估完成，结果文件如下：")
    for label, file_path in paths.items():
        print(f"- {label}: {file_path}")


if __name__ == "__main__":
    main()
