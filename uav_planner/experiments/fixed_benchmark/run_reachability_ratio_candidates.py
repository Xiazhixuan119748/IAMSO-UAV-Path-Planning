"""Run reachability probes for candidate obstacle-ratio settings.

This script is kept with the fixed benchmark so it can evaluate candidate
map-generation parameters without changing the production map generator.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from uav_planner.experiments.fixed_benchmark.reachability_probe import (  # noqa: E402
    run_reachability_probe,
    save_reachability_report,
)


RATIO_CANDIDATE_CASES = [
    {
        "name": "simple_15",
        "label": "简单-15%",
        "size": 30,
        "target_ratio": 0.15,
        "samples": 100,
        "seed_base": 2026071100,
    },
    {
        "name": "medium_20",
        "label": "中等-20%",
        "size": 60,
        "target_ratio": 0.20,
        "samples": 100,
        "seed_base": 2026072100,
    },
    {
        "name": "medium_25",
        "label": "中等-25%",
        "size": 60,
        "target_ratio": 0.25,
        "samples": 100,
        "seed_base": 2026072200,
    },
    {
        "name": "medium_30",
        "label": "中等-30%",
        "size": 60,
        "target_ratio": 0.30,
        "samples": 100,
        "seed_base": 2026072300,
    },
    {
        "name": "hard_20",
        "label": "困难-20%",
        "size": 90,
        "target_ratio": 0.20,
        "samples": 100,
        "seed_base": 2026073100,
    },
    {
        "name": "hard_25",
        "label": "困难-25%",
        "size": 90,
        "target_ratio": 0.25,
        "samples": 100,
        "seed_base": 2026073200,
    },
    {
        "name": "hard_30",
        "label": "困难-30%",
        "size": 90,
        "target_ratio": 0.30,
        "samples": 100,
        "seed_base": 2026073300,
    },
    {
        "name": "hard_35",
        "label": "困难-35%",
        "size": 90,
        "target_ratio": 0.35,
        "samples": 100,
        "seed_base": 2026073400,
    },
]


def main() -> None:
    report = run_reachability_probe(RATIO_CANDIDATE_CASES)
    report["assumption"]["purpose"] = "测试中等/困难模式提高障碍覆盖率后的理论可达率"
    paths = save_reachability_report(report)
    print("\n候选障碍覆盖率理论可达率评估完成，结果文件如下：")
    for label, file_path in paths.items():
        print(f"- {label}: {file_path}")


if __name__ == "__main__":
    main()
