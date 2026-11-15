"""Interactive entry point for the formal no-task corner experiment."""

from __future__ import annotations

from uav_planner.cli_inputs import select_algorithm
from uav_planner.experiments.no_task_corner.dataset import experiment_root
from uav_planner.experiments.no_task_corner.runner import (
    evaluate_dataset,
    print_comparison_table,
    save_report,
)


MEDIUM20_DATASET_DIR = experiment_root() / "datasets" / "medium60_density20_reachable_100"


def run_no_task_corner_experiment_mode():
    """Run the fixed no-task experiment without changing modes 1-5."""
    print("\n======== 模式 6：固定地图实验 ========")
    print("  1. 全部算法批量比较")
    print("  2. 单一算法实验")
    print("====================================")
    while True:
        experiment_choice = input("请选择实验类型 (1/2): ").strip()
        if experiment_choice in ("1", "2"):
            break
        print("输入无效，请选择 1 或 2。")

    algorithm_names = None
    if experiment_choice == "2":
        _, algorithm_name = select_algorithm()
        algorithm_names = [algorithm_name]

    options = {
        "1": (None, None),
        "2": (["simple"], None),
        "3": (["medium"], None),
        "4": (["hard"], None),
        "5": (["medium20"], MEDIUM20_DATASET_DIR),
    }
    print("\n======== 选择固定地图范围 ========")
    print("  1. 全部难度（每种 100 张）")
    print("  2. 仅简单（30x30，100 张，15% 障碍）")
    print("  3. 仅中等（60x60，100 张，15% 障碍）")
    print("  4. 仅困难（90x90，100 张，20% 障碍）")
    print("  5. 仅中等20%障碍（60x60，100 张，单独地图集）")
    print("固定端点：起点=(0, 0)，终点=(size-1, size-1)，关闭任务价值。")

    while True:
        choice = input("请选择实验范围 (1/2/3/4/5): ").strip()
        if choice in options:
            break
        print("输入无效，请选择 1、2、3、4 或 5。")

    difficulties, dataset_dir = options[choice]
    report = evaluate_dataset(
        difficulties=difficulties,
        algorithm_names=algorithm_names,
        dataset_dir=dataset_dir,
    )
    print_comparison_table(report)
    paths = save_report(report)
    print("\n模式 6 实验完成，实验结果已保存到：")
    for label, file_path in paths.items():
        print(f"- {label}: {file_path}")
