"""Runner for batch random-map comparison mode."""

from __future__ import annotations

from uav_planner.batch_comparison import (
    get_batch_iterations,
    print_batch_comparison_table,
    run_batch_comparison,
    save_batch_comparison_markdown,
    select_batch_map_size,
)
from uav_planner.cli_inputs import select_map_generation_mode
from uav_planner.config import MODEL_WEIGHTS
from uav_planner.pathfinding import model_cost


def _model_cost_label():
    return (
        "模型综合代价函数"
        f"（膨胀{MODEL_WEIGHTS[0]:.2f}，"
        f"风场{MODEL_WEIGHTS[1]:.2f}，"
        f"降雨{MODEL_WEIGHTS[2]:.2f}，"
        f"巡航{MODEL_WEIGHTS[3]:.2f}）"
    )


def run_batch_mode():
    """Run mode 5 with a fixed center start and random goals."""
    map_mode, map_size = select_batch_map_size()
    map_generation_mode, map_generation_label = select_map_generation_mode()
    iterations = get_batch_iterations(default=100)
    start = (map_size // 2, map_size // 2)
    cost_name = _model_cost_label()

    print(f"\n开始批量随机地图对比实验：{iterations} 次")
    print(f"地图模式：{map_mode}模式 {map_size}×{map_size}")
    print(f"地图生成方式：{map_generation_label}")
    print(f"固定起点：{start}")
    print(f"默认代价函数：{cost_name}")
    print("每一轮都会随机生成终点和地图，并让所有算法在同一张地图上运行。")
    print("提示：RRT、MSO、AO 运行较慢，100 次完整实验可能需要较长时间。\n")

    rows, map_stats = run_batch_comparison(
        start,
        model_cost,
        iterations=iterations,
        map_size=map_size,
        generation_mode=map_generation_mode,
    )
    print(f"平均障碍比例：{map_stats['blocked_ratio']:.2%}")
    print(f"平均起终点曼哈顿距离：{map_stats['goal_distance']:.2f}")
    print_batch_comparison_table(rows)

    output_path = save_batch_comparison_markdown(
        rows,
        iterations,
        cost_name,
        map_mode,
        map_size,
        start,
        map_stats,
        generation_mode=map_generation_mode,
        generation_label=map_generation_label,
    )
    print(f"\n比较结果已保存到: {output_path}")
