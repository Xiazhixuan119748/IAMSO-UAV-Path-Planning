"""Console input helpers for the UAV path-planning demo."""

from __future__ import annotations

from uav_planner.config import MODEL_WEIGHTS
from uav_planner.map_generator import MAP_GENERATION_LABELS, MAP_GENERATION_RANDOM, MAP_GENERATION_REACHABLE
from uav_planner.registry import ALGORITHMS, COST_FUNCTIONS


def get_user_input(map_size):
    """获取用户输入的起终点坐标。"""
    print(f"\n请输入起点和终点坐标（范围 0-{map_size - 1}）：")

    while True:
        try:
            start_x = int(input(f"起点 X (0-{map_size - 1}): "))
            start_y = int(input(f"起点 Y (0-{map_size - 1}): "))
            start = (start_x, start_y)
            if not (0 <= start_x < map_size and 0 <= start_y < map_size):
                print(f"坐标超出范围！请在 0-{map_size - 1} 之间选择。")
                continue
            break
        except ValueError:
            print("输入无效，请输入整数！")

    while True:
        try:
            goal_x = int(input(f"终点 X (0-{map_size - 1}): "))
            goal_y = int(input(f"终点 Y (0-{map_size - 1}): "))
            goal = (goal_x, goal_y)
            if not (0 <= goal_x < map_size and 0 <= goal_y < map_size):
                print(f"坐标超出范围！请在 0-{map_size - 1} 之间选择。")
                continue
            break
        except ValueError:
            print("输入无效，请输入整数！")

    return start, goal


def select_normal_map_size():
    """选择普通模式地图规模。"""
    options = {
        "1": ("简单", 30),
        "2": ("中等", 60),
        "3": ("困难", 90),
    }
    print("\n======== 选择地图难度 ========")
    print("  1. 简单模式 30×30")
    print("  2. 中等模式 60×60")
    print("  3. 困难模式 90×90")
    print("================================")

    while True:
        choice = input("请选择地图难度 (1/2/3): ").strip()
        if choice in options:
            name, map_size = options[choice]
            print(f"已选择: {name}模式 {map_size}×{map_size}")
            return name, map_size
        print("无效输入，请重新选择！")


def select_map_generation_mode():
    """选择正式运行使用的地图生成方式。"""
    options = {
        "1": (MAP_GENERATION_REACHABLE, MAP_GENERATION_LABELS[MAP_GENERATION_REACHABLE]),
        "2": (MAP_GENERATION_RANDOM, MAP_GENERATION_LABELS[MAP_GENERATION_RANDOM]),
    }
    print("\n======== 选择地图生成方式 ========")
    print("  1. 可达地图模式（保护起点到终点连通，适合算法对比）")
    print("  2. 随机压力地图模式（完全随机生成，可能天然无解）")
    print("==================================")

    while True:
        choice = input("请选择地图生成方式 (1/2): ").strip()
        if choice in options:
            mode, label = options[choice]
            print(f"已选择: {label}")
            return mode, label
        print("无效输入，请重新选择！")


def select_algorithm():
    """选择算法。"""
    print("\n========== 选择算法 ==========")
    for key, (name, _) in ALGORITHMS.items():
        print(f"  {key}. {name}")
    print("================================")

    while True:
        choice = input("请输入算法编号: ").strip()
        if choice in ALGORITHMS:
            name, func = ALGORITHMS[choice]
            print(f"已选择: {name}")
            return func, name
        print("无效输入，请重新选择！")


def select_cost_function():
    """选择代价函数。"""
    print("\n======== 选择代价函数 ========")
    for key, (name, _) in COST_FUNCTIONS.items():
        print(f"  {key}. {name}")
    print("================================")

    while True:
        choice = input("请输入代价函数编号: ").strip()
        if choice in COST_FUNCTIONS:
            name, func = COST_FUNCTIONS[choice]
            print(f"已选择: {name}")
            return func, name
        print("无效输入，请重新选择！")


def select_mode():
    """选择运行模式。"""
    print("\n========== 运行模式 ==========")
    print("  1. 单算法模式（选择一个算法+一个代价函数）")
    print("  2. 算法对比模式（选择一个代价函数，对比所有算法）")
    print("  3. 代价对比模式（选择一个算法，对比所有代价函数）")
    print("  4. 加权优化模式（输入权重，自动选择最优路径）")
    print("  5. 批量随机地图对比模式（100 次随机地图，统计所有算法指标）")
    print("  6. 左上到右下无 task 固定地图实验（可选全部算法批量或单一算法）")
    print("================================")

    while True:
        choice = input("请选择模式 (1/2/3/4/5/6): ").strip()
        if choice in ("1", "2", "3", "4", "5", "6"):
            return int(choice)
        print("无效输入，请重新选择！")


def get_weights():
    """获取用户输入的模型权重。"""
    print("\n======== 设置权重 ========")
    print("权重之和应为 1，例如：膨胀0.35 风场0.20 降雨0.25 巡航0.20")
    print(
        "直接回车使用默认权重："
        f"膨胀{MODEL_WEIGHTS[0]:.2f} "
        f"风场{MODEL_WEIGHTS[1]:.2f} "
        f"降雨{MODEL_WEIGHTS[2]:.2f} "
        f"巡航{MODEL_WEIGHTS[3]:.2f}"
    )
    print("================================")

    while True:
        try:
            raw1 = input("障碍膨胀权重 w_inf (0~1): ").strip()
            if not raw1:
                return MODEL_WEIGHTS
            w1 = float(raw1)
            w2 = float(input("风场权重 w_wind (0~1): "))
            w3 = float(input("降雨权重 w_rain (0~1): "))
            w4 = float(input("巡航价值权重 w_cruise (0~1): "))

            if w1 < 0 or w2 < 0 or w3 < 0 or w4 < 0:
                print("权重不能为负数！")
                continue

            total = w1 + w2 + w3 + w4
            if abs(total - 1.0) > 0.01:
                print(f"权重之和为 {total}，不等于 1！请重新输入。")
                continue

            return w1, w2, w3, w4
        except ValueError:
            print("输入无效，请输入数字！")
