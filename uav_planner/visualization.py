"""
可视化模块
负责绘制地图和路径。
"""

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from uav_planner.config import MODEL_WEIGHTS, NORMAL_SPEED
from uav_planner.costmap import is_layer_map, speed_grid
from uav_planner.output_paths import OUTPUTS_ROOT, RunOutputPaths
from uav_planner.visualization_utils import (
    path_xy_points,
    plot_path_curvature_overlay,
    plot_path_heading_arrows,
    plot_sparse_wind_arrows,
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

_ACTIVE_ANIMATIONS = []
ANIMATION_DIR = OUTPUTS_ROOT / "animations"
STATIC_RESULT_DIR = OUTPUTS_ROOT / "out_png"
MAP_LAYERS_DIR = OUTPUTS_ROOT / "map_layers"


def _hide_ticks(ax):
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)


def _add_grid(ax, size):
    if size <= 120:
        ax.set_xticks(np.arange(-.5, size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, size, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=0.3)


def _unpack_path_item(item):
    if len(item) == 3:
        path, label, color = item
        return path, label, color, []
    path, label, color, collision_points = item
    return path, label, color, collision_points


def _plot_collision_points(ax, collision_points):
    if not collision_points:
        return
    rows = [p[0] for p in collision_points if isinstance(p, tuple) and len(p) == 2]
    cols = [p[1] for p in collision_points if isinstance(p, tuple) and len(p) == 2]
    if not rows:
        return
    ax.scatter(
        cols,
        rows,
        marker="x",
        s=95,
        c="red",
        linewidths=2.2,
        label="Collision",
        zorder=9,
    )


def _plot_paths(ax, paths, metrics_list):
    for item, (dist, time, safe) in zip(paths, metrics_list):
        path, label, color, collision_points = _unpack_path_item(item)
        x, y = path_xy_points(path)
        linestyle = "--" if collision_points else "-"
        collision_suffix = f", C={len(collision_points)}" if collision_points else ""
        ax.plot(
            x,
            y,
            color=color,
            linewidth=3,
            linestyle=linestyle,
            label=f'{label} (D={dist:.2f}, T={time:.2f}, S={safe:.0%}{collision_suffix})',
        )
        plot_path_heading_arrows(ax, path, color=color)
        plot_path_curvature_overlay(ax, path, color=color)
        _plot_collision_points(ax, collision_points)


def _mark_start_goal(ax, start, goal):
    start_cell = Rectangle(
        (start[1] - 0.5, start[0] - 0.5),
        1,
        1,
        facecolor="yellow",
        edgecolor="black",
        linewidth=1.2,
        label="Start",
        zorder=6,
    )
    goal_cell = Rectangle(
        (goal[1] - 0.5, goal[0] - 0.5),
        1,
        1,
        facecolor="cyan",
        edgecolor="black",
        linewidth=1.2,
        label="Goal",
        zorder=6,
    )
    ax.add_patch(start_cell)
    ax.add_patch(goal_cell)


def _mark_tasks(ax, layers):
    if layers.get("tasks"):
        task_y = [p[0] for p in layers["tasks"]]
        task_x = [p[1] for p in layers["tasks"]]
        ax.scatter(task_x, task_y, c="gold", s=90, marker="*", edgecolors="black", label="Task", zorder=4)


def _plot_layered_result(passable, layers, paths, metrics_list, start, goal):
    size = passable.shape[0]
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=220, constrained_layout=True)
    overview_ax, inflation_ax, wind_ax, cruise_ax = axes.ravel()

    display_map = np.clip(layers["R"], 0.0, 1.0)
    display_map = display_map.copy()
    display_map[passable == 0] = 1.25
    rain_cmap = ListedColormap(["white", "#c7e9ff", "#69b3ff", "#1d4f91", "black"])
    overview = overview_ax.imshow(display_map, cmap=rain_cmap, interpolation="nearest", vmin=0.0, vmax=1.25)
    overview_ax.contour(
        layers["I"],
        levels=[0.15, 0.35],
        colors=["#f2a65a", "#c44536"],
        linewidths=[0.7, 1.0],
        alpha=0.75,
    )
    stride = max(6, size // 8)
    plot_sparse_wind_arrows(overview_ax, layers["W"], stride=stride, color="#263238", alpha=0.55, zorder=3)
    _mark_tasks(overview_ax, layers)
    _plot_paths(overview_ax, paths, metrics_list)
    _mark_start_goal(overview_ax, start, goal)
    overview_ax.set_title("总览：降雨 + 障碍 + 膨胀轮廓 + 风向 + 路径航向/曲率")
    overview_ax.legend(fontsize=9, loc="upper right")
    _add_grid(overview_ax, size)
    _hide_ticks(overview_ax)
    fig.colorbar(overview, ax=overview_ax, fraction=0.046, pad=0.04, label="Rain / obstacle")

    inflation = inflation_ax.imshow(layers["I"], cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0)
    inflation_ax.contour(passable == 0, levels=[0.5], colors="white", linewidths=0.8)
    _mark_start_goal(inflation_ax, start, goal)
    inflation_ax.set_title("障碍膨胀层 I：越亮越接近障碍")
    _hide_ticks(inflation_ax)
    fig.colorbar(inflation, ax=inflation_ax, fraction=0.046, pad=0.04, label="Inflation risk")

    wind_strength = np.linalg.norm(layers["W"], axis=2)
    wind_img = wind_ax.imshow(wind_strength, cmap="YlGnBu", interpolation="nearest", vmin=0.0)
    stride = max(3, size // 18)
    plot_sparse_wind_arrows(wind_ax, layers["W"], stride=stride, color="black", alpha=0.65, zorder=3)
    _mark_start_goal(wind_ax, start, goal)
    wind_ax.set_title("风场向量层 W：底色为风速，箭头为风向")
    _hide_ticks(wind_ax)
    fig.colorbar(wind_img, ax=wind_ax, fraction=0.046, pad=0.04, label="Wind speed")

    cruise = cruise_ax.imshow(layers["V"], cmap="YlOrBr", interpolation="nearest", vmin=0.0, vmax=1.0)
    _mark_tasks(cruise_ax, layers)
    _mark_start_goal(cruise_ax, start, goal)
    cruise_ax.set_title("巡航价值层 V：越亮越靠近任务价值区")
    _hide_ticks(cruise_ax)
    fig.colorbar(cruise, ax=cruise_ax, fraction=0.046, pad=0.04, label="Cruise value")

    fig.suptitle("UAV Global Path Planning Layered View", fontsize=18)
    return fig


def _model_cost_layer(layers):
    w_inf, w_wind, w_rain, w_cruise = MODEL_WEIGHTS
    wind_strength = np.linalg.norm(layers["W"], axis=2)
    max_wind = float(wind_strength.max())
    if max_wind > 1e-12:
        wind_strength = wind_strength / max_wind
    return (
        w_inf * layers["I"]
        + w_wind * wind_strength
        + w_rain * layers["R"]
        + w_cruise * (1.0 - layers["V"])
    )


def _plot_map_layers(passable, layers, start, goal):
    size = passable.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=220, constrained_layout=True)
    obstacle_ax, inflation_ax, wind_ax, rain_ax, cruise_ax, model_ax = axes.ravel()

    obstacle_img = obstacle_ax.imshow(1 - passable, cmap="binary", interpolation="nearest", vmin=0.0, vmax=1.0)
    obstacle_ax.set_title("障碍物地图 O")
    fig.colorbar(obstacle_img, ax=obstacle_ax, fraction=0.046, pad=0.04, label="Obstacle")

    inflation_img = inflation_ax.imshow(layers["I"], cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0)
    inflation_ax.contour(passable == 0, levels=[0.5], colors="white", linewidths=0.8)
    inflation_ax.set_title("障碍膨胀层 I")
    fig.colorbar(inflation_img, ax=inflation_ax, fraction=0.046, pad=0.04, label="Inflation risk")

    wind_strength = np.linalg.norm(layers["W"], axis=2)
    wind_img = wind_ax.imshow(wind_strength, cmap="YlGnBu", interpolation="nearest", vmin=0.0)
    plot_sparse_wind_arrows(
        wind_ax,
        layers["W"],
        stride=max(3, size // 18),
        color="black",
        alpha=0.65,
        zorder=3,
    )
    wind_ax.set_title("风场 W：底色风速，箭头风向")
    fig.colorbar(wind_img, ax=wind_ax, fraction=0.046, pad=0.04, label="Wind speed")

    rain_img = rain_ax.imshow(layers["R"], cmap="Blues", interpolation="nearest", vmin=0.0, vmax=1.0)
    rain_ax.set_title("降雨强度层 R")
    fig.colorbar(rain_img, ax=rain_ax, fraction=0.046, pad=0.04, label="Rain risk")

    cruise_img = cruise_ax.imshow(layers["V"], cmap="YlOrBr", interpolation="nearest", vmin=0.0, vmax=1.0)
    _mark_tasks(cruise_ax, layers)
    cruise_ax.set_title("巡航价值层 V")
    fig.colorbar(cruise_img, ax=cruise_ax, fraction=0.046, pad=0.04, label="Cruise value")

    model_img = model_ax.imshow(_model_cost_layer(layers), cmap="viridis", interpolation="nearest")
    model_ax.set_title("模型综合代价层")
    fig.colorbar(model_img, ax=model_ax, fraction=0.046, pad=0.04, label="Weighted cost")

    for layer_ax in axes.ravel():
        _mark_start_goal(layer_ax, start, goal)
        _hide_ticks(layer_ax)

    fig.suptitle("地图分层可视化", fontsize=18)
    return fig


def _densify_path(path, samples_per_cell=6):
    """Linearly interpolate a path so animation movement is smooth."""
    if not path:
        return np.empty((0, 2), dtype=float)

    points = np.asarray(path, dtype=float)
    dense = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(end - start))
        steps = max(1, int(np.ceil(distance * samples_per_cell)))
        for i in range(1, steps + 1):
            dense.append(start + (end - start) * (i / steps))
    return np.asarray(dense, dtype=float)


def _draw_animation_background(ax, passable, speed_map, start, goal):
    size = passable.shape[0]
    layers = speed_map if is_layer_map(speed_map) else None

    if layers is not None:
        display_map = np.clip(layers["R"], 0.0, 1.0).copy()
        display_map[passable == 0] = 1.25
        rain_cmap = ListedColormap(["white", "#c7e9ff", "#69b3ff", "#1d4f91", "black"])
        ax.imshow(display_map, cmap=rain_cmap, interpolation="nearest", vmin=0.0, vmax=1.25)
        ax.contour(
            layers["I"],
            levels=[0.15, 0.35],
            colors=["#f2a65a", "#c44536"],
            linewidths=[0.7, 1.0],
            alpha=0.75,
        )
        stride = max(6, size // 8)
        plot_sparse_wind_arrows(ax, layers["W"], stride=stride, color="#263238", alpha=0.5, zorder=3)
        _mark_tasks(ax, layers)
        title = "动态路径：无人机沿规划路径飞行"
    else:
        speed_map = speed_grid(speed_map)
        display_map = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if passable[i, j] == 0:
                    display_map[i, j] = 3
                elif speed_map[i, j] == NORMAL_SPEED * 0.5:
                    display_map[i, j] = 2
                elif speed_map[i, j] == NORMAL_SPEED * 0.75:
                    display_map[i, j] = 1
        cmap = ListedColormap(["white", "purple", "orange", "black"])
        ax.imshow(display_map, cmap=cmap, interpolation="nearest")
        title = "动态路径：无人机沿规划路径飞行"

    _mark_start_goal(ax, start, goal)
    _add_grid(ax, size)
    _hide_ticks(ax)
    ax.set_title(title)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)


def _animation_output_path(output_paths=None):
    if output_paths is not None:
        return output_paths.animation
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ANIMATION_DIR / f"uav_path_animation_{timestamp}.gif"


def _static_result_output_path(output_paths=None):
    if output_paths is not None:
        return output_paths.static_result
    STATIC_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return STATIC_RESULT_DIR / f"uav_path_result_{timestamp}.png"


def _map_layers_output_path(output_paths=None):
    if output_paths is not None:
        return output_paths.map_layers
    MAP_LAYERS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return MAP_LAYERS_DIR / f"uav_map_layers_{timestamp}.png"


def _save_static_result(fig, output_paths=None):
    output_path = _static_result_output_path(output_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"静态路径结果图已保存到: {output_path}")
    return output_path


def _save_map_layers(fig, output_paths=None):
    output_path = _map_layers_output_path(output_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"地图分层图已保存到: {output_path}")
    return output_path


def _limit_animation_frames(points, max_frames):
    if len(points) <= max_frames:
        return points
    indices = np.linspace(0, len(points) - 1, max_frames).astype(int)
    return points[indices]


def animate_paths(
    passable,
    speed_map,
    paths,
    start,
    goal,
    interval=80,
    save_gif=True,
    max_frames=240,
    output_paths: RunOutputPaths | None = None,
):
    """Animate one or more UAV paths on the overview map."""
    valid_paths = [_unpack_path_item(item) for item in paths if _unpack_path_item(item)[0]]
    if not valid_paths:
        return None

    dense_paths = [
        (_limit_animation_frames(_densify_path(path), max_frames), label, color)
        for path, label, color, _collision_points in valid_paths
    ]
    frame_count = max(len(points) for points, _label, _color in dense_paths)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=160)
    _draw_animation_background(ax, passable, speed_map, start, goal)

    artists = []
    for points, label, color in dense_paths:
        ax.plot(points[:, 1], points[:, 0], color=color, linewidth=1.2, alpha=0.25)
        trail_line, = ax.plot([], [], color=color, linewidth=2.8, label=label, zorder=7)
        uav_marker, = ax.plot(
            [],
            [],
            marker="^",
            markersize=12,
            markerfacecolor=color,
            markeredgecolor="black",
            linestyle="None",
            zorder=8,
        )
        artists.append((points, trail_line, uav_marker))

    ax.legend(fontsize=9, loc="upper right")

    def update(frame):
        changed = []
        for points, trail_line, uav_marker in artists:
            idx = min(frame, len(points) - 1)
            segment = points[:idx + 1]
            trail_line.set_data(segment[:, 1], segment[:, 0])
            uav_marker.set_data([points[idx, 1]], [points[idx, 0]])
            changed.extend([trail_line, uav_marker])
        return changed

    animation = FuncAnimation(fig, update, frames=frame_count, interval=interval, blit=False, repeat=False)
    fig._uav_animation = animation
    _ACTIVE_ANIMATIONS.append(animation)

    if save_gif:
        output_path = _animation_output_path(output_paths)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, int(round(1000 / interval)))
        animation.save(str(output_path), writer=PillowWriter(fps=fps), dpi=120)
        print(f"动态路径 GIF 已保存到: {output_path}")

    return fig


def plot_result(
    passable,
    speed_map,
    paths,
    metrics_list,
    start,
    goal,
    animate=True,
    save_gif=True,
    save_png=False,
    output_paths: RunOutputPaths | None = None,
):
    """
    绘制地图和多条路径。

    Args:
        passable:     通行矩阵
        speed_map:    速度矩阵
        paths:        路径列表，每个元素为 (path, label, color)
        metrics_list: 对应路径的指标列表，每个元素为 (distance, time, safety)
        start:        起点坐标
        goal:         终点坐标
    """
    layers = speed_map if is_layer_map(speed_map) else None
    if layers is not None:
        fig = _plot_layered_result(passable, layers, paths, metrics_list, start, goal)
        if save_png:
            _save_static_result(fig, output_paths)
            layer_fig = _plot_map_layers(passable, layers, start, goal)
            _save_map_layers(layer_fig, output_paths)
        if animate:
            animate_paths(
                passable,
                speed_map,
                paths,
                start,
                goal,
                save_gif=save_gif,
                output_paths=output_paths,
            )
        plt.show()
        return

    fig, ax = plt.subplots(figsize=(14, 14), dpi=300)
    speed_map = speed_grid(speed_map)
    size = passable.shape[0]

    # 构建显示地图
    display_map = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if passable[i, j] == 0:
                display_map[i, j] = 3
            elif speed_map[i, j] == NORMAL_SPEED * 0.5:
                display_map[i, j] = 2
            elif speed_map[i, j] == NORMAL_SPEED * 0.75:
                display_map[i, j] = 1
            else:
                display_map[i, j] = 0

    cmap = ListedColormap(["white", "purple", "orange", "black"])
    ax.imshow(display_map, cmap=cmap, interpolation="nearest")

    # 网格线
    _add_grid(ax, size)
    _hide_ticks(ax)

    # 绘制路径
    _plot_paths(ax, paths, metrics_list)

    # 起终点标记
    _mark_start_goal(ax, start, goal)

    ax.legend(fontsize=14)
    plt.title("UAV Global Path Planning Demo (5 Constraints)", fontsize=18)
    if save_png:
        _save_static_result(fig, output_paths)
    if animate:
        animate_paths(
            passable,
            speed_map,
            paths,
            start,
            goal,
            save_gif=save_gif,
            output_paths=output_paths,
        )
    plt.show()
