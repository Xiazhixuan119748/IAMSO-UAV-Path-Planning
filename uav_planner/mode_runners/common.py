"""Shared helpers for interactive run modes."""

from __future__ import annotations

import time as time_module

from uav_planner.geometry import collision_cells_for_path
from uav_planner.metrics import calculate_extended_metrics
from uav_planner.models import PathMetrics
from uav_planner.output_paths import RunOutputPaths
from uav_planner.visualization import plot_result


def run_single(passable, speed_map, start, goal, algo_func, algo_name, cost_func, cost_name):
    """Run one algorithm and return a display-ready result dictionary."""
    start_time = time_module.perf_counter()
    try:
        path = algo_func(passable, speed_map, cost_func, start, goal)
    except Exception as exc:
        print(f"{algo_name}+{cost_name} 运行失败: {exc}")
        path = None
    runtime = time_module.perf_counter() - start_time

    if not path:
        path_metrics = PathMetrics(
            distance=None,
            time=None,
            safety=None,
            collision=None,
            smoothness=None,
            runtime=runtime,
            success_rate=0.0,
        )
        return {
            "path": None,
            "label": f"{algo_name}+{cost_name}",
            "status": "未找到路径",
            "collision_points": [],
            **path_metrics.as_dict(),
        }

    collision_points = sorted(collision_cells_for_path(passable, path))
    extended_metrics = calculate_extended_metrics(path, speed_map, passable=passable, runtime=runtime)
    collision = extended_metrics["collision"]
    path_metrics = PathMetrics(
        distance=extended_metrics["distance"],
        time=extended_metrics["time"],
        safety=extended_metrics["safety"],
        collision=collision,
        smoothness=extended_metrics["smoothness"],
        runtime=runtime,
        success_rate=1.0 if collision == 0 else 0.0,
        task_reward=extended_metrics["task_reward"],
        rain_exposure=extended_metrics["rain_exposure"],
        inflation_exposure=extended_metrics["inflation_exposure"],
        total_exposure=extended_metrics["total_exposure"],
        score=extended_metrics["score"],
    )
    return {
        "path": path,
        "label": f"{algo_name}+{cost_name}",
        "status": "成功" if collision == 0 else "碰撞失败",
        "collision_points": collision_points,
        **path_metrics.as_dict(),
    }


def _format_value(value, precision=4):
    return "-" if value is None else f"{value:.{precision}f}"


def _format_percent(value):
    return "-" if value is None else f"{value:.2%}"


def _format_collision(value):
    return "-" if value is None else str(value)


def print_result(result):
    """Print one planning result."""
    print(
        f"  状态={result.get('status', '-')}, "
        f"路径长度={_format_value(result['distance'])}, "
        f"通行时间={_format_value(result['time'])}, "
        f"安全度={_format_percent(result['safety'])}, "
        f"碰撞次数={_format_collision(result['collision'])}, "
        f"平滑度={_format_value(result['smoothness'])}, "
        f"任务收益={_format_value(result.get('task_reward'))}, "
        f"综合评分/100={_format_value(result.get('score'))}, "
        f"运行时间={result['runtime']:.4f}s, "
        f"成功率={result['success_rate']:.2%}"
    )
    if result.get("collision_points"):
        preview = result["collision_points"][:8]
        suffix = " ..." if len(result["collision_points"]) > len(preview) else ""
        print(f"  碰撞位置={preview}{suffix}")


def print_table(results, header_col="算法"):
    """Print a comparison table."""
    print("\n" + "=" * 164)
    print(
        f"{header_col:<24} {'状态':>10} {'成功率':>8} {'路径长度':>10} {'通行时间':>10} "
        f"{'安全度':>10} {'碰撞次数':>10} {'平滑度':>10} {'任务收益':>10} "
        f"{'综合评分/100':>12} {'运行时间/s':>12}"
    )
    print("-" * 164)
    for r in results:
        print(
            f"{r['label']:<24} "
            f"{r.get('status', '-'):>10} "
            f"{r['success_rate']:>7.2%} "
            f"{_format_value(r['distance']):>10} "
            f"{_format_value(r['time']):>10} "
            f"{_format_percent(r['safety']):>10} "
            f"{_format_collision(r['collision']):>10} "
            f"{_format_value(r['smoothness']):>10} "
            f"{_format_value(r.get('task_reward')):>10} "
            f"{_format_value(r.get('score')):>12} "
            f"{r['runtime']:>12.4f}"
        )
    print("=" * 164)


def plot_paths(
    passable,
    speed_map,
    results,
    start,
    goal,
    animate=True,
    save_gif=True,
    save_png=False,
    output_paths: RunOutputPaths | None = None,
):
    """Plot all returned paths, including failed collision paths."""
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
    drawable_results = [r for r in results if r.get("path")]
    if not drawable_results:
        missing = [r["label"] for r in results if not r.get("path")]
        if missing:
            print(f"算法未规划出路径，无法绘制：{', '.join(missing)}")
        else:
            print("没有可绘制路径。")
        if not save_png:
            return

    failed = [r["label"] for r in drawable_results if (r.get("collision") or 0) > 0]
    if failed:
        print(f"以下路径发生碰撞，已绘制失败路径并标注碰撞位置：{', '.join(failed)}")

    plot_result(
        passable,
        speed_map,
        [
            (
                r["path"],
                r["label"],
                colors[i % len(colors)],
                r.get("collision_points", []),
            )
            for i, r in enumerate(drawable_results)
        ],
        [(r["distance"], r["time"], r["safety"]) for r in drawable_results],
        start,
        goal,
        animate=animate and bool(drawable_results),
        save_gif=save_gif and bool(drawable_results),
        save_png=save_png,
        output_paths=output_paths,
    )
