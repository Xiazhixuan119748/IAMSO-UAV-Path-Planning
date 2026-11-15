"""
路径指标计算模块
"""

import math

from uav_planner.config import (
    EVALUATION_SCORE_WEIGHTS,
    EXPOSURE_INFLATION_THRESHOLD,
    EXPOSURE_INTENSITY_GAMMA,
    EXPOSURE_LENGTH_BETA,
    EXPOSURE_RAIN_THRESHOLD,
    MODEL_WEIGHTS,
    NORMAL_SPEED,
)
from uav_planner.costmap import (
    C_cruise,
    C_inf,
    C_rain,
    C_wind,
    effective_speed,
    heading_vector,
    is_layer_map,
    speed_grid,
)
from uav_planner.geometry import collision_cells_for_path
from uav_planner.metaheuristic.encoding import polyline_length, sample_path


def _sampled_points(path, step=0.5):
    if not path:
        return []
    is_grid_path = all(float(point[0]).is_integer() and float(point[1]).is_integer() for point in path)
    return list(path) if is_grid_path else sample_path(path, step=step)


def _nearest_valid_cell(point, size):
    row = int(round(float(point[0])))
    col = int(round(float(point[1])))
    if 0 <= row < size and 0 <= col < size:
        return row, col
    return None


def _norm(value, min_value, max_value):
    if value is None:
        return None
    span = max_value - min_value
    if abs(span) <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, (value - min_value) / span))


def calculate_path_metrics(path, speed_map):
    """
    计算路径的各项指标。

    Args:
        path:      路径坐标列表 [(row, col), ...]
        speed_map: 速度矩阵

    Returns:
        distance: 路径步数
        time:     总通行时间
        safety:   安全度（正常区域格子占比）
    """
    if not path:
        return 0, 0, 0

    layers = speed_map if is_layer_map(speed_map) else None
    speed_map = speed_grid(speed_map)

    is_grid_path = all(float(point[0]).is_integer() and float(point[1]).is_integer() for point in path)
    if is_grid_path:
        distance = polyline_length(path)
        total_time = 0
        safety_total = 0.0

        prev = None
        for point in path:
            x, y = int(point[0]), int(point[1])
            step_distance = 0.0 if prev is None else ((point[0] - prev[0]) ** 2 + (point[1] - prev[1]) ** 2) ** 0.5
            heading = (0.0, 0.0) if prev is None else heading_vector(prev, point)
            speed = effective_speed(point, heading, layers if layers is not None else speed_map)
            total_time += step_distance / speed if speed > 0 else 0
            if layers is not None:
                risk = 0.5 * float(layers["I"][x, y]) + 0.5 * float(layers["R"][x, y])
                safety_total += max(0.0, 1.0 - risk)
            else:
                safety_total += 1.0 if speed_map[x, y] == NORMAL_SPEED else 0.0
            prev = point

        safety = safety_total / len(path)
        return distance, total_time, safety

    distance = polyline_length(path)

    total_time = 0
    safety_total = 0.0
    samples = sample_path(path, step=0.5)
    size = speed_map.shape[0]

    prev = None
    valid_count = 0
    for point in samples:
        x = int(round(point[0]))
        y = int(round(point[1]))
        if not (0 <= x < size and 0 <= y < size):
            prev = point
            continue

        step_distance = 0.0 if prev is None else ((point[0] - prev[0]) ** 2 + (point[1] - prev[1]) ** 2) ** 0.5
        heading = (0.0, 0.0) if prev is None else heading_vector(prev, point)
        speed = effective_speed((x, y), heading, layers if layers is not None else speed_map)
        total_time += step_distance / speed if speed > 0 else 0
        valid_count += 1
        if layers is not None:
            risk = 0.5 * float(layers["I"][x, y]) + 0.5 * float(layers["R"][x, y])
            safety_total += max(0.0, 1.0 - risk)
        elif speed_map[x, y] == NORMAL_SPEED:
            safety_total += 1.0
        prev = point

    safety = safety_total / valid_count if valid_count else 0

    return distance, total_time, safety


def calculate_collision_count(path, passable):
    """Count blocked grid cells intersected by the continuous path."""
    if not path:
        return 0

    if is_layer_map(passable):
        passable = 1 - passable["O"]
    size = passable.shape[0]
    collisions = set(collision_cells_for_path(passable, path))
    for point in sample_path(path, step=0.5):
        x = int(round(point[0]))
        y = int(round(point[1]))
        if not (0 <= x < size and 0 <= y < size):
            collisions.add(("out", x, y))
    return len(collisions)


def calculate_smoothness(path):
    """Calculate path smoothness as the sum of squared turning angles."""
    if not path or len(path) < 3:
        return 0.0

    smoothness = 0.0
    points = [(float(p[0]), float(p[1])) for p in path]
    for prev_point, point, next_point in zip(points[:-2], points[1:-1], points[2:]):
        v1 = (point[0] - prev_point[0], point[1] - prev_point[1])
        v2 = (next_point[0] - point[0], next_point[1] - point[1])
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 <= 1e-9 or n2 <= 1e-9:
            continue
        cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cosine = max(-1.0, min(1.0, cosine))
        angle = math.acos(cosine)
        smoothness += angle * angle
    return smoothness


def calculate_task_reward(path, layers):
    """Calculate mean cruise/task value along a path."""
    if not path or not is_layer_map(layers):
        return 0.0

    speed_map = speed_grid(layers)
    size = speed_map.shape[0]
    total = 0.0
    count = 0
    for point in _sampled_points(path):
        cell = _nearest_valid_cell(point, size)
        if cell is None:
            continue
        row, col = cell
        total += float(layers["V"][row, col])
        count += 1
    return total / count if count else 0.0


def calculate_environment_risk(path, layers):
    """Calculate mean environment risk from inflation and rain layers."""
    if not path or not is_layer_map(layers):
        return 0.0

    speed_map = speed_grid(layers)
    size = speed_map.shape[0]
    total = 0.0
    count = 0
    for point in _sampled_points(path):
        cell = _nearest_valid_cell(point, size)
        if cell is None:
            continue
        row, col = cell
        total += 0.5 * float(layers["I"][row, col]) + 0.5 * float(layers["R"][row, col])
        count += 1
    return total / count if count else 0.0


def _finalize_exposure_segment(integral, length, beta):
    if length <= 1e-12:
        return 0.0
    return integral * (length ** max(0.0, beta - 1.0))


def _continuous_exposure(path, layers, layer_key, threshold, gamma, beta, sample_step=0.5):
    if not path or not is_layer_map(layers):
        return 0.0

    points = list(sample_path(path, step=sample_step))
    size = layers["O"].shape[0]
    total_exposure = 0.0
    segment_integral = 0.0
    segment_length = 0.0
    prev_point = None

    for point in points:
        cell = _nearest_valid_cell(point, size)
        step_distance = 0.0 if prev_point is None else math.dist(prev_point, point)
        if cell is None:
            total_exposure += _finalize_exposure_segment(segment_integral, segment_length, beta)
            segment_integral = 0.0
            segment_length = 0.0
            prev_point = point
            continue

        row, col = cell
        risk_value = float(layers[layer_key][row, col])
        if risk_value > threshold:
            segment_integral += (risk_value ** gamma) * step_distance
            segment_length += step_distance
        else:
            total_exposure += _finalize_exposure_segment(segment_integral, segment_length, beta)
            segment_integral = 0.0
            segment_length = 0.0
        prev_point = point

    total_exposure += _finalize_exposure_segment(segment_integral, segment_length, beta)
    return total_exposure


def calculate_exposure_metrics(path, layers):
    """Calculate nonlinear continuous exposure for rain and obstacle-inflation risk."""
    rain_exposure = _continuous_exposure(
        path,
        layers,
        "R",
        EXPOSURE_RAIN_THRESHOLD,
        EXPOSURE_INTENSITY_GAMMA,
        EXPOSURE_LENGTH_BETA,
    )
    inflation_exposure = _continuous_exposure(
        path,
        layers,
        "I",
        EXPOSURE_INFLATION_THRESHOLD,
        EXPOSURE_INTENSITY_GAMMA,
        EXPOSURE_LENGTH_BETA,
    )
    return {
        "rain_exposure": rain_exposure,
        "inflation_exposure": inflation_exposure,
        "total_exposure": rain_exposure + inflation_exposure,
    }


def _positive_norm(value, scale):
    if value is None:
        return None
    value = max(0.0, float(value))
    scale = max(1e-9, float(scale))
    return value / (value + scale)


def calculate_evaluation_score(
    *,
    distance,
    travel_time,
    safety,
    collision,
    smoothness,
    runtime,
    task_reward,
    exposure=0.0,
    map_size=100,
    weights=EVALUATION_SCORE_WEIGHTS,
):
    """Calculate final evaluation score on a 0-100 scale. Higher is better."""
    if distance is None or travel_time is None or safety is None or smoothness is None:
        return None

    length_norm = _positive_norm(distance, map_size)
    time_norm = _positive_norm(travel_time, map_size / max(NORMAL_SPEED, 1e-9))
    mean_risk = max(0.0, 1.0 - float(safety))
    exposure_norm = _positive_norm(exposure, map_size)
    risk_norm = 0.6 * mean_risk + 0.4 * exposure_norm
    collision_norm = _positive_norm(collision or 0, 1.0)
    smoothness_norm = _positive_norm(smoothness, 10.0)
    runtime_norm = _positive_norm(runtime, 1.0)
    task_reward_norm = max(0.0, min(1.0, float(task_reward or 0.0)))

    weighted_quality = (
        weights["length"] * (1.0 - length_norm)
        + weights["time"] * (1.0 - time_norm)
        + weights["risk"] * (1.0 - risk_norm)
        + weights["collision"] * (1.0 - collision_norm)
        + weights["smoothness"] * (1.0 - smoothness_norm)
        + weights["runtime"] * (1.0 - runtime_norm)
        + weights["task"] * task_reward_norm
    )
    total_weight = sum(max(0.0, float(weights.get(key, 0.0))) for key in (
        "length",
        "time",
        "risk",
        "collision",
        "smoothness",
        "runtime",
        "task",
    ))
    if total_weight <= 1e-12:
        return None
    return max(0.0, min(100.0, 100.0 * weighted_quality / total_weight))


def calculate_extended_metrics(
    path,
    layers,
    passable=None,
    runtime=0.0,
    include_task_reward=True,
    score_weights=EVALUATION_SCORE_WEIGHTS,
):
    """Calculate path metrics plus task reward, exposure risk, and final score."""
    if not path:
        return {
            "distance": None,
            "time": None,
            "safety": None,
            "collision": None,
            "smoothness": None,
            "task_reward": None,
            "rain_exposure": None,
            "inflation_exposure": None,
            "total_exposure": None,
            "score": None,
        }

    distance, travel_time, safety = calculate_path_metrics(path, layers)
    smoothness = calculate_smoothness(path)
    collision = calculate_collision_count(path, passable) if passable is not None else 0
    task_reward = calculate_task_reward(path, layers) if include_task_reward else None
    exposure_metrics = calculate_exposure_metrics(path, layers)
    map_size = layers["O"].shape[0] if is_layer_map(layers) else speed_grid(layers).shape[0]
    score = calculate_evaluation_score(
        distance=distance,
        travel_time=travel_time,
        safety=safety,
        collision=collision,
        smoothness=smoothness,
        runtime=runtime,
        task_reward=task_reward if include_task_reward else 0.0,
        exposure=exposure_metrics["total_exposure"],
        map_size=map_size,
        weights=score_weights,
    )
    return {
        "distance": distance,
        "time": travel_time,
        "safety": safety,
        "collision": collision,
        "smoothness": smoothness,
        "task_reward": task_reward,
        "rain_exposure": exposure_metrics["rain_exposure"],
        "inflation_exposure": exposure_metrics["inflation_exposure"],
        "total_exposure": exposure_metrics["total_exposure"],
        "score": score,
    }


def calculate_layer_cost_metrics(path, layers, weights=MODEL_WEIGHTS):
    """
    计算新分层模型指标及其加权代价。

    权重只作用在四个环境模型指标上：
        障碍膨胀、风场阻力、降雨风险、巡航价值损失。
    """
    if not path or not is_layer_map(layers):
        return {
            "inflation": 0.0,
            "wind": 0.0,
            "rain": 0.0,
            "cruise": 0.0,
            "weighted_cost": 0.0,
        }

    is_grid_path = all(float(point[0]).is_integer() and float(point[1]).is_integer() for point in path)
    points = list(path) if is_grid_path else sample_path(path, step=0.5)
    size = layers["O"].shape[0]

    totals = {
        "inflation": 0.0,
        "wind": 0.0,
        "rain": 0.0,
        "cruise": 0.0,
    }
    count = 0
    prev = None

    for point in points:
        row = int(round(float(point[0])))
        col = int(round(float(point[1])))
        if not (0 <= row < size and 0 <= col < size):
            prev = point
            continue

        q = (row, col)
        h_hat = (0.0, 0.0) if prev is None else heading_vector(prev, point)
        totals["inflation"] += C_inf(q, layers)
        totals["wind"] += C_wind(q, h_hat, layers)
        totals["rain"] += C_rain(q, layers)
        totals["cruise"] += C_cruise(q, layers)
        count += 1
        prev = point

    if count == 0:
        return {
            "inflation": 0.0,
            "wind": 0.0,
            "rain": 0.0,
            "cruise": 0.0,
            "weighted_cost": 0.0,
        }

    averaged = {name: value / count for name, value in totals.items()}
    w_inf, w_wind, w_rain, w_cruise = weights
    averaged["weighted_cost"] = (
        w_inf * averaged["inflation"]
        + w_wind * averaged["wind"]
        + w_rain * averaged["rain"]
        + w_cruise * averaged["cruise"]
    )
    return averaged


def weighted_score(layer_metrics, weights=MODEL_WEIGHTS):
    """按新分层模型指标计算加权代价，值越小越优。"""
    w_inf, w_wind, w_rain, w_cruise = weights
    return (
        w_inf * layer_metrics["inflation"]
        + w_wind * layer_metrics["wind"]
        + w_rain * layer_metrics["rain"]
        + w_cruise * layer_metrics["cruise"]
    )
