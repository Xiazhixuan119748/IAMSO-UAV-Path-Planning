"""Planner cost functions."""

from __future__ import annotations

from uav_planner.config import NO_TASK_MODEL_COST_WEIGHTS, NORMAL_SPEED
from uav_planner.costmap import effective_speed, heading_vector, is_layer_map, point_cost, speed_grid, step_cost, step_length


def distance_cost(x, y, speed_map):
    """Shortest-distance cost."""
    if isinstance(x, tuple) and isinstance(y, tuple):
        return step_length(x, y)
    return 1


def time_cost(x, y, speed_map):
    """Fastest-time cost."""
    if isinstance(x, tuple) and isinstance(y, tuple):
        heading = heading_vector(x, y)
        speed = effective_speed(y, heading, speed_map)
        if speed <= 0:
            return 1e6
        return step_length(x, y) / speed

    speed_map = speed_grid(speed_map)
    if speed_map[x, y] <= 0:
        return 1e6
    return 1 / speed_map[x, y]


def safe_cost(x, y, speed_map):
    """Safety-oriented cost."""
    if isinstance(x, tuple) and isinstance(y, tuple):
        row, col = int(y[0]), int(y[1])
        if is_layer_map(speed_map):
            return 1.0 + float(speed_map["I"][row, col]) + float(speed_map["R"][row, col])
        return 0.1 if speed_map[row, col] == NORMAL_SPEED else 1

    speed_map = speed_grid(speed_map)
    if speed_map[x, y] == NORMAL_SPEED:
        return 0.1
    return 1


def model_cost(x, y, layers):
    """Layered model cost."""
    if isinstance(x, tuple) and isinstance(y, tuple) and is_layer_map(layers):
        return step_cost(x, y, layers)
    if is_layer_map(layers):
        return point_cost(x, y, layers)
    return time_cost(x, y, layers)


def no_task_model_cost(x, y, layers):
    """Layered model cost with cruise/task value disabled."""
    if isinstance(x, tuple) and isinstance(y, tuple) and is_layer_map(layers):
        return step_cost(x, y, layers, NO_TASK_MODEL_COST_WEIGHTS)
    if is_layer_map(layers):
        return point_cost(x, y, layers, NO_TASK_MODEL_COST_WEIGHTS)
    return time_cost(x, y, layers)


def make_weighted_cost(w_inf, w_wind, w_rain, w_cruise):
    """Build a layered weighted cost function."""
    weights = (w_inf, w_wind, w_rain, w_cruise)

    def weighted_cost(x, y, speed_map):
        if is_layer_map(speed_map):
            if isinstance(x, tuple) and isinstance(y, tuple):
                return step_cost(x, y, speed_map, weights)
            return point_cost(x, y, speed_map, weights)
        return time_cost(x, y, speed_map)

    return weighted_cost
