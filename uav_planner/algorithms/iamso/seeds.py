"""Collision-safe seed planners used by the production IAMSO planner."""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Callable, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from uav_planner.algorithms.common import edge_cost, iter_moves, reconstruct_path, seed_value
from uav_planner.costmap import octile_distance
from uav_planner.geometry import collision_cells_for_path, segment_has_collision


GridPoint = tuple[int, int]
Path = list[tuple[float, float]]


def _neighbors(passable: np.ndarray, current: GridPoint):
    size = int(passable.shape[0])
    for dr, dc, _ in iter_moves():
        neighbor = current[0] + dr, current[1] + dc
        if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size):
            continue
        if not passable[neighbor] or segment_has_collision(passable, current, neighbor):
            continue
        yield neighbor


def _grid_search(
    passable: np.ndarray,
    layers,
    cost_func,
    start: GridPoint,
    goal: GridPoint,
    heuristic: Callable[[GridPoint, int], float],
) -> Path | None:
    queue = [(heuristic(start, 0), 0.0, 0, start)]
    costs = {start: 0.0}
    depths = {start: 0}
    parent: dict[GridPoint, GridPoint] = {}

    while queue:
        _, current_cost, depth, current = heapq.heappop(queue)
        if current_cost > costs.get(current, math.inf):
            continue
        if current == goal:
            path = reconstruct_path(parent, start, goal)
            return [(float(row), float(col)) for row, col in path] if path else None

        for neighbor in _neighbors(passable, current):
            candidate_cost = current_cost + max(edge_cost(cost_func, current, neighbor, layers), 1e-9)
            candidate_depth = depth + 1
            if candidate_cost >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = candidate_cost
            depths[neighbor] = candidate_depth
            parent[neighbor] = current
            priority = candidate_cost + heuristic(neighbor, candidate_depth)
            heapq.heappush(queue, (priority, candidate_cost, candidate_depth, neighbor))
    return None


def astar_seed(passable, layers, cost_func, start, goal) -> Path | None:
    return _grid_search(passable, layers, cost_func, start, goal, lambda node, _depth: octile_distance(node, goal))


def distance_field_astar_seed(passable, layers, cost_func, start, goal) -> Path | None:
    """A* guided by the goal heuristic plus an obstacle-distance field."""
    clearance = distance_transform_edt(np.asarray(passable, dtype=bool))

    def heuristic(node: GridPoint, _depth: int) -> float:
        # The field only guides expansion; g still uses the active experiment cost.
        proximity = 2.0 / max(float(clearance[node]), 0.5)
        return octile_distance(node, goal) + proximity

    return _grid_search(passable, layers, cost_func, start, goal, heuristic)


def dynamic_weighted_astar_seed(passable, layers, cost_func, start, goal, epsilon: float) -> Path | None:
    """Dynamic-weight A* using the decay rule from PythonRobotics A* variants."""
    upper_depth = max(1, int(passable.shape[0]) * 2)

    def heuristic(node: GridPoint, depth: int) -> float:
        ratio = min(1.0, depth / upper_depth)
        weight = 1.0 + epsilon - epsilon * ratio
        return weight * octile_distance(node, goal)

    return _grid_search(passable, layers, cost_func, start, goal, heuristic)


def _segment_cost(a: GridPoint, b: GridPoint, layers, cost_func) -> float:
    distance = math.dist(a, b)
    steps = max(1, int(math.ceil(distance * 2.0)))
    previous = a
    total = 0.0
    for index in range(1, steps + 1):
        ratio = index / steps
        point = (
            int(round(a[0] + (b[0] - a[0]) * ratio)),
            int(round(a[1] + (b[1] - a[1]) * ratio)),
        )
        if point == previous:
            continue
        total += max(edge_cost(cost_func, previous, point, layers), 1e-9)
        previous = point
    return total


def theta_star_seed(passable, layers, cost_func, start, goal) -> Path | None:
    """Theta* with line-of-sight parent relaxation and the active cost function."""
    queue = [(octile_distance(start, goal), 0.0, start)]
    costs = {start: 0.0}
    parent: dict[GridPoint, GridPoint] = {start: start}

    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current_cost > costs.get(current, math.inf):
            continue
        if current == goal:
            break

        for neighbor in _neighbors(passable, current):
            source = current
            current_parent = parent[current]
            if current_parent != current and not segment_has_collision(passable, current_parent, neighbor):
                source = current_parent
            candidate_cost = costs[source] + _segment_cost(source, neighbor, layers, cost_func)
            if candidate_cost >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = candidate_cost
            parent[neighbor] = source
            heapq.heappush(
                queue,
                (candidate_cost + octile_distance(neighbor, goal), candidate_cost, neighbor),
            )

    if goal not in parent:
        return None
    path: list[GridPoint] = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return [(float(row), float(col)) for row, col in path]


def dijkstra_seed(passable, layers, cost_func, start, goal) -> Path | None:
    return _grid_search(passable, layers, cost_func, start, goal, lambda _node, _depth: 0.0)


def _reconstruct_bidirectional(
    parent_forward: dict[GridPoint, GridPoint],
    parent_backward: dict[GridPoint, GridPoint],
    start: GridPoint,
    goal: GridPoint,
    meet: GridPoint,
) -> Path | None:
    forward: list[GridPoint] = [meet]
    while forward[-1] != start:
        if forward[-1] not in parent_forward:
            return None
        forward.append(parent_forward[forward[-1]])
    forward.reverse()

    backward: list[GridPoint] = []
    current = meet
    while current != goal:
        if current not in parent_backward:
            return None
        current = parent_backward[current]
        backward.append(current)
    return [(float(row), float(col)) for row, col in forward + backward]


def _bidirectional_grid_search(
    passable: np.ndarray,
    layers,
    cost_func,
    start: GridPoint,
    goal: GridPoint,
    heuristic_forward: Callable[[GridPoint, int], float],
    heuristic_backward: Callable[[GridPoint, int], float],
    *,
    theta: bool = False,
) -> Path | None:
    """Expand two cost-aware frontiers until they connect."""
    if start == goal:
        return [(float(start[0]), float(start[1]))]

    forward_queue = [(heuristic_forward(start, 0), 0.0, 0, start)]
    backward_queue = [(heuristic_backward(goal, 0), 0.0, 0, goal)]
    forward_cost = {start: 0.0}
    backward_cost = {goal: 0.0}
    forward_depth = {start: 0}
    backward_depth = {goal: 0}
    parent_forward: dict[GridPoint, GridPoint] = {start: start}
    parent_backward: dict[GridPoint, GridPoint] = {goal: goal}

    while forward_queue and backward_queue:
        expand_forward = forward_queue[0][0] <= backward_queue[0][0]
        queue = forward_queue if expand_forward else backward_queue
        _, current_cost, depth, current = heapq.heappop(queue)
        costs = forward_cost if expand_forward else backward_cost
        if current_cost > costs.get(current, math.inf):
            continue

        if current in forward_cost and current in backward_cost:
            meet = _reconstruct_bidirectional(
                parent_forward, parent_backward, start, goal, current
            )
            if meet:
                return meet

        for neighbor in _neighbors(passable, current):
            if expand_forward:
                source = current
                if theta and parent_forward[current] != current:
                    candidate_source = parent_forward[current]
                    if not segment_has_collision(passable, candidate_source, neighbor):
                        source = candidate_source
                step = _segment_cost(source, neighbor, layers, cost_func)
                candidate_cost = forward_cost[source] + step
                if candidate_cost >= forward_cost.get(neighbor, math.inf):
                    continue
                forward_cost[neighbor] = candidate_cost
                forward_depth[neighbor] = depth + 1
                parent_forward[neighbor] = source
                heapq.heappush(
                    forward_queue,
                    (candidate_cost + heuristic_forward(neighbor, depth + 1), candidate_cost, depth + 1, neighbor),
                )
            else:
                # Backward g is the forward cost from a predecessor to the goal.
                source_next = current
                if theta and parent_backward[current] != current:
                    candidate_next = parent_backward[current]
                    if not segment_has_collision(passable, candidate_next, neighbor):
                        source_next = candidate_next
                step = _segment_cost(neighbor, source_next, layers, cost_func)
                candidate_cost = backward_cost[source_next] + step
                if candidate_cost >= backward_cost.get(neighbor, math.inf):
                    continue
                backward_cost[neighbor] = candidate_cost
                backward_depth[neighbor] = depth + 1
                parent_backward[neighbor] = source_next
                heapq.heappush(
                    backward_queue,
                    (candidate_cost + heuristic_backward(neighbor, depth + 1), candidate_cost, depth + 1, neighbor),
                )

            if neighbor in forward_cost and neighbor in backward_cost:
                meet = _reconstruct_bidirectional(
                    parent_forward, parent_backward, start, goal, neighbor
                )
                if meet:
                    return meet
    return None


def bidirectional_astar_seed(passable, layers, cost_func, start, goal) -> Path | None:
    return _bidirectional_grid_search(
        passable, layers, cost_func, start, goal,
        lambda node, _depth: octile_distance(node, goal),
        lambda node, _depth: octile_distance(node, start),
    )


def bidirectional_distance_field_astar_seed(passable, layers, cost_func, start, goal) -> Path | None:
    clearance = distance_transform_edt(np.asarray(passable, dtype=bool))

    def heuristic(node: GridPoint, target: GridPoint) -> float:
        return octile_distance(node, target) + 2.0 / max(float(clearance[node]), 0.5)

    return _bidirectional_grid_search(
        passable, layers, cost_func, start, goal,
        lambda node, _depth: heuristic(node, goal),
        lambda node, _depth: heuristic(node, start),
    )


def bidirectional_theta_star_seed(passable, layers, cost_func, start, goal) -> Path | None:
    return _bidirectional_grid_search(
        passable, layers, cost_func, start, goal,
        lambda node, _depth: octile_distance(node, goal),
        lambda node, _depth: octile_distance(node, start),
        theta=True,
    )


def bidirectional_dijkstra_seed(passable, layers, cost_func, start, goal) -> Path | None:
    return _bidirectional_grid_search(
        passable, layers, cost_func, start, goal,
        lambda _node, _depth: 0.0,
        lambda _node, _depth: 0.0,
    )


def bidirectional_dynamic_weighted_astar_seed(
    passable, layers, cost_func, start, goal, epsilon: float
) -> Path | None:
    upper_depth = max(1, int(passable.shape[0]) * 2)

    def heuristic(node: GridPoint, target: GridPoint, depth: int) -> float:
        ratio = min(1.0, depth / upper_depth)
        weight = 1.0 + epsilon - epsilon * ratio
        return weight * octile_distance(node, target)

    return _bidirectional_grid_search(
        passable, layers, cost_func, start, goal,
        lambda node, depth: heuristic(node, goal, depth),
        lambda node, depth: heuristic(node, start, depth),
    )


def _steer(a: Sequence[float], b: Sequence[float], step_size: float) -> tuple[float, float]:
    distance = math.dist(a, b)
    if distance <= step_size:
        return float(b[0]), float(b[1])
    ratio = step_size / max(distance, 1e-9)
    return float(a[0] + (b[0] - a[0]) * ratio), float(a[1] + (b[1] - a[1]) * ratio)


def safe_rrt_seed(passable, start, goal, *, star: bool, random_offset: int) -> Path | None:
    """RRT/RRT* seed with exact segment collision checks for every edge."""
    size = int(passable.shape[0])
    rng = random.Random(seed_value(start, goal, random_offset))
    start_f = float(start[0]), float(start[1])
    goal_f = float(goal[0]), float(goal[1])
    nodes = [start_f]
    parents: list[int | None] = [None]
    costs = [0.0]
    step_size = max(2.0, size / 18.0)
    max_iterations = max(1400, size * size // 2)

    for _ in range(max_iterations):
        sample = goal_f if rng.random() < 0.20 else (rng.uniform(0, size - 1), rng.uniform(0, size - 1))
        nearest = min(range(len(nodes)), key=lambda index: math.dist(nodes[index], sample))
        new_node = _steer(nodes[nearest], sample, step_size)
        if segment_has_collision(passable, nodes[nearest], new_node):
            continue

        parent_index = nearest
        new_cost = costs[nearest] + math.dist(nodes[nearest], new_node)
        near_indices: list[int] = []
        if star:
            radius = step_size * 2.5
            near_indices = [index for index, node in enumerate(nodes) if math.dist(node, new_node) <= radius]
            for index in near_indices:
                candidate_cost = costs[index] + math.dist(nodes[index], new_node)
                if candidate_cost < new_cost and not segment_has_collision(passable, nodes[index], new_node):
                    parent_index = index
                    new_cost = candidate_cost

        nodes.append(new_node)
        parents.append(parent_index)
        costs.append(new_cost)
        new_index = len(nodes) - 1

        if star:
            for index in near_indices:
                rewired_cost = new_cost + math.dist(new_node, nodes[index])
                if rewired_cost < costs[index] and not segment_has_collision(passable, new_node, nodes[index]):
                    parents[index] = new_index
                    costs[index] = rewired_cost

        if not star and math.dist(new_node, goal_f) <= step_size * 1.5:
            if not segment_has_collision(passable, new_node, goal_f):
                nodes.append(goal_f)
                parents.append(new_index)
                return _reconstruct_tree_path(nodes, parents, len(nodes) - 1)

    visible = [index for index, node in enumerate(nodes) if not segment_has_collision(passable, node, goal_f)]
    if not visible:
        return None
    best = min(visible, key=lambda index: costs[index] + math.dist(nodes[index], goal_f))
    nodes.append(goal_f)
    parents.append(best)
    return _reconstruct_tree_path(nodes, parents, len(nodes) - 1)


def _reconstruct_tree_path(nodes, parents, index: int) -> Path:
    path = []
    while index is not None:
        path.append(nodes[index])
        index = parents[index]
    path.reverse()
    return [(float(row), float(col)) for row, col in path]


def _is_safe(passable: np.ndarray, path) -> bool:
    return bool(path) and not collision_cells_for_path(passable, path)


def build_seed_paths(passable, layers, cost_func, start, goal) -> tuple[list[Path], list[str]]:
    """Build four primary and up to four diversity seed slots."""
    candidates = [
        ("astar_current", lambda: astar_seed(passable, layers, cost_func, start, goal)),
        ("distance_field_astar_current", lambda: distance_field_astar_seed(passable, layers, cost_func, start, goal)),
        ("theta_star_current", lambda: theta_star_seed(passable, layers, cost_func, start, goal)),
        ("dijkstra_current", lambda: dijkstra_seed(passable, layers, cost_func, start, goal)),
        ("dynamic_weighted_astar_e1", lambda: dynamic_weighted_astar_seed(passable, layers, cost_func, start, goal, 1.0)),
        ("dynamic_weighted_astar_e2", lambda: dynamic_weighted_astar_seed(passable, layers, cost_func, start, goal, 2.0)),
        ("safe_rrt", lambda: safe_rrt_seed(passable, start, goal, star=False, random_offset=31)),
        ("safe_rrt_star", lambda: safe_rrt_seed(passable, start, goal, star=True, random_offset=37)),
    ]
    paths: list[Path] = []
    labels: list[str] = []
    for label, generate in candidates:
        try:
            path = generate()
        except Exception:
            path = None
        if not _is_safe(passable, path):
            continue
        paths.append(path)
        labels.append(label)
    return paths, labels


def build_bidirectional_seed_paths(passable, layers, cost_func, start, goal) -> tuple[list[Path], list[str]]:
    """Use bidirectional variants for all compatible grid seeds."""
    candidates = [
        ("bidirectional_astar_current", lambda: bidirectional_astar_seed(passable, layers, cost_func, start, goal)),
        ("bidirectional_distance_field_astar_current", lambda: bidirectional_distance_field_astar_seed(passable, layers, cost_func, start, goal)),
        ("bidirectional_theta_star_current", lambda: bidirectional_theta_star_seed(passable, layers, cost_func, start, goal)),
        ("bidirectional_dijkstra_current", lambda: bidirectional_dijkstra_seed(passable, layers, cost_func, start, goal)),
        ("bidirectional_dynamic_weighted_astar_e1", lambda: bidirectional_dynamic_weighted_astar_seed(passable, layers, cost_func, start, goal, 1.0)),
        ("bidirectional_dynamic_weighted_astar_e2", lambda: bidirectional_dynamic_weighted_astar_seed(passable, layers, cost_func, start, goal, 2.0)),
        ("safe_rrt", lambda: safe_rrt_seed(passable, start, goal, star=False, random_offset=31)),
        ("safe_rrt_star", lambda: safe_rrt_seed(passable, start, goal, star=True, random_offset=37)),
    ]
    paths: list[Path] = []
    labels: list[str] = []
    for label, generate in candidates:
        try:
            path = generate()
        except Exception:
            path = None
        if _is_safe(passable, path):
            paths.append(path)
            labels.append(label)
    return paths, labels


def build_bidirectional_grid_only_seed_paths(
    passable, layers, cost_func, start, goal
) -> tuple[list[Path], list[str]]:
    """Build the bidirectional grid seeds without RRT/RRT* diversity seeds."""
    candidates = [
        ("bidirectional_astar_current", lambda: bidirectional_astar_seed(passable, layers, cost_func, start, goal)),
        ("bidirectional_distance_field_astar_current", lambda: bidirectional_distance_field_astar_seed(passable, layers, cost_func, start, goal)),
        ("bidirectional_theta_star_current", lambda: bidirectional_theta_star_seed(passable, layers, cost_func, start, goal)),
        ("bidirectional_dijkstra_current", lambda: bidirectional_dijkstra_seed(passable, layers, cost_func, start, goal)),
        ("bidirectional_dynamic_weighted_astar_e1", lambda: bidirectional_dynamic_weighted_astar_seed(passable, layers, cost_func, start, goal, 1.0)),
        ("bidirectional_dynamic_weighted_astar_e2", lambda: bidirectional_dynamic_weighted_astar_seed(passable, layers, cost_func, start, goal, 2.0)),
    ]
    paths: list[Path] = []
    labels: list[str] = []
    for label, generate in candidates:
        try:
            path = generate()
        except Exception:
            path = None
        if _is_safe(passable, path):
            paths.append(path)
            labels.append(label)
    return paths, labels


def build_bidirectional_with_rrt_seed_paths(
    passable, layers, cost_func, start, goal
) -> tuple[list[Path], list[str]]:
    """Build bidirectional grid seeds plus safe RRT, excluding RRT*."""
    paths, labels = build_bidirectional_grid_only_seed_paths(
        passable, layers, cost_func, start, goal
    )
    try:
        rrt_path = safe_rrt_seed(
            passable, start, goal, star=False, random_offset=31
        )
    except Exception:
        rrt_path = None
    if _is_safe(passable, rrt_path):
        paths.append(rrt_path)
        labels.append("safe_rrt")
    return paths, labels
