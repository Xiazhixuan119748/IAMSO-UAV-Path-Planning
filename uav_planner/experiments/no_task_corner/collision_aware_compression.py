"""Compatibility exports for collision-aware compression ablations."""

from uav_planner.metaheuristic.collision_aware import (
    collision_aware_encode,
    collision_aware_initial_population,
    line_of_sight_simplify,
)

__all__ = [
    "line_of_sight_simplify",
    "collision_aware_encode",
    "collision_aware_initial_population",
]
