"""Shared data structures for map and path-planning results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MapLayers:
    """Typed view of the layered map while keeping dict compatibility elsewhere."""

    O: np.ndarray
    I: np.ndarray
    W: np.ndarray
    R: np.ndarray
    V: np.ndarray
    tasks: list[tuple[int, int]]
    speed_map: np.ndarray

    @classmethod
    def from_mapping(cls, layers: dict[str, Any]) -> "MapLayers":
        return cls(
            O=layers["O"],
            I=layers["I"],
            W=layers["W"],
            R=layers["R"],
            V=layers["V"],
            tasks=list(layers.get("tasks", [])),
            speed_map=layers["speed_map"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "O": self.O,
            "I": self.I,
            "W": self.W,
            "R": self.R,
            "V": self.V,
            "tasks": self.tasks,
            "speed_map": self.speed_map,
        }


@dataclass(frozen=True)
class PathMetrics:
    distance: float | None
    time: float | None
    safety: float | None
    collision: int | None
    smoothness: float | None
    runtime: float
    success_rate: float
    task_reward: float | None = None
    rain_exposure: float | None = None
    inflation_exposure: float | None = None
    total_exposure: float | None = None
    score: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "distance": self.distance,
            "time": self.time,
            "safety": self.safety,
            "collision": self.collision,
            "smoothness": self.smoothness,
            "runtime": self.runtime,
            "success_rate": self.success_rate,
            "task_reward": self.task_reward,
            "rain_exposure": self.rain_exposure,
            "inflation_exposure": self.inflation_exposure,
            "total_exposure": self.total_exposure,
            "score": self.score,
        }
