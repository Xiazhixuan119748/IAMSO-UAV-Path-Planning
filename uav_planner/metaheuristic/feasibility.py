"""Feasibility-first comparison and elite archive for path candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


Evaluation = Mapping[str, Any]


def is_feasible(evaluation: Evaluation) -> bool:
    """Return whether an evaluated path is free of cell and segment collisions."""
    return (
        int(evaluation.get("segment_collision_count", 0)) == 0
        and int(evaluation.get("collision_count", 0)) == 0
    )


def feasibility_key(evaluation: Evaluation) -> tuple[int, int, int, float, float]:
    """Return the strict ordering key used by IAMSO candidate selection."""
    feasible_rank = 0 if is_feasible(evaluation) else 1
    return (
        feasible_rank,
        int(evaluation.get("segment_collision_count", 0)),
        int(evaluation.get("collision_count", 0)),
        float(evaluation["value"]),
        float(evaluation.get("smoothness", 0.0)),
    )


@dataclass
class FeasibleArchive:
    """Keep the best collision-free path independently of population updates."""

    evaluation: dict[str, Any] | None = None
    vector: np.ndarray | None = None
    source: str | None = None

    def consider(
        self,
        evaluation: Evaluation,
        *,
        vector: np.ndarray | None = None,
        source: str,
    ) -> bool:
        if not is_feasible(evaluation):
            return False
        if self.evaluation is not None and feasibility_key(evaluation) >= feasibility_key(self.evaluation):
            return False

        self.evaluation = dict(evaluation)
        self.vector = None if vector is None else np.asarray(vector, dtype=float).copy()
        self.source = source
        return True
