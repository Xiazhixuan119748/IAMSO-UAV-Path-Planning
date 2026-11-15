"""Path encoding and geometry helpers for metaheuristic planners."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


Point = tuple[float, float]


def decode(vector: np.ndarray, start: Sequence[float], goal: Sequence[float]) -> list[Point]:
    """Decode a flat waypoint vector into start + waypoints + goal."""
    coords = np.asarray(vector, dtype=float).reshape(-1, 2)
    points = [(float(start[0]), float(start[1]))]
    points.extend((float(row), float(col)) for row, col in coords)
    points.append((float(goal[0]), float(goal[1])))
    return points


def encode_from_path(path: Sequence[Sequence[float]], waypoint_count: int) -> np.ndarray:
    """Sample a path into a fixed-size flat waypoint vector."""
    if waypoint_count <= 0:
        return np.array([], dtype=float)
    if not path:
        raise ValueError("path must not be empty")

    pts = np.asarray(path, dtype=float)
    if len(pts) == 1:
        sampled = np.repeat(pts, waypoint_count, axis=0)
        return sampled.reshape(-1)

    cumulative = np.zeros(len(pts), dtype=float)
    deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative[1:] = np.cumsum(deltas)
    total = cumulative[-1]
    if total <= 1e-9:
        sampled = np.repeat(pts[:1], waypoint_count, axis=0)
        return sampled.reshape(-1)

    targets = np.linspace(0.0, total, waypoint_count + 2)[1:-1]
    sampled = []
    for target in targets:
        idx = int(np.searchsorted(cumulative, target, side="right") - 1)
        idx = min(max(idx, 0), len(pts) - 2)
        span = cumulative[idx + 1] - cumulative[idx]
        ratio = 0.0 if span <= 1e-9 else (target - cumulative[idx]) / span
        sampled.append(pts[idx] + ratio * (pts[idx + 1] - pts[idx]))
    return np.asarray(sampled, dtype=float).reshape(-1)


def smooth_path(points: Sequence[Sequence[float]], samples_per_segment: int = 8) -> list[Point]:
    """Smooth a polyline with a Catmull-Rom spline implemented in pure numpy."""
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2 or samples_per_segment <= 1:
        return [(float(r), float(c)) for r, c in pts]

    padded = np.vstack([pts[0], pts, pts[-1]])
    curve: list[Point] = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for step in range(samples_per_segment):
            t = step / samples_per_segment
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            curve.append((float(point[0]), float(point[1])))
    curve.append((float(pts[-1, 0]), float(pts[-1, 1])))
    return curve


def sample_line(a: Sequence[float], b: Sequence[float], step: float = 0.5) -> list[Point]:
    """Discretely sample a line segment for collision and cost checks."""
    ar = np.asarray(a, dtype=float)
    br = np.asarray(b, dtype=float)
    distance = float(np.linalg.norm(br - ar))
    count = max(1, int(math.ceil(distance / max(step, 1e-6))))
    samples = []
    for i in range(count + 1):
        ratio = i / count
        point = ar + ratio * (br - ar)
        samples.append((float(point[0]), float(point[1])))
    return samples


def sample_path(path: Sequence[Sequence[float]], step: float = 0.5) -> list[Point]:
    """Sample every segment in a path without duplicating segment joins."""
    if not path:
        return []
    samples: list[Point] = []
    for a, b in zip(path[:-1], path[1:]):
        segment = sample_line(a, b, step)
        if samples:
            segment = segment[1:]
        samples.extend(segment)
    return samples or [(float(path[0][0]), float(path[0][1]))]


def clip_vector(vector: np.ndarray, size: int) -> np.ndarray:
    """Keep waypoint coordinates inside map bounds."""
    return np.clip(vector, 0.0, float(size - 1))


def polyline_length(path: Sequence[Sequence[float]]) -> float:
    """Return Euclidean length of a polyline."""
    if len(path) <= 1:
        return 0.0
    pts = np.asarray(path, dtype=float)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def as_float_path(path: Iterable[Sequence[float]]) -> list[Point]:
    """Normalize a path-like object to float row/col tuples."""
    return [(float(p[0]), float(p[1])) for p in path]
