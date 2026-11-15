"""Lightweight plotting helpers for UAV map and path visualization."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def path_xy_points(path: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert (row, col) path points to plotting coordinates (x=col, y=row)."""
    if not path:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    points = np.asarray([(float(point[1]), float(point[0])) for point in path], dtype=float)
    return points[:, 0], points[:, 1]


def compute_path_headings(path: Sequence[Sequence[float]]) -> np.ndarray:
    """Compute a heading angle for each point on the path."""
    x, y = path_xy_points(path)
    if len(x) == 0:
        return np.empty(0, dtype=float)

    headings = np.zeros(len(x), dtype=float)
    if len(x) == 1:
        return headings

    for index in range(len(x)):
        if index == 0:
            dx = x[1] - x[0]
            dy = y[1] - y[0]
        elif index == len(x) - 1:
            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]
        else:
            dx = x[index + 1] - x[index - 1]
            dy = y[index + 1] - y[index - 1]
        headings[index] = math.atan2(dy, dx) if abs(dx) > 1e-12 or abs(dy) > 1e-12 else 0.0

    return headings


def compute_path_curvature(path: Sequence[Sequence[float]]) -> np.ndarray:
    """Compute signed discrete curvature for each point on the path."""
    x, y = path_xy_points(path)
    curvature = np.zeros(len(x), dtype=float)
    if len(x) < 3:
        return curvature

    for index in range(1, len(x) - 1):
        prev_point = np.array([x[index - 1], y[index - 1]], dtype=float)
        point = np.array([x[index], y[index]], dtype=float)
        next_point = np.array([x[index + 1], y[index + 1]], dtype=float)

        v1 = point - prev_point
        v2 = next_point - point
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-12 or n2 <= 1e-12:
            continue

        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        sin_theta = max(-1.0, min(1.0, cross / (n1 * n2)))
        curvature[index] = 2.0 * sin_theta / max(n1 + n2, 1e-12)

    return curvature


def plot_arrow(
    ax,
    x,
    y,
    yaw,
    arrow_length=1.0,
    origin_point_plot_style=None,
    head_width=0.16,
    fc="r",
    ec="k",
    alpha=0.8,
    linewidth=0.8,
    zorder=6,
):
    """Plot one or more 2D arrows."""
    if np.isscalar(x):
        dx = arrow_length * math.cos(float(yaw))
        dy = arrow_length * math.sin(float(yaw))
        ax.arrow(
            float(x),
            float(y),
            dx,
            dy,
            head_width=head_width,
            length_includes_head=True,
            fc=fc,
            ec=ec,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
        )
        if origin_point_plot_style is not None:
            ax.plot(float(x), float(y), origin_point_plot_style, zorder=zorder)
        return

    for one_x, one_y, one_yaw in zip(x, y, yaw):
        plot_arrow(
            ax,
            one_x,
            one_y,
            one_yaw,
            arrow_length=arrow_length,
            origin_point_plot_style=origin_point_plot_style,
            head_width=head_width,
            fc=fc,
            ec=ec,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
        )


def plot_curvature(
    ax,
    x_list,
    y_list,
    heading_list,
    curvature,
    k=0.45,
    color="#00bcd4",
    label=None,
    alpha=0.35,
    linewidth=1.1,
    connector_stride=4,
    zorder=5,
):
    """Plot a curvature offset line beside the original path."""
    if len(x_list) == 0:
        return None

    offset_x = [
        x + value * k * math.cos(yaw - math.pi / 2.0)
        for x, yaw, value in zip(x_list, heading_list, curvature)
    ]
    offset_y = [
        y + value * k * math.sin(yaw - math.pi / 2.0)
        for y, yaw, value in zip(y_list, heading_list, curvature)
    ]

    line_kwargs = {
        "color": color,
        "linestyle": "--",
        "linewidth": linewidth,
        "alpha": alpha,
        "zorder": zorder,
    }
    if label:
        line_kwargs["label"] = label
    line = ax.plot(offset_x, offset_y, **line_kwargs)[0]

    stride = max(1, int(connector_stride))
    for index, (x0, y0, x1, y1) in enumerate(zip(x_list, y_list, offset_x, offset_y)):
        if index % stride != 0:
            continue
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            linewidth=max(0.5, linewidth * 0.55),
            alpha=min(0.6, alpha + 0.1),
            zorder=zorder,
        )

    return line


def plot_path_heading_arrows(ax, path, color, max_arrows=7, arrow_length=1.05, zorder=8):
    """Plot sparse heading arrows along a path."""
    x, y = path_xy_points(path)
    if len(x) < 2:
        return

    headings = compute_path_headings(path)
    count = min(max_arrows, len(x))
    indices = np.unique(np.linspace(0, len(x) - 1, count, dtype=int))
    plot_arrow(
        ax,
        x[indices],
        y[indices],
        headings[indices],
        arrow_length=arrow_length,
        origin_point_plot_style=None,
        head_width=max(0.14, arrow_length * 0.22),
        fc=color,
        ec="black",
        alpha=0.75,
        linewidth=0.7,
        zorder=zorder,
    )


def plot_path_curvature_overlay(ax, path, color, scale=0.55, zorder=5):
    """Plot a low-noise curvature overlay for a path."""
    if not path or len(path) < 3:
        return None

    x, y = path_xy_points(path)
    headings = compute_path_headings(path)
    curvature = compute_path_curvature(path)
    if not np.any(np.abs(curvature) > 1e-9):
        return None

    return plot_curvature(
        ax,
        x,
        y,
        headings,
        curvature,
        k=scale,
        color=color,
        label=None,
        alpha=0.22,
        linewidth=1.0,
        connector_stride=max(3, len(path) // 18),
        zorder=zorder,
    )


def plot_sparse_wind_arrows(ax, wind_field, stride, color="#263238", alpha=0.6, zorder=3):
    """Plot a sparse wind-direction arrow field."""
    stride = max(1, int(stride))
    size = wind_field.shape[0]
    arrow_length = max(0.75, min(1.6, stride * 0.18))

    for row in range(0, size, stride):
        for col in range(0, size, stride):
            dr = float(wind_field[row, col, 0])
            dc = float(wind_field[row, col, 1])
            if math.hypot(dr, dc) <= 1e-9:
                continue
            plot_arrow(
                ax,
                col,
                row,
                math.atan2(dr, dc),
                arrow_length=arrow_length,
                origin_point_plot_style=None,
                head_width=max(0.12, arrow_length * 0.20),
                fc=color,
                ec=color,
                alpha=alpha,
                linewidth=0.6,
                zorder=zorder,
            )
