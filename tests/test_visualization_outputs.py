"""Regression tests for grouped single-run visualization outputs."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from uav_planner.mode_runners.common import plot_paths
from uav_planner.output_paths import OUTPUTS_ROOT, create_run_output_paths
from uav_planner.visualization import animate_paths


def _sample_map(size=6):
    passable = np.ones((size, size), dtype=int)
    passable[2, 3] = 0
    obstacles = 1 - passable
    layers = {
        "O": obstacles,
        "I": obstacles.astype(float),
        "W": np.zeros((size, size, 2), dtype=float),
        "R": np.linspace(0.0, 1.0, size * size, dtype=float).reshape(size, size),
        "V": np.flipud(np.eye(size, dtype=float)),
        "tasks": [(1, 4)],
        "speed_map": np.ones((size, size), dtype=float),
    }
    return passable, layers


class VisualizationOutputTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_run_directory_and_artifact_names_are_windows_safe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_paths = create_run_output_paths(
                "A*",
                "简单",
                timestamp="20260710_235959",
                outputs_root=temp_dir,
            )

            self.assertEqual(output_paths.directory.name, "AStar+简单+20260710_235959")
            self.assertEqual(output_paths.static_result.name, "uav_path_result_20260710_235959.png")
            self.assertEqual(output_paths.map_layers.name, "uav_map_layers_20260710_235959.png")
            self.assertEqual(output_paths.animation.name, "uav_path_animation_20260710_235959.gif")

            duplicate_paths = create_run_output_paths(
                "A*",
                "简单",
                timestamp="20260710_235959",
                outputs_root=temp_dir,
            )
            self.assertEqual(duplicate_paths.directory.name, "AStar+简单+20260710_235959_2")

        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(OUTPUTS_ROOT, project_root / "outputs")
        self.assertTrue(OUTPUTS_ROOT.is_absolute())

    def test_static_layers_and_animation_share_one_directory(self):
        passable, layers = _sample_map()
        path = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 4), (5, 5)]
        result = {
            "path": path,
            "label": "A*+最短距离",
            "collision_points": [],
            "distance": 7.0,
            "time": 8.0,
            "safety": 0.9,
            "collision": 0,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_paths = create_run_output_paths(
                "A*",
                "简单",
                timestamp="20260710_235959",
                outputs_root=temp_dir,
            )
            with patch("uav_planner.visualization.plt.show"):
                plot_paths(
                    passable,
                    layers,
                    [result],
                    (0, 0),
                    (5, 5),
                    animate=False,
                    save_gif=False,
                    save_png=True,
                    output_paths=output_paths,
                )
            animate_paths(
                passable,
                layers,
                [(path, result["label"], "blue", [])],
                (0, 0),
                (5, 5),
                interval=100,
                max_frames=4,
                output_paths=output_paths,
            )

            for artifact in (
                output_paths.static_result,
                output_paths.map_layers,
                output_paths.animation,
            ):
                self.assertEqual(artifact.parent, output_paths.directory)
                self.assertTrue(artifact.is_file(), artifact)
                self.assertGreater(artifact.stat().st_size, 0)

    def test_failed_planning_still_saves_static_map_outputs(self):
        passable, layers = _sample_map()
        failed_result = {"path": None, "label": "DFS+最短距离"}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_paths = create_run_output_paths(
                "DFS",
                "困难",
                timestamp="20260710_235959",
                outputs_root=temp_dir,
            )
            with patch("uav_planner.visualization.plt.show"):
                plot_paths(
                    passable,
                    layers,
                    [failed_result],
                    (0, 0),
                    (5, 5),
                    animate=True,
                    save_gif=True,
                    save_png=True,
                    output_paths=output_paths,
                )

            self.assertTrue(output_paths.static_result.is_file())
            self.assertTrue(output_paths.map_layers.is_file())
            self.assertFalse(output_paths.animation.exists())


if __name__ == "__main__":
    unittest.main()
