import unittest
from unittest.mock import patch

import numpy as np

from uav_planner.geometry import collision_cells_for_path
from uav_planner.experiments.no_task_corner.run_iamso_seed_ablation import (
    NO_TASK_LAYER_WEIGHTS,
    iamso_seed_ablation_planner,
)
from uav_planner.experiments.no_task_corner.run_iamso_bidirectional_ablation import (
    iamso_bidirectional_no_final_smoothing_planner,
)
from uav_planner.algorithms.iamso.seeds import (
    build_bidirectional_seed_paths,
    build_bidirectional_grid_only_seed_paths,
    build_bidirectional_with_rrt_seed_paths,
    build_seed_paths,
)


class IAMSOSeedAblationTests(unittest.TestCase):
    @patch("uav_planner.experiments.no_task_corner.run_iamso_bidirectional_ablation.iamso_optimize")
    @patch("uav_planner.experiments.no_task_corner.run_iamso_bidirectional_ablation.build_bidirectional_seed_paths")
    def test_no_final_smoothing_returns_optimizer_path_unchanged(self, build_seeds, optimize):
        validated_path = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        build_seeds.return_value = ([validated_path], ["bidirectional_astar_current"])
        optimize.return_value = {
            "path": validated_path,
            "solution_source": "optimized",
            "feasible": True,
        }

        result = iamso_bidirectional_no_final_smoothing_planner(
            np.ones((2, 2), dtype=bool),
            np.ones((2, 2), dtype=float),
            lambda _a, _b, _layers: 1.0,
            (0, 0),
            (1, 1),
        )

        self.assertEqual(list(result), validated_path)

    def test_seed_set_is_safe_and_contains_primary_and_diversity_sources(self):
        passable = np.ones((20, 20), dtype=bool)
        passable[5:16, 10] = False

        paths, labels = build_seed_paths(
            passable,
            np.ones((20, 20), dtype=float),
            lambda _a, _b, _layers: 1.0,
            (0, 0),
            (19, 19),
        )

        self.assertIn("astar_current", labels)
        self.assertIn("distance_field_astar_current", labels)
        self.assertIn("theta_star_current", labels)
        self.assertIn("dijkstra_current", labels)
        self.assertIn("dynamic_weighted_astar_e1", labels)
        self.assertIn("dynamic_weighted_astar_e2", labels)
        self.assertTrue(any(label.startswith("safe_rrt") for label in labels))
        self.assertTrue(all(not collision_cells_for_path(passable, path) for path in paths))

    def test_bidirectional_grid_seed_set_is_safe(self):
        passable = np.ones((20, 20), dtype=bool)
        passable[5:16, 10] = False

        paths, labels = build_bidirectional_seed_paths(
            passable,
            np.ones((20, 20), dtype=float),
            lambda _a, _b, _layers: 1.0,
            (0, 0),
            (19, 19),
        )

        self.assertIn("bidirectional_astar_current", labels)
        self.assertIn("bidirectional_dijkstra_current", labels)
        self.assertTrue(any(label.startswith("safe_rrt") for label in labels))
        self.assertTrue(all(not collision_cells_for_path(passable, path) for path in paths))

    def test_bidirectional_grid_only_seed_set_excludes_sampling_seeds(self):
        passable = np.ones((20, 20), dtype=bool)
        paths, labels = build_bidirectional_grid_only_seed_paths(
            passable,
            np.ones((20, 20), dtype=float),
            lambda _a, _b, _layers: 1.0,
            (0, 0),
            (19, 19),
        )

        self.assertTrue(paths)
        self.assertFalse(any(label.startswith("safe_rrt") for label in labels))
        self.assertTrue(all(label.startswith("bidirectional_") for label in labels))

    def test_rrt_only_seed_set_excludes_rrt_star(self):
        passable = np.ones((20, 20), dtype=bool)
        paths, labels = build_bidirectional_with_rrt_seed_paths(
            passable,
            np.ones((20, 20), dtype=float),
            lambda _a, _b, _layers: 1.0,
            (0, 0),
            (19, 19),
        )

        self.assertTrue(paths)
        self.assertIn("safe_rrt", labels)
        self.assertNotIn("safe_rrt_star", labels)

    @patch("uav_planner.experiments.no_task_corner.run_iamso_seed_ablation.iamso_optimize")
    @patch("uav_planner.experiments.no_task_corner.run_iamso_seed_ablation.build_seed_paths")
    def test_planner_disables_fallback_and_cruise_value(self, build_seeds, optimize):
        build_seeds.return_value = ([[(0.0, 0.0), (1.0, 1.0)]], ["astar_current"])
        optimize.return_value = {
            "path": [(0.0, 0.0), (1.0, 1.0)],
            "solution_source": "optimized",
            "feasible": True,
        }

        result = iamso_seed_ablation_planner(
            np.ones((2, 2), dtype=bool),
            np.ones((2, 2), dtype=float),
            lambda _a, _b, _layers: 1.0,
            (0, 0),
            (1, 1),
        )

        self.assertTrue(result)
        self.assertIsNone(optimize.call_args.kwargs["fallback_path"])
        self.assertEqual(optimize.call_args.kwargs["base_weights"], NO_TASK_LAYER_WEIGHTS)
        self.assertEqual(NO_TASK_LAYER_WEIGHTS[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
