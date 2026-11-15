import inspect
import unittest
from unittest.mock import patch

import numpy as np

from uav_planner.algorithms.iamso.planner import (
    _NO_TASK_LAYER_WEIGHTS,
    _cardinal_safe_path,
    iamso_planner,
)
from uav_planner.costs import no_task_model_cost
from uav_planner.geometry import collision_cells_for_path, collision_segment_count_for_path, smooth_path
from uav_planner.metaheuristic.feasibility import FeasibleArchive, feasibility_key, is_feasible
from uav_planner.metaheuristic.iamso import iamso_optimize
from uav_planner.experiments.no_task_corner.run_iamso_nf import iamso_no_fallback_planner


def _evaluation(value, segments=0, cells=0, smoothness=0.0, path=None):
    return {
        "value": float(value),
        "segment_collision_count": float(segments),
        "collision_count": float(cells),
        "smoothness": float(smoothness),
        "path": list(path or [(0.0, 0.0), (1.0, 1.0)]),
    }


class FeasibilityPriorityTests(unittest.TestCase):
    def test_collision_free_path_beats_lower_scalar_collision_path(self):
        feasible = _evaluation(5000.0)
        collision = _evaluation(1.0, segments=1, cells=1)

        self.assertTrue(is_feasible(feasible))
        self.assertFalse(is_feasible(collision))
        self.assertLess(feasibility_key(feasible), feasibility_key(collision))

    def test_collision_order_uses_segments_then_cells(self):
        one_segment = _evaluation(100.0, segments=1, cells=5)
        two_segments = _evaluation(1.0, segments=2, cells=1)
        fewer_cells = _evaluation(1000.0, segments=1, cells=2)

        self.assertLess(feasibility_key(one_segment), feasibility_key(two_segments))
        self.assertLess(feasibility_key(fewer_cells), feasibility_key(one_segment))

    def test_archive_keeps_best_feasible_and_rejects_collisions(self):
        archive = FeasibleArchive()

        self.assertFalse(archive.consider(_evaluation(1.0, segments=1, cells=1), source="collision"))
        self.assertTrue(archive.consider(_evaluation(20.0), source="first"))
        self.assertFalse(archive.consider(_evaluation(30.0), source="worse"))
        self.assertTrue(archive.consider(_evaluation(10.0), source="better"))

        self.assertEqual(archive.source, "better")
        self.assertEqual(archive.evaluation["value"], 10.0)


class IAMSOFallbackTests(unittest.TestCase):
    def test_optimizer_default_search_budget(self):
        parameters = inspect.signature(iamso_optimize).parameters

        self.assertEqual(parameters["population_size"].default, 24)
        self.assertEqual(parameters["iterations"].default, 80)

    def test_optimizer_passes_map_to_custom_population_builder(self):
        passable = np.ones((3, 3), dtype=np.uint8)
        received = {}

        def builder(
            _rng, _population_size, _waypoint_count, _size, _start, _goal, _seeds, *, passable
        ):
            received["passable"] = passable
            return np.asarray([[1.0, 1.0]], dtype=float)

        result = iamso_optimize(
            passable,
            np.ones((3, 3), dtype=float),
            lambda _a, _b, _layers: 1.0,
            (0, 0),
            (2, 2),
            population_size=1,
            iterations=0,
            waypoint_count=1,
            initial_population_builder=builder,
            random_seed=1,
        )

        self.assertIs(received["passable"], passable)
        self.assertTrue(result["feasible"])

    def test_cardinal_fallback_avoids_diagonal_corner_cutting(self):
        passable = np.ones((5, 5), dtype=np.uint8)
        passable[0, 1] = 0
        passable[1, 0] = 0
        passable[1, 1] = 0
        start = (0, 0)
        goal = (4, 4)

        # Open a cardinal corridor from the otherwise boxed-in start.
        passable[1, 0] = 1
        path = _cardinal_safe_path(
            passable,
            np.ones((5, 5), dtype=float),
            lambda _p, _q, _layers: 1.0,
            start,
            goal,
        )

        self.assertTrue(path)
        self.assertFalse(collision_cells_for_path(passable, path))
        self.assertEqual(collision_segment_count_for_path(passable, path), 0)

    def test_optimizer_returns_raw_safe_seed_when_vector_space_is_infeasible(self):
        passable = np.ones((7, 7), dtype=np.uint8)
        passable[:6, 3] = 0
        fallback = [(0, 0), (6, 0), (6, 6), (0, 6)]

        result = iamso_optimize(
            passable,
            np.ones((7, 7), dtype=float),
            lambda _row, _col, _layers: 1.0,
            (0, 0),
            (0, 6),
            population_size=1,
            iterations=0,
            waypoint_count=1,
            seed_paths=None,
            fallback_path=fallback,
            random_seed=1,
        )

        self.assertTrue(result["feasible"])
        self.assertTrue(result["used_seed_fallback"])
        self.assertEqual(result["solution_source"], "seed_fallback")
        self.assertEqual(result["collision_count"], 0)
        self.assertEqual(result["segment_collision_count"], 0)
        self.assertEqual(result["path"], [(float(r), float(c)) for r, c in fallback])

    @patch("uav_planner.algorithms.iamso.planner.iamso_optimize")
    @patch("uav_planner.algorithms.iamso.planner.build_bidirectional_seed_paths")
    def test_planner_promotes_nf_bi_ca_under_iamso_name(self, build_seeds, optimize):
        verified_path = [(0.0, 0.0), (0.0, 2.0), (2.0, 2.0)]
        build_seeds.return_value = ([verified_path], ["bidirectional_astar_current"])
        optimize.return_value = {
            "path": verified_path,
            "solution_source": "initial_population",
            "feasible": True,
        }

        result = iamso_planner(
            np.ones((3, 3), dtype=np.uint8),
            np.ones((3, 3), dtype=float),
            no_task_model_cost,
            (0, 0),
            (2, 2),
        )

        self.assertEqual(list(result), smooth_path(verified_path, samples_per_segment=10))
        self.assertFalse(result.used_fallback)
        self.assertEqual(result.solution_source, "initial_population")
        self.assertTrue(result.feasible)
        self.assertEqual(result.seed_labels, ["bidirectional_astar_current"])
        optimize.assert_called_once()
        self.assertEqual(optimize.call_args.kwargs["seed_paths"], [verified_path])
        self.assertIsNone(optimize.call_args.kwargs["fallback_path"])
        self.assertEqual(optimize.call_args.kwargs["base_weights"], _NO_TASK_LAYER_WEIGHTS)
        self.assertEqual(
            optimize.call_args.kwargs["initial_population_builder"].__name__,
            "collision_aware_initial_population",
        )

    @patch("uav_planner.experiments.no_task_corner.run_iamso_nf.iamso_optimize")
    @patch("uav_planner.experiments.no_task_corner.run_iamso_nf._seed_paths")
    def test_no_fallback_ablation_disables_only_raw_fallback(self, seed_paths, optimize):
        seed_paths.return_value = [[(0, 0), (1, 1)]]
        optimize.return_value = {
            "path": [(0.0, 0.0), (1.0, 1.0)],
            "solution_source": "optimized",
            "feasible": True,
        }

        result = iamso_no_fallback_planner(
            np.ones((2, 2), dtype=np.uint8),
            np.ones((2, 2), dtype=float),
            lambda _row, _col, _layers: 1.0,
            (0, 0),
            (1, 1),
        )

        self.assertTrue(result)
        optimize.assert_called_once()
        self.assertEqual(optimize.call_args.kwargs["seed_paths"], seed_paths.return_value)
        self.assertIsNone(optimize.call_args.kwargs["fallback_path"])


if __name__ == "__main__":
    unittest.main()
