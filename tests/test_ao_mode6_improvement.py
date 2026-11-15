import unittest
from unittest.mock import patch

import numpy as np

from uav_planner.algorithms.ao import planner
from uav_planner.costs import no_task_model_cost
from uav_planner.geometry import PathResult
from uav_planner.metaheuristic import ao as ao_module


class Mode6AOImprovementTests(unittest.TestCase):
    def test_mode6_ao_uses_feasibility_priority_and_safe_smoothing(self):
        passable = np.ones((3, 3), dtype=bool)
        passable[1, 1] = False
        safe_path = [(0.0, 0.0), (0.0, 2.0), (2.0, 2.0)]
        captured = {}

        def fake_optimize(*args, **kwargs):
            captured.update(kwargs)
            return {"path": safe_path}

        with (
            patch.object(planner, "astar", lambda *args: safe_path),
            patch.object(planner, "ao_optimize", fake_optimize),
            patch.object(
                planner,
                "smooth_planner_result",
                lambda *args, **kwargs: PathResult([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]),
            ),
        ):
            result = planner.ao_planner(
                passable, np.ones((3, 3)), no_task_model_cost, (0, 0), (2, 2)
            )

        self.assertTrue(captured["feasibility_priority"])
        self.assertTrue(captured["collision_safe_smoothing"])
        self.assertEqual(result, safe_path)

    def test_internal_safe_smoothing_falls_back_to_raw_path(self):
        passable = np.ones((3, 3), dtype=bool)
        passable[1, 1] = False
        vector = np.array([0.0, 2.0])

        with patch.object(
            ao_module,
            "smooth_path",
            lambda path: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        ):
            result = ao_module._evaluate_candidate(
                vector,
                passable,
                np.ones((3, 3)),
                lambda row, col, speed_map: 1.0,
                (0, 0),
                (2, 2),
                True,
            )

        self.assertEqual(result.collision_count, 0)
        self.assertEqual(result.path, [(0.0, 0.0), (0.0, 2.0), (2.0, 2.0)])


if __name__ == "__main__":
    unittest.main()
