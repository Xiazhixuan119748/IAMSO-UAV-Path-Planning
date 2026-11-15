import unittest

import numpy as np

from uav_planner.experiments.no_task_corner.map_generator import is_strictly_reachable


class NoTaskCornerMapGenerationTests(unittest.TestCase):
    def test_strict_reachability_rejects_diagonal_corner_crossing(self):
        passable = np.ones((3, 3), dtype=np.uint8)
        passable[0, 1] = 0
        passable[1, 0] = 0

        self.assertFalse(is_strictly_reachable(passable, (0, 0), (2, 2)))

    def test_strict_reachability_accepts_collision_free_route(self):
        passable = np.ones((4, 4), dtype=np.uint8)
        passable[1, 1:3] = 0

        self.assertTrue(is_strictly_reachable(passable, (0, 0), (3, 3)))

if __name__ == "__main__":
    unittest.main()
