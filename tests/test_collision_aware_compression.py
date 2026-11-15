import unittest

import numpy as np

from uav_planner.metaheuristic.collision_aware import (
    collision_aware_encode,
)
from uav_planner.geometry import collision_cells_for_path
from uav_planner.metaheuristic.encoding import decode


class CollisionAwareCompressionTests(unittest.TestCase):
    def test_preserves_required_corner_and_fills_fixed_waypoint_count(self):
        passable = np.ones((9, 9), dtype=bool)
        passable[0:7, 4] = False
        path = [(0, 0), (7, 0), (7, 5), (8, 8)]

        encoded = collision_aware_encode(path, passable, waypoint_count=6)
        decoded = decode(encoded, (0, 0), (8, 8))

        self.assertEqual(encoded.shape, (12,))
        self.assertFalse(collision_cells_for_path(passable, decoded))

    def test_rejects_path_that_needs_more_corners_than_budget(self):
        passable = np.ones((11, 11), dtype=bool)
        passable[0:9, 3] = False
        passable[2:11, 7] = False
        path = [(0, 0), (9, 0), (9, 5), (1, 5), (1, 10), (10, 10)]

        encoded = collision_aware_encode(path, passable, waypoint_count=2)

        self.assertIsNone(encoded)


if __name__ == "__main__":
    unittest.main()
