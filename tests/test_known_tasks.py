import unittest

import numpy as np

from uav_planner.experiments.multi_uav.known_tasks import run_one, scenario


def straight(_passable, _layers, _cost, start, goal):
    return [start, goal]


class KnownTasksTests(unittest.TestCase):
    def fixture(self):
        size = 10
        z = np.zeros((size, size))
        layers = dict(O=z, I=z, R=z, W=np.zeros((size, size, 2)), V=z,
                      tasks=[], speed_map=np.ones_like(z))
        stations = [(0, 0), (0, 9), (9, 0), (9, 9), (5, 5)]
        return size, layers, np.ones_like(z, dtype=bool), stations

    def test_parallel_time_range_and_unique_tasks(self):
        size, layers, mask, stations = self.fixture()
        tasks = [(0, 3), (3, 9), (9, 6), (6, 0)]
        result, flights, events, _ = run_one(size, 42, layers, mask, stations, tasks, "test", straight, 10)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["total_distance"], 24)
        self.assertAlmostEqual(result["total_runtime"], 6)
        self.assertEqual(result["charging_count"], 0)
        self.assertLessEqual(result["max_sortie_distance"], size)
        self.assertEqual(len([e for e in events if e["kind"] == "task_completed"]), 4)
        self.assertEqual({tuple(f["start"]) for f in flights if f["start_time"] == 0}, set(stations[:4]))

    def test_failure_not_success(self):
        size, layers, mask, stations = self.fixture()
        result, _, _, _ = run_one(size, 42, layers, mask, stations, [(3, 3)], "none", lambda *a: None, 10)
        self.assertFalse(result["success"])
        self.assertEqual(result["task_success_rate"], 0)

    def test_instant_charges_and_sortie_bound(self):
        size, layers, mask, stations = self.fixture()
        tasks = [(1, 3), (2, 5), (4, 5), (6, 5), (8, 5), (5, 2), (5, 8)]
        result, flights, events, _ = run_one(size, 42, layers, mask, stations, tasks, "test", straight, 10)
        self.assertTrue(result["success"])
        self.assertLessEqual(result["max_sortie_distance"], size + 1e-8)
        for event in events:
            if event["kind"] == "charged":
                self.assertTrue(any(f["end_time"] == event["time"] and f["uav"] == event["uav"] for f in flights))

    def test_scene_exact_launches_and_center(self):
        _, _, stations, tasks = scenario(30, 42, 10)
        self.assertEqual(stations, [(0, 0), (0, 29), (29, 0), (29, 29), (15, 15)])
        self.assertEqual(len(set(tasks)), 10)


if __name__ == "__main__":
    unittest.main()
