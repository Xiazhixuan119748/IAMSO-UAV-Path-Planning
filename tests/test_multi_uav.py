import unittest

import numpy as np

from uav_planner.experiments.multi_uav.simulation import Config, Simulation, make_scenario
from uav_planner.experiments.multi_uav.planners import safe_astar


def straight(_passable, layers, _cost, start, goal):
    assert layers["tasks"] == []
    assert not layers["V"].any()
    return [start, goal]


def simulation(**kwargs):
    config = Config(size=10, tasks=2, sensor_radius=2, **kwargs)
    z = np.zeros((10, 10))
    layers = dict(O=z.copy(), I=z.copy(), R=z.copy(), W=np.zeros((10, 10, 2)),
                  V=np.ones((10, 10)), tasks=[(7, 7)], speed_map=np.ones((10, 10)) * 3)
    return Simulation(config, layers, np.ones((10, 10), dtype=bool), [(0, 0), (9, 9)],
                      [(2, 2), (7, 7)], straight)


class MultiUAVTests(unittest.TestCase):
    def test_invalid_config(self):
        for params in (dict(dt=0), dict(battery=5), dict(sensor_radius=float("nan")), dict(uavs=0)):
            with self.assertRaises(ValueError):
                Config(**params)

    def test_private_truth_and_complete_run(self):
        sim = simulation()
        result = sim.run()
        self.assertTrue(result["success"])
        self.assertEqual(result["completed"], 2)
        completed = [e["task"] for e in sim.events if e["kind"] == "task_completed"]
        self.assertEqual(sorted(completed), [0, 1])
        self.assertGreaterEqual(result["minimum_battery"], sim.config.reserve - 1e-7)
        self.assertEqual(len({tuple(row["time"] for row in u.trajectory) for u in sim.uavs}), 1)

    def test_sensing_along_segment_and_no_duplicate_discovery(self):
        sim = simulation()
        sim.sense(sim.uavs[0], (0, 2), (4, 2))
        sim.sense(sim.uavs[1], (0, 2), (4, 2))
        self.assertIn(0, sim.known)
        self.assertNotIn(1, sim.known)
        self.assertEqual(sum(e["kind"] == "task_discovered" for e in sim.events), 1)

    def test_directed_cache_and_reserve(self):
        sim = simulation(battery=8, reserve=5)
        self.assertIsNone(sim.feasible(sim.uavs[0], (9, 9)))
        sim.layers["W"][:, :, 0] = 2
        outbound = sim.plan((0, 1), (8, 1))
        inbound = sim.plan((8, 1), (0, 1))
        self.assertLess(outbound.energy, inbound.energy)

    def test_failed_planner_never_falls_back_or_succeeds(self):
        sim = simulation()
        sim.planner = lambda *args: None
        result = sim.run()
        self.assertFalse(result["success"])
        self.assertIsNone(result["finish_time"])
        self.assertGreater(result["planning_failures"], 0)
        self.assertEqual(result["termination"], "no_feasible_action")

    def test_collision_rejected(self):
        sim = simulation()
        sim.passable[4, 4] = 0
        self.assertIsNone(sim.plan((0, 0), (9, 9)))

    def test_strict_astar_does_not_cut_obstacle_corner(self):
        sim = simulation()
        sim.passable[1, 0] = 0
        sim.planner = safe_astar
        self.assertIsNotNone(sim.plan((0, 0), (3, 3)))

    def test_charging_restores_energy_after_fixed_delay(self):
        sim = simulation()
        u = sim.uavs[0]
        u.battery = 10
        sim.recharge(u)
        self.assertEqual(u.mode, "charging")
        sim.advance(u, sim.config.charge_time - 1)
        self.assertEqual(u.battery, 10)
        sim.advance(u, 1)
        self.assertEqual(u.battery, sim.config.battery)
        self.assertEqual(u.charges, 1)

    def test_timeout_is_partial_not_success(self):
        result = simulation(max_time=0.5).run()
        self.assertFalse(result["success"])
        self.assertEqual(result["termination"], "simulation_timeout")

    def test_zero_wall_budget_does_not_start_planning(self):
        sim = simulation()
        sim.deadline = 0
        self.assertIsNone(sim.plan((0, 0), (3, 3)))
        self.assertEqual(sim.planning_calls, 0)

    def test_reservations_are_unique(self):
        sim = simulation()
        sim.sense(sim.uavs[0], (0, 0), (4, 4))
        sim.schedule()
        assigned = [u.task for u in sim.uavs if u.task is not None]
        self.assertEqual(len(assigned), len(set(assigned)))
        for task in assigned:
            self.assertEqual(sim.known[task]["state"], "assigned")

    def test_generated_scenario_reproducible_and_private(self):
        config = Config(size=15, tasks=3)
        a, mask, stations, truth = make_scenario(config)
        b, _, other_stations, other_truth = make_scenario(config)
        np.testing.assert_array_equal(a["O"], b["O"])
        self.assertEqual(stations, other_stations)
        self.assertEqual(truth, other_truth)
        self.assertTrue(all(mask[p] for p in truth + stations))
        self.assertFalse(a["V"].any())
        self.assertEqual(a["tasks"], [])


if __name__ == "__main__":
    unittest.main()
