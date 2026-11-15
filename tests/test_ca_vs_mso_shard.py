import unittest
from unittest.mock import patch

import numpy as np

from uav_planner.experiments.no_task_corner.aggregate_ca_vs_mso_shards import (
    _exact_mcnemar_two_sided,
    runtime_analysis,
    validate_and_merge_shards,
)
from uav_planner.experiments.no_task_corner.run_ca_vs_mso_shard import (
    ALGORITHM_NAME,
    MSO_NAME,
    WAYPOINT_COUNT,
    iamso_collision_aware_planner,
    shard_ids,
)
from uav_planner.experiments.no_task_corner.run_ca_vs_mso_parallel import (
    benchmark_shards,
    single_map_shards,
)


class CAVsMSOShardTests(unittest.TestCase):
    @staticmethod
    def _shard_report(difficulty, start_index, end_index):
        ids = [
            f"{difficulty}_{index:03d}"
            for index in range(start_index, end_index + 1)
        ]
        return {
            "run_id": "test-run",
            "dataset_name": "no_task_corner",
            "dataset_version": "1.1",
            "cost_profile": "no_task_model_cost",
            "evaluation_profile": "no_task",
            "map_count": len(ids),
            "algorithms": [MSO_NAME, ALGORITHM_NAME],
            "ablation": {
                "waypoint_count": WAYPOINT_COUNT,
                "fallback_path_enabled": False,
                "seed_paths_enabled": True,
                "bidirectional_seed_paths_enabled": True,
                "collision_aware_compression_enabled": True,
                "final_smoothing_enabled": True,
            },
            "shard": {
                "difficulty": difficulty,
                "start_index": start_index,
                "end_index": end_index,
                "map_ids": ids,
            },
            "details": [
                {"map_id": map_id, "algorithm": algorithm}
                for map_id in ids
                for algorithm in (MSO_NAME, ALGORITHM_NAME)
            ],
        }

    def test_shard_ids_are_closed_and_non_overlapping(self):
        first = shard_ids("hard", 1, 50)
        second = shard_ids("hard", 51, 100)
        self.assertFalse(first & second)
        self.assertEqual(len(first | second), 100)

    def test_parallel_partition_covers_each_map_once(self):
        shards = benchmark_shards()
        for difficulty in ("simple", "medium", "hard"):
            ids = [
                map_id
                for shard in shards
                if shard.difficulty == difficulty
                for map_id in shard.map_ids
            ]
            self.assertEqual(len(ids), 100)
            self.assertEqual(len(set(ids)), 100)

    def test_single_map_partition_is_complete_and_checkpointable(self):
        shards = single_map_shards()
        self.assertEqual(len(shards), 300)
        self.assertTrue(all(shard.start_index == shard.end_index for shard in shards))

    @patch("uav_planner.experiments.no_task_corner.run_ca_vs_mso_shard.iamso_optimize")
    @patch("uav_planner.experiments.no_task_corner.run_ca_vs_mso_shard.build_bidirectional_seed_paths")
    def test_ca_planner_keeps_fallback_disabled(self, build_seeds, optimize):
        build_seeds.return_value = ([[(0.0, 0.0), (1.0, 1.0)]], ["seed"])
        optimize.return_value = {
            "path": [(0.0, 0.0), (1.0, 1.0)],
            "solution_source": "optimized",
            "feasible": True,
        }
        result = iamso_collision_aware_planner(
            np.ones((2, 2), dtype=bool),
            np.ones((2, 2), dtype=float),
            lambda _a, _b, _layers: 1.0,
            (0, 0),
            (1, 1),
        )
        self.assertTrue(result)
        self.assertIsNone(optimize.call_args.kwargs["fallback_path"])
        self.assertEqual(optimize.call_args.kwargs["waypoint_count"], WAYPOINT_COUNT)
        self.assertEqual(
            optimize.call_args.kwargs["initial_population_builder"].__name__,
            "collision_aware_initial_population",
        )

    def test_aggregate_validation_accepts_complete_non_overlapping_shards(self):
        reports = [
            self._shard_report("simple", 1, 1),
            self._shard_report("simple", 2, 2),
        ]

        details = validate_and_merge_shards(
            reports, difficulties=("simple",), expected_count=2
        )

        self.assertEqual(len(details), 4)

    def test_aggregate_validation_rejects_duplicate_shard_range(self):
        report = self._shard_report("simple", 1, 1)

        with self.assertRaisesRegex(ValueError, "duplicate shard range"):
            validate_and_merge_shards(
                [report, report], difficulties=("simple",), expected_count=1
            )

    def test_exact_mcnemar_uses_only_discordant_pairs(self):
        self.assertAlmostEqual(_exact_mcnemar_two_sided(9, 0), 0.00390625)
        self.assertEqual(_exact_mcnemar_two_sided(0, 0), 1.0)

    def test_runtime_analysis_reports_hard_outlier_trim(self):
        details = []
        for difficulty in ("simple", "medium", "hard"):
            for index in range(1, 101):
                map_id = f"{difficulty}_{index:03d}"
                for algorithm in (MSO_NAME, ALGORITHM_NAME):
                    runtime = 100.0 if map_id in {"hard_094", "hard_097"} else 1.0
                    details.append(
                        {
                            "difficulty": difficulty,
                            "map_id": map_id,
                            "algorithm": algorithm,
                            "runtime": runtime,
                        }
                    )

        analysis = runtime_analysis(details)

        self.assertEqual(analysis["hard"][ALGORITHM_NAME]["median"], 1.0)
        self.assertEqual(
            analysis["hard"][ALGORITHM_NAME]["mean_excluding_hard_094_097"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
