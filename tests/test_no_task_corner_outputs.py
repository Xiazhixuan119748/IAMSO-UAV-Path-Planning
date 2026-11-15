import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from uav_planner.experiments.no_task_corner import runner


class NoTaskCornerOutputTests(unittest.TestCase):
    @staticmethod
    def _summary_row():
        return {
            "algorithm": "Dijkstra",
            "attempts": 100,
            "path_found_rate": 1.0,
            "success_rate": 0.19,
            "no_path_count": 0,
            "collision_failure_count": 81,
            "exception_count": 0,
            "avg_distance": 43.8444,
            "avg_time": 18.5616,
            "avg_safety": 0.7998,
            "avg_collision": 0.81,
            "avg_smoothness": 0.6448,
            "avg_rain_exposure": 5.9646,
            "avg_inflation_exposure": 19.9526,
            "avg_total_exposure": 25.9173,
            "avg_score": 53.7312,
            "avg_runtime": 0.0635,
        }

    def test_single_difficulty_uses_mode5_style_chinese_name(self):
        stem = runner._result_file_stem(["simple"], timestamp="20260714_200000")

        self.assertEqual(
            stem,
            "算法批量比较结果_简单模式_固定地图无任务价值_20260714_200000",
        )

    def test_multiple_difficulties_use_all_difficulties_label(self):
        stem = runner._result_file_stem(
            ["simple", "medium", "hard"],
            timestamp="20260714_200000",
        )

        self.assertEqual(
            stem,
            "算法批量比较结果_全部难度模式_固定地图无任务价值_20260714_200000",
        )

    def test_save_report_writes_only_markdown(self):
        report = {"selected_difficulties": ["hard"]}
        with TemporaryDirectory() as temp_dir, patch.object(
            runner, "no_task_corner_results_dir", return_value=Path(temp_dir)
        ), patch.object(
            runner, "_result_file_stem", return_value="算法批量比较结果_困难模式_固定地图无任务价值_时间戳"
        ), patch.object(runner, "_write_json") as write_json, patch.object(
            runner, "_write_summary_csv"
        ) as write_summary_csv, patch.object(
            runner, "_write_details_csv"
        ) as write_details_csv, patch.object(runner, "_write_markdown") as write_markdown:
            paths = runner.save_report(report)

        self.assertEqual(list(paths), ["实验报告 Markdown"])
        self.assertTrue(paths["实验报告 Markdown"].endswith(".md"))
        write_markdown.assert_called_once()
        write_json.assert_not_called()
        write_summary_csv.assert_not_called()
        write_details_csv.assert_not_called()

    def test_results_directory_is_under_project_root(self):
        self.assertEqual(
            runner.no_task_corner_results_dir(),
            Path(__file__).resolve().parents[1] / "无价值固定地图批量生成结果",
        )

    def test_single_algorithm_file_name_includes_algorithm(self):
        stem = runner._result_file_stem(
            ["medium"],
            timestamp="20260716_120000",
            algorithm_names=["IAMSO"],
        )

        self.assertEqual(
            stem,
            "单算法固定地图实验_IAMSO_中等模式_无任务价值_20260716_120000",
        )

    def test_resolve_algorithms_returns_only_requested_algorithm(self):
        selected = runner._resolve_algorithms(["IAMSO"])

        self.assertEqual([name for name, _ in selected], ["IAMSO"])
        self.assertEqual(selected[0][1].__name__, "iamso_planner")

    def test_resolve_algorithms_rejects_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown algorithms: missing"):
            runner._resolve_algorithms(["missing"])

    def test_evaluate_dataset_runs_only_selected_algorithm(self):
        other_algorithm = Mock(return_value=None)
        selected_algorithm = Mock(return_value=None)
        record = {
            "id": "simple_001",
            "file": "simple_001.npz",
            "difficulty": "simple",
            "difficulty_label": "简单",
            "map_size": 2,
            "seed": 1,
            "start": [0, 0],
            "goal": [1, 1],
        }
        index = {
            "dataset_name": "test",
            "dataset_version": "1",
            "root": "maps",
            "difficulty_specs": {"simple": {"label": "简单", "target_obstacle_ratio": 0.15}},
            "value_mode": "no_task",
            "start_rule": "fixed_upper_left",
            "goal_rule": "fixed_lower_right",
        }
        layers = {"tasks": []}
        passable = np.ones((2, 2), dtype=bool)
        algorithms = {
            "1": ("Dijkstra", other_algorithm),
            "2": ("IAMSO", selected_algorithm),
        }
        with patch.object(runner, "ALGORITHMS", algorithms), patch.object(
            runner, "collect_records", return_value=(index, [record])
        ), patch.object(
            runner, "load_map_bundle", return_value=(layers, passable, {})
        ):
            report = runner.evaluate_dataset(
                difficulties=["simple"],
                algorithm_names=["IAMSO"],
            )

        other_algorithm.assert_not_called()
        selected_algorithm.assert_called_once()
        self.assertEqual(report["algorithms"], ["IAMSO"])
        self.assertEqual(len(report["details"]), 1)

    def test_summary_csv_contains_only_confirmed_columns(self):
        report = {
            "summary": {
                "overall": [self._summary_row()],
                "by_difficulty": {"simple": [self._summary_row()]},
            }
        }
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.csv"
            runner._write_summary_csv(output_path, report)
            with output_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(
            list(rows[0]),
            [
                "algorithm",
                "success_rate",
                "avg_distance",
                "avg_time",
                "avg_safety",
                "avg_collision",
                "avg_score",
                "avg_runtime",
                "collision_failure_count",
            ],
        )
        self.assertEqual(len(rows), 1)

    def test_markdown_table_contains_only_confirmed_columns(self):
        lines = []
        runner._append_markdown_table(lines, [self._summary_row()], "总体结果")
        header = lines[2]

        expected_columns = [
            "算法",
            "成功率",
            "平均路径长度",
            "平均通行时间",
            "平均安全度",
            "平均碰撞次数",
            "平均综合评分/100",
            "平均运行时间/s",
            "碰撞失败次数",
        ]
        self.assertEqual([item.strip() for item in header.strip("|").split("|")], expected_columns)


if __name__ == "__main__":
    unittest.main()
