import unittest
from unittest.mock import patch

from uav_planner.mode_runners.no_task_corner import run_no_task_corner_experiment_mode


class NoTaskCornerModeTests(unittest.TestCase):
    @patch("uav_planner.mode_runners.no_task_corner.save_report", return_value={})
    @patch("uav_planner.mode_runners.no_task_corner.print_comparison_table")
    @patch("uav_planner.mode_runners.no_task_corner.evaluate_dataset")
    @patch(
        "uav_planner.mode_runners.no_task_corner.select_algorithm",
        return_value=(object(), "IAMSO"),
    )
    @patch("builtins.input", side_effect=["2", "3"])
    def test_single_algorithm_and_map_scope_are_forwarded(
        self,
        _input,
        select_algorithm,
        evaluate_dataset,
        _print_table,
        _save_report,
    ):
        evaluate_dataset.return_value = {"summary": {"overall": []}}

        run_no_task_corner_experiment_mode()

        select_algorithm.assert_called_once_with()
        evaluate_dataset.assert_called_once_with(
            difficulties=["medium"],
            algorithm_names=["IAMSO"],
        )

    @patch("uav_planner.mode_runners.no_task_corner.save_report", return_value={})
    @patch("uav_planner.mode_runners.no_task_corner.print_comparison_table")
    @patch("uav_planner.mode_runners.no_task_corner.evaluate_dataset")
    @patch("uav_planner.mode_runners.no_task_corner.select_algorithm")
    @patch("builtins.input", side_effect=["1", "4"])
    def test_batch_mode_keeps_all_algorithms(
        self,
        _input,
        select_algorithm,
        evaluate_dataset,
        _print_table,
        _save_report,
    ):
        evaluate_dataset.return_value = {"summary": {"overall": []}}

        run_no_task_corner_experiment_mode()

        select_algorithm.assert_not_called()
        evaluate_dataset.assert_called_once_with(
            difficulties=["hard"],
            algorithm_names=None,
        )


if __name__ == "__main__":
    unittest.main()
