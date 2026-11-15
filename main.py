"""UAV global path-planning console entry point."""

from __future__ import annotations

from uav_planner.cli_inputs import (
    get_user_input,
    get_weights,
    select_algorithm,
    select_cost_function,
    select_map_generation_mode,
    select_mode,
    select_normal_map_size,
)
from uav_planner.mode_runners import (
    run_algorithm_comparison_mode,
    run_batch_mode,
    run_cost_comparison_mode,
    run_no_task_corner_experiment_mode,
    run_single_algorithm_mode,
    run_weighted_mode,
)


def main():
    mode = select_mode()

    if mode == 5:
        run_batch_mode()
        return
    if mode == 6:
        run_no_task_corner_experiment_mode()
        return

    difficulty_name, map_size = select_normal_map_size()
    map_generation_mode, map_generation_label = select_map_generation_mode()
    start, goal = get_user_input(map_size)

    if mode == 1:
        algo_func, algo_name = select_algorithm()
        cost_func, cost_name = select_cost_function()
        run_single_algorithm_mode(
            map_size,
            start,
            goal,
            algo_func,
            algo_name,
            cost_func,
            cost_name,
            map_generation_mode,
            map_generation_label,
            difficulty_name=difficulty_name,
        )
    elif mode == 2:
        cost_func, cost_name = select_cost_function()
        run_algorithm_comparison_mode(map_size, start, goal, cost_func, cost_name, map_generation_mode, map_generation_label)
    elif mode == 3:
        algo_func, algo_name = select_algorithm()
        run_cost_comparison_mode(map_size, start, goal, algo_func, algo_name, map_generation_mode, map_generation_label)
    elif mode == 4:
        algo_func, algo_name = select_algorithm()
        weights = get_weights()
        run_weighted_mode(map_size, start, goal, algo_func, algo_name, weights, map_generation_mode, map_generation_label)


if __name__ == "__main__":
    main()
