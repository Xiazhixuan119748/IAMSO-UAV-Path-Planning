# Task Allocation Research Notes

Date: 2026-10-23. Scope: four corner launches, known tasks, five recharge
stations, sortie distance <= grid side count, instantaneous recharge.

## Current Implementation

`known_tasks.run_one` implements event-driven nearest-feasible greedy dispatch.
On arrival a vehicle sorts unassigned tasks by Euclidean distance, then accepts
the first whose actual planned outbound plus one feasible return fits remaining
range. The task is reserved immediately. Simultaneous arrivals are processed in
vehicle-ID order. There is no cross-vehicle bidding, joint assignment, route
sequence optimization, or fleet completion-time objective. Return stations are
tried in Euclidean order, not ranked by optimized future work. A visited-station
set prevents cycles but can restrict useful future station revisits.

IAMSO, MSO, AO, A*, Dijkstra and RRT only generate point-to-point paths here.
They do not allocate the tasks. Observed differences are interactions between
this shared dispatcher and each planner's paths/failures. The dispatcher sorts
using straight-line distance and uses real paths for feasibility, not for a
global best-cost choice. The older hidden-search simulator has a different
idle-vehicle score rule; it is not the dispatcher used by these known-task runs.

## Methods And Fit

1. Linear sum assignment: construct a vehicle-by-task cost matrix and jointly
   choose a minimum-sum matching for one round. Handles unequal task/vehicle
   counts. Include idle dummy choices and mask infeasible assignments. It removes
   sequential ID priority but does not optimize the complete sequence or reset
   range at charging stations. SciPy uses modified Jonker-Volgenant, not literally
   the Hungarian implementation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
2. CBAA/CBBA: agents bid for tasks or task bundles, exchange winner information
   and resolve conflicts through consensus. Appropriate when decentralized
   communication is part of the research. Convergence guarantees depend on the
   paper's scoring/connectivity assumptions; adding arbitrary charging costs
   does not automatically retain them. Choi, Brunet, How (2009):
   https://dspace.mit.edu/entities/publication/b0bf0a05-be3b-433b-9f4b-ce314ed5178b
3. Vehicle routing: jointly assign visits and order each vehicle's route. A
   makespan objective balances completion time; minimizing sum distance alone
   can produce unbalanced work. Official OR-Tools example:
   https://developers.google.com/optimization/routing/vrp
   Its basic distance dimension does NOT itself model battery refill. Repeated
   station visits, range resets, distinct starts and flexible end stations need
   explicit modeling and validation. No automatic optimality claim for heuristic
   routing solutions.
4. Fuel-constrained routing with refueling depots: targets and depot visits are
   planned together while keeping fuel feasible throughout a route. Sundar and
   Rathinam's cited paper is single-UAV, not a ready-made four-UAV allocator:
   https://arxiv.org/abs/1304.0494
   The authors' lab also lists Levy/Sundar/Rathinam (2014), Heuristics for Routing
   Heterogeneous Unmanned Vehicles with Fuel Constraints, as a multi-vehicle
   direction for deeper review:
   https://autonomy.engr.tamu.edu/research/path_planning_fuel/

## Recommended Next Comparison (Proposal, Not Implemented)

Keep current greedy as baseline; add joint one-round assignment first, then
range-feasible route insertion with cross-vehicle relocate/swap improvements.
For each insertion evaluate the complete affected sortie, including return or
recharge, and the resulting fleet makespan, not just distance to the new task.
Prefer completion feasibility first; evaluate time, distance and recharge count
separately or declare normalized objective weights before running experiments.

Centralized route construction is a closer match to current shared-state known-
task assumptions than distributed consensus. CBBA is a useful later baseline
if communication and dynamic discoveries become central again. These are design
recommendations inferred from the sources, not measured improvements.

Use the same scenes and same path planner to compare allocators; then hold the
allocator fixed to compare planners. Cache directed waypoint costs on static
maps; weather can make forward/reverse travel times unequal. Do not assume all
unvisited edges need expensive IAMSO evaluation: cheap bounds can screen them,
but final accepted plans require real-path verification. One failed stochastic
planning call is not proof that a task or entire scene is infeasible.
