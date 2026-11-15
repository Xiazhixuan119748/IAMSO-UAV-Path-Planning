"""Synchronous multi-UAV simulation with private task truth and directed reserves."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time

import numpy as np
from scipy.ndimage import convolve, label

from uav_planner.costmap import C_wind, effective_speed, heading_vector
from uav_planner.costs import no_task_model_cost
from uav_planner.geometry import segment_has_collision
from uav_planner.map_generator import generate_map
from uav_planner.metaheuristic.encoding import sample_path


@dataclass(frozen=True)
class Config:
    size: int = 60
    seed: int = 42
    uavs: int = 3
    tasks: int = 10
    sensor_radius: float = 5.0
    battery: float = 65.0
    reserve: float = 5.0
    charge_time: float = 15.0
    service_time: float = 3.0
    service_power: float = 0.2
    dt: float = 0.5
    max_time: float = 1500.0
    wall_budget: float = 300.0
    candidates: int = 2
    time_weight: float = 0.5
    length_energy: float = 0.35
    time_energy: float = 0.2
    rain_energy: float = 0.2
    wind_energy: float = 0.2

    def __post_init__(self):
        for name in ("size", "seed", "uavs", "tasks", "candidates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.size < 8 or self.uavs < 1 or self.tasks < 1 or self.candidates < 1:
            raise ValueError("size >= 8; uavs, tasks, candidates >= 1 required")
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("sensor_radius", "battery", "charge_time", "dt", "max_time", "wall_budget"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("reserve", "service_time", "service_power", "length_energy", "time_energy", "rain_energy", "wind_energy"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.reserve >= self.battery or not 0 <= self.time_weight <= 1:
            raise ValueError("reserve < battery and 0 <= time_weight <= 1 required")


@dataclass
class Leg:
    points: list
    durations: list
    energies: list

    @property
    def duration(self):
        return sum(self.durations)

    @property
    def energy(self):
        return sum(self.energies)


@dataclass
class UAV:
    id: int
    position: tuple
    battery: float
    mode: str = "idle"
    target: tuple | None = None
    task: int | None = None
    leg: Leg | None = None
    segment: int = 0
    elapsed: float = 0.0
    timer: float = 0.0
    used: float = 0.0
    distance: float = 0.0
    charges: int = 0
    charge_seconds: float = 0.0
    minimum_battery: float = math.inf
    trajectory: list = field(default_factory=list)
    observed: set = field(default_factory=set)


def make_scenario(config):
    """Keep all sampled truth and both stations in the launch component."""
    layers, passable = generate_map(
        (0, 0), (config.size - 1, config.size - 1), seed=config.seed, size=config.size
    )
    components, _ = label(passable)
    reachable = components == components[0, 0]
    cells = np.argwhere(reachable)
    if len(cells) < config.tasks + 2:
        raise ValueError("launch component is too small for distinct tasks and stations")
    station2 = tuple(map(int, cells[np.argmax(np.sum(cells * cells, axis=1))]))
    stations = [(0, 0), station2]
    eligible = [tuple(map(int, p)) for p in cells if tuple(p) not in stations]
    rng = np.random.default_rng(config.seed + 10000)
    truth = [eligible[i] for i in rng.choice(len(eligible), config.tasks, replace=False)]
    layers = dict(layers, V=np.zeros_like(layers["V"]), tasks=[])
    return layers, reachable, stations, truth


class Simulation:
    def __init__(self, config, layers, searchable, stations, truth, planner):
        self.config = config
        shape = (config.size, config.size)
        if np.shape(searchable) != shape:
            raise ValueError("searchable shape must match config.size")
        if len(truth) != config.tasks or len(set(map(tuple, truth))) != len(truth):
            raise ValueError("truth must contain exactly N distinct task positions")
        if not stations:
            raise ValueError("at least one charging station required")
        for p in list(stations) + list(truth):
            if len(p) != 2 or any(int(x) != x or not 0 <= x < config.size for x in p):
                raise ValueError("stations and tasks must be in-bounds grid cells")
            if not searchable[tuple(p)] or layers["O"][tuple(p)]:
                raise ValueError("stations and tasks must be searchable and passable")
        self.layers = dict(layers, V=np.zeros(shape), tasks=[])
        self.passable = 1 - np.asarray(layers["O"])
        self.searchable = np.asarray(searchable, dtype=bool).copy()
        self.stations = list(map(tuple, stations))
        self._truth = np.asarray(truth, dtype=float)
        self.planner = planner
        self.uavs = [UAV(i, self.stations[0], config.battery) for i in range(config.uavs)]
        self.covered = np.zeros(shape, dtype=bool)
        self.known = {}
        self.events = []
        self.frames = []
        self.cache = {}
        self.t = 0.0
        self.planning_seconds = 0.0
        self.planning_calls = 0
        self.planning_failures = 0
        self.rejected_energy = 0
        self.sensed_cells = 0
        self.rows, self.cols = np.indices(shape)
        self.deadline = math.inf
        radius = math.ceil(config.sensor_radius)
        rr, cc = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        self.disk = (rr * rr + cc * cc <= config.sensor_radius ** 2).astype(float)

    def event(self, kind, uav=None, **data):
        self.events.append(dict(time=self.t, kind=kind, uav=None if uav is None else uav.id, **data))

    def plan(self, start, goal):
        start, goal = tuple(start), tuple(goal)
        key = (start, goal)
        if key in self.cache:
            return self.cache[key]
        if start == goal:
            return Leg([start], [], [])
        if time.perf_counter() >= self.deadline:
            return None
        began = time.perf_counter()
        path = self.planner(self.passable, self.layers, no_task_model_cost, start, goal)
        self.planning_seconds += time.perf_counter() - began
        self.planning_calls += 1
        valid = path is not None and len(path) >= 2
        if valid:
            valid = (
                np.allclose(path[0], start) and np.allclose(path[-1], goal)
                and np.isfinite(np.asarray(path, dtype=float)).all()
                and all(not segment_has_collision(self.passable, a, b) for a, b in zip(path, path[1:]))
            )
        if not valid:
            self.planning_failures += 1
            self.event("planning_failed", start=start, goal=goal)
            self.cache[key] = None
            return None
        points = [tuple(map(float, p)) for p in sample_path(path, step=0.5)]
        durations, energies = [], []
        c = self.config
        for a, b in zip(points, points[1:]):
            length = math.dist(a, b)
            mid = tuple(int(round((x + y) / 2)) for x, y in zip(a, b))
            heading = heading_vector(a, b)
            speed = effective_speed(mid, heading, self.layers)
            if not math.isfinite(speed) or speed <= 0:
                self.planning_failures += 1
                self.cache[key] = None
                return None
            duration = length / speed
            energy = (c.length_energy * length + c.time_energy * duration
                      + c.rain_energy * length * float(self.layers["R"][mid])
                      + c.wind_energy * length * C_wind(mid, heading, self.layers))
            durations.append(duration)
            energies.append(energy)
        result = Leg(points, durations, energies)
        self.cache[key] = result
        return result

    def return_leg(self, position):
        options = [(leg.energy, station, leg) for station in self.stations
                   if (leg := self.plan(position, station)) is not None]
        return min(options, key=lambda item: item[0]) if options else None

    def feasible(self, uav, goal, service=0.0):
        leg = self.plan(uav.position, goal)
        returning = self.return_leg(goal) if leg is not None else None
        if leg is None or returning is None:
            return None
        if leg.energy + service + returning[0] + self.config.reserve > uav.battery + 1e-9:
            self.rejected_energy += 1
            return None
        return leg

    def sense(self, uav, start, end):
        a, b = np.asarray(start), np.asarray(end)
        delta = b - a
        norm = float(delta @ delta)
        if norm:
            fraction = np.clip(((self.rows - a[0]) * delta[0] + (self.cols - a[1]) * delta[1]) / norm, 0, 1)
        else:
            fraction = 0.0
        distance2 = (self.rows - a[0] - fraction * delta[0]) ** 2 + (self.cols - a[1] - fraction * delta[1]) ** 2
        sensed = (distance2 <= self.config.sensor_radius ** 2) & self.searchable
        # Count each cell once per leg, so repeat coverage does not depend on dt.
        observed = set(map(int, np.flatnonzero(sensed))) - uav.observed
        self.sensed_cells += len(observed)
        uav.observed.update(observed)
        self.covered |= sensed
        fractions = np.clip((self._truth - a) @ delta / norm, 0, 1) if norm else np.zeros(len(self._truth))
        distances = np.linalg.norm(self._truth - (a + fractions[:, None] * delta), axis=1)
        for task in np.flatnonzero(distances <= self.config.sensor_radius):
            task = int(task)
            if task not in self.known:
                self.known[task] = dict(position=tuple(map(int, self._truth[task])), state="pending",
                                        discoverer=uav.id, discovered=self.t, owner=None)
                self.event("task_discovered", uav, task=task)

    def assign(self, uav, mode, goal, leg, task=None):
        uav.mode, uav.target, uav.leg, uav.task = mode, tuple(goal), leg, task
        uav.segment, uav.elapsed = 0, 0.0
        uav.observed.clear()
        if task is not None:
            self.known[task].update(state="assigned", owner=uav.id)
        self.event("assigned", uav, mode=mode, goal=tuple(goal), task=task,
                   planned_time=leg.duration, planned_energy=leg.energy)

    def recharge(self, uav):
        returning = self.return_leg(uav.position)
        if returning is None or returning[0] + self.config.reserve > uav.battery + 1e-9:
            uav.mode = "stranded"
            self.event("stranded", uav)
        elif returning[1] == tuple(uav.position):
            if uav.battery < self.config.battery - 1e-9:
                uav.mode, uav.timer = "charging", self.config.charge_time
                self.event("charge_started", uav)
        else:
            self.assign(uav, "returning", returning[1], returning[2])

    def schedule(self):
        c = self.config
        idle = [u for u in self.uavs if u.mode == "idle"]
        # shortcut: committed flights are not interrupted; add safe mid-flight replanning for later experiments.
        for task, info in self.known.items():
            if info["state"] != "pending":
                continue
            options = []
            for u in idle:
                leg = self.feasible(u, info["position"], c.service_time * c.service_power)
                if leg is not None:
                    score = (c.time_weight * (leg.duration + c.service_time) / c.max_time
                             + (1 - c.time_weight) * (leg.energy + c.service_time * c.service_power) / c.battery)
                    options.append((score, u.id, u, leg))
            if options:
                _, _, u, leg = min(options, key=lambda item: item[:2])
                self.assign(u, "task", info["position"], leg, task)
                idle.remove(u)
        unavailable = self.covered.copy()
        for u in self.uavs:
            if u.mode == "search":
                unavailable |= (self.rows - u.target[0]) ** 2 + (self.cols - u.target[1]) ** 2 <= c.sensor_radius ** 2
        for u in idle:
            gains = convolve((self.searchable & ~unavailable).astype(float), self.disk, mode="constant")
            distance = np.hypot(self.rows - u.position[0], self.cols - u.position[1])
            scores = np.where(self.searchable & (gains > 0), gains / (1 + distance), -1.0)
            order = np.argsort(scores.ravel(), kind="stable")[::-1]
            options = []
            tried = 0
            for flat in order:
                goal = tuple(map(int, np.unravel_index(flat, scores.shape)))
                # Expand the shortlist after energy rejection instead of declaring a top-k failure.
                if scores[goal] < 0 or (tried >= c.candidates and options) or tried >= max(c.candidates, 12):
                    break
                if goal == tuple(u.position):
                    continue
                if (tuple(u.position), goal) in self.cache and self.cache[(tuple(u.position), goal)] is None:
                    continue
                tried += 1
                leg = self.feasible(u, goal)
                if leg is not None:
                    cost = (c.time_weight * leg.duration / c.max_time
                            + (1 - c.time_weight) * leg.energy / c.battery)
                    options.append((cost / gains[goal], goal, leg))
            if options:
                _, goal, leg = min(options, key=lambda item: item[0])
                self.assign(u, "search", goal, leg)
                unavailable |= (self.rows - goal[0]) ** 2 + (self.cols - goal[1]) ** 2 <= c.sensor_radius ** 2
            else:
                self.recharge(u)

    def advance(self, uav, dt):
        c = self.config
        if uav.mode in ("charging", "servicing"):
            consumed = min(dt, uav.timer)
            uav.timer = max(0.0, uav.timer - dt)
            if uav.mode == "charging":
                uav.charge_seconds += consumed
                if uav.timer <= 1e-9:
                    uav.battery, uav.mode = c.battery, "idle"
                    uav.charges += 1
                    self.event("charge_completed", uav)
            else:
                energy = consumed * c.service_power
                uav.battery -= energy
                uav.used += energy
                if uav.timer <= 1e-9:
                    self.known[uav.task].update(state="completed", completed=self.t)
                    self.event("task_completed", uav, task=uav.task)
                    uav.mode, uav.task = "idle", None
        elif uav.mode in ("search", "task", "returning"):
            remaining = dt
            while uav.segment < len(uav.leg.durations) and remaining > 1e-12:
                i = uav.segment
                duration = uav.leg.durations[i]
                step = min(remaining, duration - uav.elapsed)
                start = uav.position
                uav.elapsed += step
                fraction = min(1.0, uav.elapsed / duration) if duration > 0 else 1.0
                a, b = np.asarray(uav.leg.points[i]), np.asarray(uav.leg.points[i + 1])
                uav.position = tuple(a + (b - a) * fraction)
                energy = uav.leg.energies[i] * step / duration if duration > 0 else 0.0
                uav.battery -= energy
                uav.used += energy
                uav.distance += math.dist(start, uav.position)
                self.sense(uav, start, uav.position)
                remaining -= step
                if uav.elapsed >= duration - 1e-12:
                    uav.segment += 1
                    uav.elapsed = 0.0
            if uav.segment == len(uav.leg.durations):
                uav.position = uav.target
                self.event("arrived", uav, mode=uav.mode, goal=uav.target)
                if uav.mode == "task":
                    self.known[uav.task]["state"] = "executing"
                    uav.mode, uav.timer = "servicing", c.service_time
                elif uav.mode == "returning":
                    uav.mode, uav.timer = "charging", c.charge_time
                    self.event("charge_started", uav)
                else:
                    uav.mode = "idle"
        uav.minimum_battery = min(uav.minimum_battery, uav.battery)
        if uav.battery < -1e-7:
            raise RuntimeError("energy accounting violated: negative battery")

    def snapshot(self):
        for u in self.uavs:
            u.trajectory.append(dict(time=self.t, row=float(u.position[0]), col=float(u.position[1]),
                                     battery=u.battery, mode=u.mode))
        self.frames.append(dict(time=self.t, discovered=len(self.known), completed=self.completed,
                                coverage=float(self.covered.sum() / self.searchable.sum())))

    @property
    def completed(self):
        return sum(task["state"] == "completed" for task in self.known.values())

    def run(self):
        began = time.perf_counter()
        self.deadline = began + self.config.wall_budget
        for u in self.uavs:
            self.sense(u, u.position, u.position)
            u.minimum_battery = u.battery
        self.snapshot()
        reason = "simulation_timeout"
        while self.t < self.config.max_time - 1e-9:
            if time.perf_counter() >= self.deadline:
                reason = "wall_budget"
                break
            self.schedule()
            if time.perf_counter() >= self.deadline:
                reason = "wall_budget"
                break
            if not any(u.mode not in ("idle", "stranded") for u in self.uavs):
                reason = "no_feasible_action"
                break
            dt = min(self.config.dt, self.config.max_time - self.t)
            self.t += dt
            # All positions advance before next scheduling decision; no UAV gets a private clock.
            for u in self.uavs:
                self.advance(u, dt)
            self.snapshot()
            if self.completed == self.config.tasks:
                reason = "all_tasks_completed"
                break
        self.event("terminated", reason=reason)
        energy = sum(u.used for u in self.uavs)
        return dict(config=asdict(self.config), success=self.completed == self.config.tasks,
                    termination=reason, simulation_time=self.t,
                    finish_time=self.t if self.completed == self.config.tasks else None,
                    discovered=len(self.known), completed=self.completed,
                    task_completion_fraction=self.completed / self.config.tasks,
                    discovery_fraction=len(self.known) / self.config.tasks,
                    mean_discovery_time=float(np.mean([x["discovered"] for x in self.known.values()])) if self.known else None,
                    coverage=float(self.covered.sum() / self.searchable.sum()),
                    repeat_search_fraction=1 - self.covered.sum().item() / self.sensed_cells if self.sensed_cells else 0.0,
                    total_energy=energy, energy_per_completed_task=energy / self.completed if self.completed else None,
                    total_distance=sum(u.distance for u in self.uavs),
                    charges=sum(u.charges for u in self.uavs),
                    charge_seconds=sum(u.charge_seconds for u in self.uavs),
                    minimum_battery=min(u.minimum_battery for u in self.uavs),
                    stranded=sum(u.mode == "stranded" for u in self.uavs),
                    energy_rejections=self.rejected_energy, planning_calls=self.planning_calls,
                    planning_failures=self.planning_failures, planning_seconds=self.planning_seconds,
                    wall_seconds=time.perf_counter() - began,
                    tasks=[dict(id=i, **info) for i, info in self.known.items()],
                    final_uavs=[dict(id=u.id, position=u.position, battery=u.battery, mode=u.mode) for u in self.uavs])
