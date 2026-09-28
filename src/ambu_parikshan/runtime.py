"""Deterministic, headless application runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydrolab3d.core import Simulator
from hydrolab3d.core import MultiUUVSimulator
from hydrolab3d.communication import AcousticChannel
from hydrolab3d.vehicles import apply_wrench_fault

from .actions import action_source
from .config import FleetScenario, Scenario


@dataclass
class RunResult:
    """In-memory evidence from one finished scenario execution."""

    scenario_name: str
    seed: int | None
    dt: float
    termination_reason: str
    records: list[dict[str, object]]
    world: object | None = None
    fleet_seeds: dict[str, int | None] | None = None


def build_simulator(scenario: Scenario) -> Simulator:
    """Construct the validated HydroLab-3D kernel from scenario data."""
    return Simulator(
        initial_pose=scenario.initial_pose,
        initial_body_velocity=scenario.initial_body_velocity,
        dt=scenario.dt,
        dynamics_kwargs=scenario.dynamics_kwargs,
        sensor_config=scenario.sensor_config,
        sensor_seed=scenario.seed,
    )


def run_scenario(scenario: Scenario) -> RunResult:
    """Run a finite scenario; every movement passes through ``Simulator.step``."""
    simulator = build_simulator(scenario)
    source = action_source(scenario.action)
    source.reset()
    simulator.reset()
    records: list[dict[str, object]] = []
    for step in range(scenario.step_limit):
        action = source.action(simulator.state())
        result = simulator.step(action)
        records.append({
            "step": step + 1,
            "action": np.asarray(action, dtype=float).copy(),
            "truth": result["truth"],
            "observation": result["observation"],
        })
    # A duration is rounded up to complete fixed steps; max_steps wins ties.
    duration_steps = int(np.ceil(scenario.duration / scenario.dt)) if scenario.duration is not None else None
    reason = "max_steps" if scenario.max_steps is not None and (duration_steps is None or scenario.max_steps <= duration_steps) else "duration"
    return RunResult(scenario.name, scenario.seed, scenario.dt, reason, records, scenario.world)


def run_fleet_scenario(scenario: FleetScenario) -> RunResult:
    """Run independent UUVs, local inboxes, and simulation-time communication."""
    fleet = MultiUUVSimulator({name: build_simulator(vehicle) for name, vehicle in scenario.vehicles.items()})
    sources = {name: action_source(vehicle.action) for name, vehicle in scenario.vehicles.items()}
    channel = AcousticChannel(**scenario.communication) if scenario.communication else None
    fleet.reset(); [source.reset() for source in sources.values()]
    records = []
    for step in range(scenario.step_limit):
        state = fleet.state(); actions = {name: source.action(state[name]) for name, source in sources.items()}
        for fault in scenario.faults:
            # Closed interval [start_time, end_time], tolerant of binary step sums.
            epsilon = scenario.dt * 1e-9
            if fault["start_time"] - epsilon <= fleet.time and ("end_time" not in fault or fleet.time <= fault["end_time"] + epsilon):
                name = fault["vehicle"]; actions[name] = apply_wrench_fault(actions[name], scale=fault.get("scale", 1.0), stuck_action=fault.get("stuck_action"), failed_axes=fault.get("failed_axes", ()))
        result = fleet.step(actions); inboxes = {name: [] for name in scenario.vehicles}
        if channel:
            for sender, truth in result.items():
                for recipient, recipient_truth in result.items():
                    if sender != recipient:
                        channel.send(sender, recipient, {"pose": truth["truth"]["pose"].tolist()}, fleet.time, truth["truth"]["pose"][:3], recipient_truth["truth"]["pose"][:3])
            for packet in channel.receive(fleet.time): inboxes[packet.recipient].append(packet)
        records.append({"step": step + 1, "vehicles": result, "actions": {name: np.asarray(action).copy() for name, action in actions.items()}, "inboxes": inboxes})
    limits = [(vehicle.max_steps, "max_steps") for vehicle in scenario.vehicles.values() if vehicle.max_steps is not None]
    limits += [(int(np.ceil(vehicle.duration / vehicle.dt)), "duration") for vehicle in scenario.vehicles.values() if vehicle.duration is not None]
    _, reason = min(limits)
    return RunResult(scenario.name, scenario.seed, scenario.dt, reason, records, scenario.world, {name: vehicle.seed for name, vehicle in scenario.vehicles.items()})
