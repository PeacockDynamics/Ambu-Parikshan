"""Deterministic composition of independent HydroLab-3D vehicles."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .simulator import Simulator


class MultiUUVSimulator:
    """Advance independent single-UUV kernels on a shared fixed timestep.

    It intentionally does not merge state or observations: each vehicle owns
    its own truth and sensor RNG.  Interaction models belong in dedicated
    world/communication components.
    """

    def __init__(self, vehicles: Mapping[str, Simulator]):
        if not isinstance(vehicles, Mapping) or len(vehicles) < 2:
            raise ValueError("vehicles must map at least two unique names to Simulator instances")
        if not all(isinstance(name, str) and name for name in vehicles) or not all(isinstance(simulator, Simulator) for simulator in vehicles.values()):
            raise TypeError("vehicles must map non-empty string names to Simulator instances")
        if len({sim.dt for sim in vehicles.values()}) != 1:
            raise ValueError("all vehicle simulators must use the same dt")
        if len({id(simulator) for simulator in vehicles.values()}) != len(vehicles):
            raise ValueError("every vehicle must own a distinct Simulator instance")
        if len({sim.time for sim in vehicles.values()}) != 1:
            raise ValueError("all vehicle simulators must begin at the same simulation time")
        self._vehicles = dict(vehicles)
        self._dt = next(iter(self._vehicles.values())).dt

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def time(self) -> float:
        return next(iter(self._vehicles.values())).time

    def reset(self) -> dict[str, dict[str, object]]:
        return {name: simulator.reset() for name, simulator in self._vehicles.items()}

    def state(self) -> dict[str, dict[str, object]]:
        return {name: simulator.state() for name, simulator in self._vehicles.items()}

    def step(self, actions: Mapping[str, object]) -> dict[str, dict[str, object]]:
        if set(actions) != set(self._vehicles):
            raise ValueError("actions must contain exactly one action for every vehicle")
        # Validate the entire application boundary before any member advances.
        validated = {}
        for name in self._vehicles:
            action = np.asarray(actions[name], dtype=float)
            if action.shape != (6,) or not np.all(np.isfinite(action)):
                raise ValueError("every fleet action must be a finite vector with shape (6,)")
            validated[name] = action.copy()
        return {name: simulator.step(validated[name]) for name, simulator in self._vehicles.items()}
