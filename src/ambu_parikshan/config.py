"""Scenario configuration for the Ambu-Parikshan application layer.

This module deliberately translates YAML into the existing HydroLab-3D
``Simulator`` contract.  It does not define a second vehicle model.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from hydrolab3d.world import BoxObstacle, FlatSeabed, HeightField, SphereObstacle, WaterVolume, WorldGeometry


class ScenarioError(ValueError):
    """Raised when a scenario cannot be translated into simulator inputs."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScenarioError(f"{name} must be a mapping")
    return value


def _only_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ScenarioError(f"{name} contains unsupported field(s): {', '.join(sorted(unknown))}")


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ScenarioError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"{name} must be a finite number") from exc
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and strictly positive" if positive else "finite"
        raise ScenarioError(f"{name} must be {qualifier}")
    return result


def _vector(value: Any, name: str, length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ScenarioError(f"{name} must be a length-{length} sequence")
    return [_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


@dataclass(frozen=True)
class Scenario:
    """Validated application scenario, ready to construct a Simulator."""

    name: str
    dt: float
    max_steps: int | None
    duration: float | None
    seed: int | None
    initial_pose: list[float]
    initial_body_velocity: list[float]
    dynamics_kwargs: dict[str, Any]
    sensor_config: dict[str, Any]
    action: dict[str, Any]
    world: WorldGeometry | None = None

    @property
    def step_limit(self) -> int:
        """Return the inclusive runtime limit implied by duration/steps."""
        limits = []
        if self.max_steps is not None:
            limits.append(self.max_steps)
        if self.duration is not None:
            limits.append(int(np.ceil(self.duration / self.dt)))
        return min(limits)


@dataclass(frozen=True)
class FleetScenario:
    name: str
    seed: int | None
    vehicles: dict[str, Scenario]
    communication: dict[str, Any]
    faults: list[dict[str, Any]]
    world: WorldGeometry | None = None

    @property
    def dt(self) -> float:
        return next(iter(self.vehicles.values())).dt

    @property
    def step_limit(self) -> int:
        return min(vehicle.step_limit for vehicle in self.vehicles.values())


_DYNAMICS_FIELDS = {
    "mass", "inertia", "added_inertia", "linear_damping", "quadratic_damping",
    "fluid_density", "displaced_volume", "center_of_gravity", "center_of_buoyancy",
    "current_velocity_world", "gravity",
}


def _seed(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= 2**64:
        raise ScenarioError(f"{name} must be an integer in [0, 2**64) or null")
    return value


def _derived_seed(seed: int | None, identifier: str) -> int | None:
    """Stable per-vehicle seed derivation; never use process-randomized hash()."""
    if seed is None:
        return None
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _world(raw: Any) -> WorldGeometry | None:
    if raw is None:
        return None
    raw = _mapping(raw, "world")
    _only_keys(raw, {"bounds", "surface_z", "seabed", "obstacles"}, "world")
    bounds = _mapping(raw.get("bounds"), "world.bounds")
    _only_keys(bounds, {"minimum", "maximum"}, "world.bounds")
    volume = WaterVolume(_vector(bounds.get("minimum"), "world.bounds.minimum", 3), _vector(bounds.get("maximum"), "world.bounds.maximum", 3), _number(raw.get("surface_z", 0.0), "world.surface_z"))
    seabed_raw = _mapping(raw.get("seabed"), "world.seabed")
    seabed_type = seabed_raw.get("type", "flat")
    if seabed_type == "flat":
        _only_keys(seabed_raw, {"type", "z"}, "world.seabed")
        seabed = FlatSeabed(_number(seabed_raw.get("z"), "world.seabed.z"))
    elif seabed_type == "height_field":
        _only_keys(seabed_raw, {"type", "x", "y", "heights"}, "world.seabed")
        x, y = seabed_raw.get("x"), seabed_raw.get("y")
        if not isinstance(x, list) or not isinstance(y, list):
            raise ScenarioError("world.seabed.x and world.seabed.y must be sequences")
        seabed = HeightField(x, y, seabed_raw.get("heights"))
    else:
        raise ScenarioError("world.seabed.type must be 'flat' or 'height_field'")
    obstacles = []
    for index, item in enumerate(raw.get("obstacles", [])):
        item = _mapping(item, f"world.obstacles[{index}]")
        if item.get("type") == "sphere":
            _only_keys(item, {"type", "center", "radius"}, f"world.obstacles[{index}]")
            obstacles.append(SphereObstacle(_vector(item.get("center"), f"world.obstacles[{index}].center", 3), _number(item.get("radius"), f"world.obstacles[{index}].radius", positive=True)))
        elif item.get("type") == "box":
            _only_keys(item, {"type", "minimum", "maximum"}, f"world.obstacles[{index}]")
            obstacles.append(BoxObstacle(_vector(item.get("minimum"), f"world.obstacles[{index}].minimum", 3), _vector(item.get("maximum"), f"world.obstacles[{index}].maximum", 3)))
        else:
            raise ScenarioError(f"world.obstacles[{index}].type must be 'sphere' or 'box'")
    return WorldGeometry(volume, seabed, obstacles)


def scenario_from_mapping(raw: Any) -> Scenario:
    """Validate a YAML-compatible mapping against the public Simulator inputs."""
    # Parsing is a translation boundary: never mutate caller mappings, even
    # when YAML aliases make nested values share identity.
    raw = copy.deepcopy(_mapping(raw, "scenario"))
    if "vehicles" in raw:
        _only_keys(raw, {"name", "simulation", "vehicles", "communication", "faults", "world"}, "fleet scenario")
        if not isinstance(raw.get("name"), str) or not raw["name"].strip() or not isinstance(raw.get("vehicles"), dict) or len(raw["vehicles"]) < 2:
            raise ScenarioError("fleet scenario requires a name and at least two vehicles")
        simulation = _mapping(raw.get("simulation"), "simulation")
        vehicles = {}
        for index, (identifier, vehicle) in enumerate(raw["vehicles"].items()):
            if not isinstance(identifier, str) or not identifier:
                raise ScenarioError("vehicle identifiers must be non-empty strings")
            vehicle = _mapping(vehicle, f"vehicles.{identifier}")
            vehicle = copy.deepcopy(vehicle)
            action = vehicle.pop("action", {"mode": "zero"})
            sensors = vehicle.pop("sensors", {})
            explicit_seed = vehicle.pop("seed", None)
            seed = _seed(explicit_seed, f"vehicles.{identifier}.seed") if explicit_seed is not None else _derived_seed(_seed(simulation.get("seed"), "simulation.seed"), identifier)
            local_simulation = dict(simulation); local_simulation["seed"] = seed
            vehicles[identifier] = scenario_from_mapping({"name": f"{raw['name']}/{identifier}", "simulation": local_simulation, "vehicle": vehicle, "sensors": sensors, "action": action})
        communication = _mapping(raw.get("communication", {}), "communication")
        _only_keys(communication, {"max_range", "latency", "packet_loss", "seed", "blackouts"}, "communication")
        if communication:
            if "max_range" not in communication:
                raise ScenarioError("communication.max_range is required when communication is configured")
            communication = {"max_range": _number(communication["max_range"], "communication.max_range", positive=True), "latency": _number(communication.get("latency", 0), "communication.latency"), "packet_loss": _number(communication.get("packet_loss", 0), "communication.packet_loss"), "seed": _seed(communication.get("seed", simulation.get("seed")), "communication.seed"), "blackouts": copy.deepcopy(communication.get("blackouts", []))}
            if communication["latency"] < 0 or not 0 <= communication["packet_loss"] <= 1 or not isinstance(communication["blackouts"], list):
                raise ScenarioError("communication latency/loss/blackouts are invalid")
        faults = copy.deepcopy(raw.get("faults", []))
        if not isinstance(faults, list): raise ScenarioError("faults must be a list")
        for fault in faults:
            _only_keys(_mapping(fault, "fault"), {"vehicle", "start_time", "end_time", "scale", "failed_axes", "stuck_action"}, "fault")
            if fault.get("vehicle") not in vehicles: raise ScenarioError("fault.vehicle must identify a configured vehicle")
            fault["start_time"] = _number(fault.get("start_time", 0), "fault.start_time")
            if fault["start_time"] < 0: raise ScenarioError("fault.start_time must be non-negative")
            if "end_time" in fault:
                fault["end_time"] = _number(fault["end_time"], "fault.end_time")
                if fault["end_time"] < fault["start_time"]: raise ScenarioError("fault.end_time must not precede start_time")
            if "scale" in fault:
                fault["scale"] = _number(fault["scale"], "fault.scale")
                if fault["scale"] < 0: raise ScenarioError("fault.scale must be non-negative")
            if "failed_axes" in fault:
                if not isinstance(fault["failed_axes"], list) or any(isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis < 6 for axis in fault["failed_axes"]): raise ScenarioError("fault.failed_axes must contain indices from 0 through 5")
            if "stuck_action" in fault: fault["stuck_action"] = _vector(fault["stuck_action"], "fault.stuck_action", 6)
        return FleetScenario(raw["name"].strip(), _seed(simulation.get("seed"), "simulation.seed"), vehicles, communication, faults, _world(raw.get("world")))
    _only_keys(raw, {"name", "simulation", "vehicle", "sensors", "action", "world"}, "scenario")
    for required in ("name", "simulation", "vehicle"):
        if required not in raw:
            raise ScenarioError(f"scenario is missing required field: {required}")
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ScenarioError("name must be a non-empty string")

    simulation = _mapping(raw["simulation"], "simulation")
    _only_keys(simulation, {"dt", "max_steps", "duration", "seed"}, "simulation")
    if "dt" not in simulation:
        raise ScenarioError("simulation is missing required field: dt")
    if "max_steps" not in simulation and "duration" not in simulation:
        raise ScenarioError("simulation requires max_steps and/or duration")
    dt = _number(simulation["dt"], "simulation.dt", positive=True)
    max_steps = simulation.get("max_steps")
    if max_steps is not None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ScenarioError("simulation.max_steps must be a positive integer")
    duration = simulation.get("duration")
    if duration is not None:
        duration = _number(duration, "simulation.duration", positive=True)
    seed = _seed(simulation.get("seed"), "simulation.seed")

    vehicle = _mapping(raw["vehicle"], "vehicle")
    _only_keys(vehicle, {"initial_pose", "initial_body_velocity", "dynamics"}, "vehicle")
    for required in ("initial_pose", "initial_body_velocity", "dynamics"):
        if required not in vehicle:
            raise ScenarioError(f"vehicle is missing required field: {required}")
    dynamics = _mapping(vehicle["dynamics"], "vehicle.dynamics")
    _only_keys(dynamics, _DYNAMICS_FIELDS, "vehicle.dynamics")
    required_dynamics = _DYNAMICS_FIELDS - {"gravity"}
    missing = required_dynamics - set(dynamics)
    if missing:
        raise ScenarioError("vehicle.dynamics is missing required field(s): " + ", ".join(sorted(missing)))
    parsed_dynamics = {
        "mass": _number(dynamics["mass"], "vehicle.dynamics.mass", positive=True),
        "inertia": _vector(dynamics["inertia"], "vehicle.dynamics.inertia", 3),
        "added_inertia": _vector(dynamics["added_inertia"], "vehicle.dynamics.added_inertia", 6),
        "linear_damping": _vector(dynamics["linear_damping"], "vehicle.dynamics.linear_damping", 6),
        "quadratic_damping": _vector(dynamics["quadratic_damping"], "vehicle.dynamics.quadratic_damping", 6),
        "fluid_density": _number(dynamics["fluid_density"], "vehicle.dynamics.fluid_density", positive=True),
        "displaced_volume": _number(dynamics["displaced_volume"], "vehicle.dynamics.displaced_volume", positive=True),
        "center_of_gravity": _vector(dynamics["center_of_gravity"], "vehicle.dynamics.center_of_gravity", 3),
        "center_of_buoyancy": _vector(dynamics["center_of_buoyancy"], "vehicle.dynamics.center_of_buoyancy", 3),
        "current_velocity_world": _vector(dynamics["current_velocity_world"], "vehicle.dynamics.current_velocity_world", 3),
    }
    if "gravity" in dynamics:
        parsed_dynamics["gravity"] = _number(dynamics["gravity"], "vehicle.dynamics.gravity", positive=True)

    sensors = raw.get("sensors", {})
    sensors = _mapping(sensors, "sensors")
    _only_keys(sensors, {"depth", "heading", "dvl", "gyroscope", "accelerometer"}, "sensors")
    parsed_sensors = {}
    for name, settings in sensors.items():
        settings = _mapping(settings, f"sensors.{name}")
        _only_keys(settings, {"bias", "noise_std"}, f"sensors.{name}")
        dimensions = 1 if name in {"depth", "heading"} else 3
        parsed = {}
        for field in ("bias", "noise_std"):
            value = settings.get(field, 0.0)
            if dimensions == 1:
                parsed[field] = _number(value, f"sensors.{name}.{field}")
            else:
                parsed[field] = _vector(value, f"sensors.{name}.{field}", dimensions) if isinstance(value, (list, tuple)) else _number(value, f"sensors.{name}.{field}")
            if field == "noise_std" and (np.any(np.asarray(parsed[field]) < 0)):
                raise ScenarioError(f"sensors.{name}.noise_std must be non-negative")
        parsed_sensors[name] = parsed

    action = raw.get("action", {"mode": "zero"})
    action = _mapping(action, "action")
    mode = action.get("mode")
    if mode == "zero":
        _only_keys(action, {"mode"}, "action")
    elif mode == "constant_wrench":
        _only_keys(action, {"mode", "wrench"}, "action")
        action = {"mode": mode, "wrench": _vector(action.get("wrench"), "action.wrench", 6)}
    elif mode == "waypoint":
        _only_keys(action, {"mode", "waypoint", "surge_force", "heave_kp", "yaw_kp", "max_heave_force", "max_yaw_moment"}, "action")
        action = {"mode": mode, "waypoint": _vector(action.get("waypoint"), "action.waypoint", 3), "surge_force": _number(action.get("surge_force", 0), "action.surge_force"), "heave_kp": _number(action.get("heave_kp", 0), "action.heave_kp"), "yaw_kp": _number(action.get("yaw_kp", 0), "action.yaw_kp"), "max_heave_force": _number(action.get("max_heave_force", 0), "action.max_heave_force"), "max_yaw_moment": _number(action.get("max_yaw_moment", 0), "action.max_yaw_moment")}
    else:
        raise ScenarioError("action.mode must be 'zero', 'constant_wrench', or 'waypoint'")

    return Scenario(
        name=raw["name"].strip(), dt=dt, max_steps=max_steps, duration=duration, seed=seed,
        initial_pose=_vector(vehicle["initial_pose"], "vehicle.initial_pose", 6),
        initial_body_velocity=_vector(vehicle["initial_body_velocity"], "vehicle.initial_body_velocity", 6),
        dynamics_kwargs=parsed_dynamics, sensor_config=parsed_sensors, action=copy.deepcopy(action), world=_world(raw.get("world")),
    )


def load_scenario(path: str | Path) -> Scenario:
    """Load one YAML scenario file without executing scenario-provided code."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ScenarioError(f"could not read scenario {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ScenarioError(f"invalid YAML in {path}: {exc}") from exc
    return scenario_from_mapping(raw)
