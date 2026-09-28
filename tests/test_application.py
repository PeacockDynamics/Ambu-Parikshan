"""Regression tests for the Ambu-Parikshan application layer."""

import json

import numpy as np
import pytest

from ambu_parikshan.config import ScenarioError, scenario_from_mapping
from ambu_parikshan.logging import write_run_artifacts
from ambu_parikshan.runtime import run_scenario


def valid_mapping():
    return {
        "name": "application-test",
        "simulation": {"dt": 0.02, "max_steps": 4, "seed": 42},
        "vehicle": {
            "initial_pose": [0, 0, -5, 0, 0, 0],
            "initial_body_velocity": [0, 0, 0, 0, 0, 0],
            "dynamics": {
                "mass": 50.0, "inertia": [2, 3, 4],
                "added_inertia": [5, 8, 10, 0.5, 0.7, 0.9],
                "linear_damping": [8, 12, 15, 1.5, 2, 2.5],
                "quadratic_damping": [12, 18, 22, 2, 2.5, 3],
                "fluid_density": 1025.0, "displaced_volume": 50 / 1025,
                "center_of_gravity": [0, 0, 0], "center_of_buoyancy": [0, 0, 0.08],
                "current_velocity_world": [0.15, -0.05, 0],
            },
        },
        "sensors": {},
        "action": {"mode": "constant_wrench", "wrench": [10, 0, 0, 0, 0, 0]},
    }


def test_scenario_runtime_is_deterministic():
    scenario = scenario_from_mapping(valid_mapping())
    first = run_scenario(scenario)
    second = run_scenario(scenario)
    assert len(first.records) == 4
    assert first.termination_reason == "max_steps"
    for a, b in zip(first.records, second.records):
        assert np.array_equal(a["truth"]["pose"], b["truth"]["pose"])
        assert np.array_equal(a["observation"]["dvl"], b["observation"]["dvl"])


@pytest.mark.parametrize("mutate", [
    lambda value: value.pop("vehicle"),
    lambda value: value["vehicle"].update({"initial_pose": [0, 0]}),
    lambda value: value["simulation"].update({"dt": 0}),
    lambda value: value["vehicle"]["dynamics"].update({"mass": -1}),
    lambda value: value.update({"unexpected": True}),
])
def test_invalid_scenarios_fail_clearly(mutate):
    value = valid_mapping()
    mutate(value)
    with pytest.raises(ScenarioError):
        scenario_from_mapping(value)


def test_artifacts_include_deterministic_metadata(tmp_path):
    result = run_scenario(scenario_from_mapping(valid_mapping()))
    paths = write_run_artifacts(result, tmp_path / "run")
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata == {
        "dt": 0.02, "scenario_name": "application-test", "seed": 42,
        "steps": 4, "termination_reason": "max_steps", "vehicle_ids": [],
        "vehicle_seeds": None,
    }
    assert len(paths["trajectory"].read_text().splitlines()) == 5


def test_scenario_constructs_core_world_geometry():
    value = valid_mapping()
    value["world"] = {
        "bounds": {"minimum": [-10, -10, -20], "maximum": [10, 10, 1]},
        "seabed": {"type": "flat", "z": -15},
        "obstacles": [{"type": "sphere", "center": [1, 0, -5], "radius": 1}],
    }
    scenario = scenario_from_mapping(value)
    assert scenario.world.altitude([0, 0, -5]) == 10
    assert scenario.world.collisions([1, 0, -5]) == [0]
