"""Focused regressions for the 0.1.0 release-candidate repair pass."""
import copy
import csv

import numpy as np
import pytest

from ambu_parikshan.actions import WaypointAction
from ambu_parikshan.config import ScenarioError, scenario_from_mapping
from ambu_parikshan.interactive import FixedStepAccumulator, interactive_supported
from ambu_parikshan.logging import write_run_artifacts
from ambu_parikshan.renderer import obstacle_marker_geometry, panda_rotation_matrix, visual_states
from ambu_parikshan.runtime import build_simulator, run_fleet_scenario, run_scenario
from hydrolab3d.communication import AcousticChannel
from hydrolab3d.core import MultiUUVSimulator
from hydrolab3d.utils.transforms import rotation_body_to_world
from hydrolab3d.world import HeightField, SphereObstacle
from test_application import valid_mapping


def _state(z=-5):
    return {"pose": np.array([0, 0, z, 0, 0, 0.0])}


@pytest.mark.parametrize(("target_z", "expected_sign"), [(-8, -1), (-2, 1), (-5, 0)])
def test_waypoint_depth_force_obeys_hydrolab_upward_z(target_z, expected_sign):
    source = WaypointAction({"waypoint": [0, 0, target_z], "surge_force": 0, "heave_kp": 4, "yaw_kp": 0, "max_heave_force": 20, "max_yaw_moment": 0})
    value = source.action(_state())[2]
    assert np.sign(value) == expected_sign


def test_config_parse_is_nonmutating_and_derives_distinct_fleet_seeds():
    base = valid_mapping(); vehicle = base["vehicle"]
    raw = {"name": "fleet", "simulation": {"dt": .02, "max_steps": 2, "seed": 4}, "vehicles": {
        "alpha": {**copy.deepcopy(vehicle), "sensors": {"depth": {"noise_std": .1}}, "action": {"mode": "zero"}},
        "bravo": {**copy.deepcopy(vehicle), "sensors": {"depth": {"noise_std": .1}}, "action": {"mode": "zero"}},
    }}
    original = copy.deepcopy(raw)
    first, second = scenario_from_mapping(raw), scenario_from_mapping(raw)
    assert raw == original
    assert first.vehicles["alpha"].seed != first.vehicles["bravo"].seed
    assert first.vehicles["alpha"].seed == second.vehicles["alpha"].seed
    run = run_fleet_scenario(first)
    depth_noise = [row["vehicles"][name]["observation"]["depth"] + row["vehicles"][name]["truth"]["pose"][2] for name in ("alpha", "bravo") for row in run.records]
    assert not np.array_equal(depth_noise[::2], depth_noise[1::2])


def test_explicit_fleet_seed_overrides_stable_derivation():
    base = valid_mapping(); vehicle = base["vehicle"]
    scenario = scenario_from_mapping({"name": "fleet", "simulation": {"dt": .02, "max_steps": 1, "seed": 4}, "vehicles": {"a": {**copy.deepcopy(vehicle), "seed": 123}, "b": copy.deepcopy(vehicle)}})
    assert scenario.vehicles["a"].seed == 123
    assert scenario.vehicles["b"].seed != 123


def test_fleet_world_is_retained_and_invalid_world_rejected():
    base = valid_mapping(); vehicle = base["vehicle"]
    raw = {"name": "fleet", "simulation": {"dt": .02, "max_steps": 1}, "vehicles": {"a": copy.deepcopy(vehicle), "b": copy.deepcopy(vehicle)}, "world": {"bounds": {"minimum": [-2, -2, -10], "maximum": [2, 2, 1]}, "seabed": {"type": "flat", "z": -8}}}
    assert scenario_from_mapping(raw).world.altitude([0, 0, -5]) == 3
    raw["world"] = {"unknown": 1}
    with pytest.raises(ScenarioError): scenario_from_mapping(raw)


def test_fleet_rejects_aliases_misaligned_clocks_and_partial_actions():
    scenario = scenario_from_mapping(valid_mapping()); one = build_simulator(scenario)
    with pytest.raises(ValueError, match="distinct"): MultiUUVSimulator({"a": one, "b": one})
    a, b = build_simulator(scenario), build_simulator(scenario); a.step(np.zeros(6))
    with pytest.raises(ValueError, match="same simulation time"): MultiUUVSimulator({"a": a, "b": b})
    a, b = build_simulator(scenario), build_simulator(scenario); fleet = MultiUUVSimulator({"a": a, "b": b})
    with pytest.raises(ValueError, match="fleet action"): fleet.step({"a": np.zeros(6), "b": np.zeros(5)})
    assert a.time == b.time == 0


def test_fleet_logs_identify_vehicle_and_reconstruct_seeds(tmp_path):
    base = valid_mapping(); vehicle = base["vehicle"]
    scenario = scenario_from_mapping({"name": "fleet", "simulation": {"dt": .02, "max_steps": 1, "seed": 9}, "vehicles": {"a": copy.deepcopy(vehicle), "b": copy.deepcopy(vehicle)}})
    paths = write_run_artifacts(run_fleet_scenario(scenario), tmp_path)
    rows = list(csv.DictReader(paths["trajectory"].open()))
    assert {row["vehicle_id"] for row in rows} == {"a", "b"}
    import json
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["seed"] == 9 and metadata["vehicle_seeds"]["a"] != metadata["vehicle_seeds"]["b"]


def test_packet_payload_is_snapshot_and_time_contract_is_explicit():
    channel = AcousticChannel(10, latency=1); payload = {"nested": [1]}
    channel.send("a", "b", payload, 0, [0, 0, 0], [1, 0, 0]); payload["nested"].append(2)
    assert channel.receive(1)[0].payload == {"nested": [1]}
    with pytest.raises(ValueError): channel.send("a", "b", {}, float("nan"), [0, 0, 0], [1, 0, 0])
    with pytest.raises(ValueError): channel.receive(-1)


def test_channel_blackout_is_send_time_policy_and_reset_keeps_configuration():
    channel = AcousticChannel(10, latency=1, blackouts=set())
    assert channel.send("a", "b", {"n": 1}, 0, [0, 0, 0], [1, 0, 0])
    channel.blackouts.add("b")
    assert channel.receive(1)[0].payload == {"n": 1}
    channel.reset()
    assert not channel.send("a", "b", {}, 0, [0, 0, 0], [1, 0, 0])


def test_geometry_rejects_nonfinite_grid_and_collision_radius():
    with pytest.raises(ValueError): HeightField([0, float("nan")], [0, 1], [[0, 1], [0, 1]])
    with pytest.raises(ValueError): SphereObstacle([0, 0, 0], 1).contains([0, 0, 0], float("nan"))


def test_renderer_obstacle_branching_does_not_evaluate_box_fields_for_sphere():
    center, scale = obstacle_marker_geometry(SphereObstacle([1, 2, 3], 4))
    assert np.array_equal(center, [1, 2, 3]) and scale == 4


def test_accumulator_uses_bounded_frame_time_and_resets():
    accumulator = FixedStepAccumulator(.02, .25)
    assert accumulator.add_frame_time(.051) == 2
    assert accumulator.value == pytest.approx(.011)
    assert accumulator.add_frame_time(1) == 13
    accumulator.reset(); assert accumulator.value == 0
    with pytest.raises(ValueError): accumulator.add_frame_time(-1)


def test_renderer_snapshots_and_matrix_conversion_preserve_hydrolab_attitude():
    record = {"truth": {"time": 0., "pose": np.array([1, 2, 3, .2, -.3, .4]), "body_velocity": np.zeros(6)}}
    state = visual_states(record)["uuv"]; record["truth"]["pose"][0] = 99
    assert state["pose"][0] == 1
    matrix = panda_rotation_matrix(state["pose"])
    assert np.allclose(matrix[:3, :3].T, rotation_body_to_world(.2, -.3, .4))
    assert np.allclose(matrix[3, :3], [1, 2, 3])


def test_interactive_rejects_fleet_instead_of_silently_changing_semantics():
    base = valid_mapping(); vehicle = base["vehicle"]
    fleet = scenario_from_mapping({"name": "fleet", "simulation": {"dt": .02, "max_steps": 1}, "vehicles": {"a": copy.deepcopy(vehicle), "b": copy.deepcopy(vehicle)}})
    with pytest.raises(ValueError, match="single-UUV"): interactive_supported(fleet)


def test_non_grid_duration_reports_duration_reason():
    raw = valid_mapping(); raw["simulation"] = {"dt": .03, "duration": .10, "seed": 1}
    result = run_scenario(scenario_from_mapping(raw))
    assert len(result.records) == 4 and result.termination_reason == "duration"


@pytest.mark.parametrize("patch", [lambda value: value["sensors"].update({"dvl": {"bias": [1, 2]}}), lambda value: value["sensors"].update({"depth": {"noise_std": -1}})])
def test_sensor_configuration_fails_early_for_invalid_measurement_shapes(patch):
    raw = valid_mapping(); raw["sensors"] = {}; patch(raw)
    with pytest.raises(ScenarioError): scenario_from_mapping(raw)


def test_fault_interval_is_closed_and_uses_pre_step_time():
    base = valid_mapping(); vehicle = base["vehicle"]
    raw = {"name": "fleet", "simulation": {"dt": .1, "max_steps": 4}, "vehicles": {
        "a": {**copy.deepcopy(vehicle), "action": {"mode": "constant_wrench", "wrench": [1, 0, 0, 0, 0, 0]}},
        "b": copy.deepcopy(vehicle),
    }, "faults": [{"vehicle": "a", "start_time": .1, "end_time": .2, "failed_axes": [0]}]}
    result = run_fleet_scenario(scenario_from_mapping(raw))
    assert [record["actions"]["a"][0] for record in result.records] == [1, 0, 0, 1]
