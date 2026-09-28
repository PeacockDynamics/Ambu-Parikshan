import copy

import numpy as np

from ambu_parikshan.config import scenario_from_mapping
from ambu_parikshan.logging import write_run_artifacts
from ambu_parikshan.runtime import run_fleet_scenario
from test_application import valid_mapping


def test_fleet_runtime_communication_faults_and_logging(tmp_path):
    base = valid_mapping(); vehicle = base["vehicle"]
    raw = {"name": "fleet", "simulation": {"dt": .02, "max_steps": 4, "seed": 4}, "vehicles": {
        "alpha": {**copy.deepcopy(vehicle), "action": {"mode": "constant_wrench", "wrench": [10, 0, 0, 0, 0, 0]}},
        "bravo": {**copy.deepcopy(vehicle), "initial_pose": [2, 0, -5, 0, 0, 0], "action": {"mode": "zero"}},
    }, "communication": {"max_range": 20, "latency": .02, "seed": 9}, "faults": [{"vehicle": "alpha", "start_time": .02, "scale": 0, "failed_axes": [0]}]}
    scenario = scenario_from_mapping(raw)
    first, second = run_fleet_scenario(scenario), run_fleet_scenario(scenario)
    assert first.records[0]["vehicles"]["alpha"]["truth"]["body_velocity"][0] > 0
    assert first.records[1]["actions"]["alpha"][0] == 0
    assert first.records[1]["inboxes"]["alpha"]
    assert np.array_equal(first.records[-1]["vehicles"]["bravo"]["truth"]["pose"], second.records[-1]["vehicles"]["bravo"]["truth"]["pose"])
    paths = write_run_artifacts(first, tmp_path)
    assert len(paths["trajectory"].read_text().splitlines()) == 9
