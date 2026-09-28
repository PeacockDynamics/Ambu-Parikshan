import numpy as np

from ambu_parikshan import load_scenario, run_scenario


def test_waypoint_application_reduces_forward_tracking_error():
    scenario = load_scenario("scenarios/waypoint_navigation.yaml")
    result = run_scenario(scenario)
    initial = np.linalg.norm(np.asarray(scenario.action["waypoint"]) - np.asarray(scenario.initial_pose[:3]))
    final = np.linalg.norm(np.asarray(scenario.action["waypoint"]) - result.records[-1]["truth"]["pose"][:3])
    assert final < initial
