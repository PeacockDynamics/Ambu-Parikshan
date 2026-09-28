import pytest

from hydrolab3d.metrics import final_waypoint_error, minimum_inter_vehicle_separation, trajectory_length
from hydrolab3d.vehicles import apply_wrench_fault


def test_metrics_are_small_and_deterministic():
    assert trajectory_length([[0, 0, 0], [3, 4, 0], [3, 4, 12]]) == 17
    assert final_waypoint_error([0, 0, 0], [3, 4, 0]) == 5
    assert minimum_inter_vehicle_separation([[0, 0, 0], [3, 4, 0], [10, 0, 0]]) == 5


def test_wrench_faults_are_isolated_and_explicit():
    assert apply_wrench_fault([1, 2, 3, 4, 5, 6], scale=.5, failed_axes=[1, 4]).tolist() == [.5, 0, 1.5, 2, 0, 3]
    assert apply_wrench_fault([1] * 6, stuck_action=[2] * 6).tolist() == [2] * 6
    with pytest.raises(ValueError): apply_wrench_fault([1] * 6, failed_axes=[6])
