import numpy as np
import pytest

from hydrolab3d.core import Simulator
from hydrolab3d.vehicles.uuv import (
    integrate_six_dof_euler,
)


def base_initial_pose():
    return np.array(
        [
            0.0,
            0.0,
            -5.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def base_initial_velocity():
    return np.zeros(
        6,
        dtype=float,
    )


def base_dynamics(current=None):
    if current is None:
        current = np.array(
            [0.15, -0.05, 0.0],
            dtype=float,
        )

    mass = 50.0
    fluid_density = 1025.0

    return {
        "mass": mass,

        "inertia": np.array(
            [2.0, 3.0, 4.0],
            dtype=float,
        ),

        "added_inertia": np.array(
            [
                5.0,
                8.0,
                10.0,
                0.5,
                0.7,
                0.9,
            ],
            dtype=float,
        ),

        "linear_damping": np.array(
            [
                8.0,
                12.0,
                15.0,
                1.5,
                2.0,
                2.5,
            ],
            dtype=float,
        ),

        "quadratic_damping": np.array(
            [
                12.0,
                18.0,
                22.0,
                2.0,
                2.5,
                3.0,
            ],
            dtype=float,
        ),

        "fluid_density": (
            fluid_density
        ),

        "displaced_volume": (
            mass
            / fluid_density
        ),

        "center_of_gravity": np.array(
            [0.0, 0.0, 0.0],
            dtype=float,
        ),

        "center_of_buoyancy": np.array(
            [0.0, 0.0, 0.08],
            dtype=float,
        ),

        "current_velocity_world": (
            np.asarray(
                current,
                dtype=float,
            ).copy()
        ),
    }


def base_sensor_config():
    return {
        "depth": {
            "bias": 0.03,
            "noise_std": 0.04,
        },

        "heading": {
            "bias": np.deg2rad(
                0.75
            ),
            "noise_std": np.deg2rad(
                1.0
            ),
        },

        "dvl": {
            "bias": np.array(
                [
                    0.01,
                    -0.01,
                    0.005,
                ],
                dtype=float,
            ),
            "noise_std": 0.015,
        },

        "gyroscope": {
            "bias": np.array(
                [
                    0.001,
                    -0.001,
                    0.0005,
                ],
                dtype=float,
            ),
            "noise_std": 0.002,
        },

        "accelerometer": {
            "bias": np.array(
                [
                    0.02,
                    -0.01,
                    0.03,
                ],
                dtype=float,
            ),
            "noise_std": 0.025,
        },
    }


def make_simulator(
    *,
    pose=None,
    velocity=None,
    current=None,
    sensor_config=None,
    sensor_seed=1234,
):
    if pose is None:
        pose = (
            base_initial_pose()
        )

    if velocity is None:
        velocity = (
            base_initial_velocity()
        )

    if sensor_config is None:
        sensor_config = (
            base_sensor_config()
        )

    return Simulator(
        initial_pose=pose,
        initial_body_velocity=velocity,
        dt=0.02,
        dynamics_kwargs=(
            base_dynamics(
                current=current
            )
        ),
        sensor_config=(
            sensor_config
        ),
        sensor_seed=(
            sensor_seed
        ),
    )


def rollout(
    simulator,
    actions,
):
    times = []
    poses = []
    velocities = []

    depth = []
    heading = []
    dvl = []
    gyroscope = []
    accelerometer = []

    for action in actions:
        result = simulator.step(
            action
        )

        truth = result[
            "truth"
        ]

        observation = result[
            "observation"
        ]

        times.append(
            truth["time"]
        )

        poses.append(
            truth["pose"]
        )

        velocities.append(
            truth[
                "body_velocity"
            ]
        )

        depth.append(
            observation[
                "depth"
            ]
        )

        heading.append(
            observation[
                "heading"
            ]
        )

        dvl.append(
            observation[
                "dvl"
            ]
        )

        gyroscope.append(
            observation[
                "gyroscope"
            ]
        )

        accelerometer.append(
            observation[
                "accelerometer"
            ]
        )

    return {
        "times": np.asarray(
            times,
            dtype=float,
        ),

        "poses": np.asarray(
            poses,
            dtype=float,
        ),

        "velocities": np.asarray(
            velocities,
            dtype=float,
        ),

        "depth": np.asarray(
            depth,
            dtype=float,
        ),

        "heading": np.asarray(
            heading,
            dtype=float,
        ),

        "dvl": np.asarray(
            dvl,
            dtype=float,
        ),

        "gyroscope": np.asarray(
            gyroscope,
            dtype=float,
        ),

        "accelerometer": np.asarray(
            accelerometer,
            dtype=float,
        ),
    }


def test_reset_restores_exact_initial_state():
    pose = np.array(
        [
            1.0,
            -2.0,
            -6.0,
            0.1,
            -0.05,
            0.2,
        ],
        dtype=float,
    )

    velocity = np.array(
        [
            0.2,
            -0.03,
            0.01,
            0.005,
            -0.004,
            0.006,
        ],
        dtype=float,
    )

    simulator = make_simulator(
        pose=pose,
        velocity=velocity,
    )

    simulator.step(
        np.array(
            [
                10.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
    )

    state = simulator.reset()

    assert (
        state["time"]
        == 0.0
    )

    assert np.array_equal(
        state["pose"],
        pose,
    )

    assert np.array_equal(
        state[
            "body_velocity"
        ],
        velocity,
    )


def test_state_snapshot_is_independent():
    simulator = (
        make_simulator()
    )

    snapshot = (
        simulator.state()
    )

    snapshot[
        "pose"
    ][0] = 999.0

    snapshot[
        "body_velocity"
    ][0] = 999.0

    new_snapshot = (
        simulator.state()
    )

    assert (
        new_snapshot[
            "pose"
        ][0]
        != 999.0
    )

    assert (
        new_snapshot[
            "body_velocity"
        ][0]
        != 999.0
    )


@pytest.mark.parametrize(
    "bad_action",
    [
        np.zeros(
            3,
            dtype=float,
        ),

        np.array(
            [
                0.0,
                0.0,
                np.nan,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        ),

        np.array(
            [
                0.0,
                0.0,
                0.0,
                np.inf,
                0.0,
                0.0,
            ],
            dtype=float,
        ),
    ],
)
def test_invalid_actions_are_rejected(
    bad_action,
):
    simulator = (
        make_simulator()
    )

    with pytest.raises(
        ValueError
    ):
        simulator.step(
            bad_action
        )


@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.01,
        np.nan,
        np.inf,
    ],
)
def test_invalid_timesteps_are_rejected(
    dt,
):
    with pytest.raises(
        ValueError
    ):
        Simulator(
            initial_pose=(
                base_initial_pose()
            ),

            initial_body_velocity=(
                base_initial_velocity()
            ),

            dt=dt,

            dynamics_kwargs=(
                base_dynamics()
            ),
        )


def test_one_step_matches_direct_integrator_exactly():
    pose = np.array(
        [
            1.25,
            -0.75,
            -6.0,
            np.deg2rad(4.0),
            np.deg2rad(-3.0),
            np.deg2rad(20.0),
        ],
        dtype=float,
    )

    velocity = np.array(
        [
            0.35,
            -0.08,
            0.04,
            0.015,
            -0.010,
            0.025,
        ],
        dtype=float,
    )

    action = np.array(
        [
            17.0,
            -3.0,
            4.5,
            0.20,
            -0.15,
            0.35,
        ],
        dtype=float,
    )

    dynamics = (
        base_dynamics()
    )

    simulator = Simulator(
        initial_pose=pose,

        initial_body_velocity=(
            velocity
        ),

        dt=0.02,

        dynamics_kwargs=(
            dynamics
        ),

        sensor_config={},

        sensor_seed=1,
    )

    result = simulator.step(
        action
    )

    (
        direct_pose,
        direct_velocity,
        _,
    ) = integrate_six_dof_euler(
        pose,
        velocity,
        0.02,
        **dynamics,
        actuator_wrench=action,
    )

    assert np.array_equal(
        result[
            "truth"
        ]["pose"],
        direct_pose,
    )

    assert np.array_equal(
        result[
            "truth"
        ][
            "body_velocity"
        ],
        direct_velocity,
    )

    assert result[
        "truth"
    ]["time"] == pytest.approx(
        0.02,
        abs=1e-15,
    )


def test_reset_replay_is_exact():
    simulator = make_simulator(
        sensor_seed=4567,
    )

    actions = np.array(
        [
            [
                10.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],

            [
                8.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.04,
            ],

            [
                5.0,
                1.0,
                -1.0,
                0.02,
                0.0,
                -0.03,
            ],

            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    run_a = rollout(
        simulator,
        actions,
    )

    simulator.reset()

    run_b = rollout(
        simulator,
        actions,
    )

    for key in run_a:
        assert np.array_equal(
            run_a[key],
            run_b[key],
        )


def test_sensor_seed_changes_observations_not_truth():
    actions = np.array(
        [
            [
                10.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],

            [
                8.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.04,
            ],

            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    sim_a = make_simulator(
        sensor_seed=100,
    )

    sim_b = make_simulator(
        sensor_seed=101,
    )

    run_a = rollout(
        sim_a,
        actions,
    )

    run_b = rollout(
        sim_b,
        actions,
    )

    assert np.array_equal(
        run_a["times"],
        run_b["times"],
    )

    assert np.array_equal(
        run_a["poses"],
        run_b["poses"],
    )

    assert np.array_equal(
        run_a["velocities"],
        run_b["velocities"],
    )

    observation_changed = any(
        not np.array_equal(
            run_a[key],
            run_b[key],
        )
        for key in [
            "depth",
            "heading",
            "dvl",
            "gyroscope",
            "accelerometer",
        ]
    )

    assert observation_changed


def test_configuration_is_independently_owned():
    current = np.array(
        [
            0.2,
            -0.05,
            0.0,
        ],
        dtype=float,
    )

    inertia = np.array(
        [
            2.0,
            3.0,
            4.0,
        ],
        dtype=float,
    )

    dynamics = base_dynamics(
        current=current,
    )

    dynamics[
        "inertia"
    ] = inertia

    sensors = (
        base_sensor_config()
    )

    dvl_bias = sensors[
        "dvl"
    ]["bias"]

    reference_dynamics = (
        base_dynamics(
            current=np.array(
                [
                    0.2,
                    -0.05,
                    0.0,
                ],
                dtype=float,
            )
        )
    )

    reference_sensors = (
        base_sensor_config()
    )

    simulator = Simulator(
        initial_pose=(
            base_initial_pose()
        ),

        initial_body_velocity=(
            base_initial_velocity()
        ),

        dt=0.02,

        dynamics_kwargs=(
            dynamics
        ),

        sensor_config=(
            sensors
        ),

        sensor_seed=77,
    )

    reference = Simulator(
        initial_pose=(
            base_initial_pose()
        ),

        initial_body_velocity=(
            base_initial_velocity()
        ),

        dt=0.02,

        dynamics_kwargs=(
            reference_dynamics
        ),

        sensor_config=(
            reference_sensors
        ),

        sensor_seed=77,
    )

    current[:] = 999.0
    inertia[:] = 999.0
    dvl_bias[:] = 999.0

    sensors[
        "depth"
    ]["bias"] = 999.0

    actions = np.array(
        [
            [
                8.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],

            [
                6.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.03,
            ],

            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    test_run = rollout(
        simulator,
        actions,
    )

    reference_run = rollout(
        reference,
        actions,
    )

    for key in test_run:
        assert np.array_equal(
            test_run[key],
            reference_run[key],
        )


def test_current_changes_zero_action_trajectory():
    still = make_simulator(
        current=np.zeros(
            3,
            dtype=float,
        ),
        sensor_config={},
    )

    flowing = make_simulator(
        current=np.array(
            [
                0.4,
                -0.15,
                0.0,
            ],
            dtype=float,
        ),
        sensor_config={},
    )

    zero_action = np.zeros(
        6,
        dtype=float,
    )

    still_state = None
    flowing_state = None

    for _ in range(200):
        still_state = still.step(
            zero_action
        )["truth"]

        flowing_state = flowing.step(
            zero_action
        )["truth"]

    separation = np.linalg.norm(
        flowing_state[
            "pose"
        ][:3]
        - still_state[
            "pose"
        ][:3]
    )

    assert separation > 0.0


def test_long_rollout_remains_finite():
    simulator = make_simulator(
        sensor_seed=8080,
    )

    for index in range(500):
        t = (
            index
            * 0.02
        )

        action = np.array(
            [
                8.0
                + 2.0
                * np.sin(
                    0.4 * t
                ),

                np.sin(
                    0.3 * t
                ),

                0.5
                * np.sin(
                    0.2 * t
                ),

                0.02
                * np.sin(
                    0.6 * t
                ),

                0.03
                * np.sin(
                    0.5 * t
                ),

                0.04
                * np.sin(
                    0.35 * t
                ),
            ],
            dtype=float,
        )

        result = simulator.step(
            action
        )

        truth = result[
            "truth"
        ]

        observation = result[
            "observation"
        ]

        assert np.all(
            np.isfinite(
                truth["pose"]
            )
        )

        assert np.all(
            np.isfinite(
                truth[
                    "body_velocity"
                ]
            )
        )

        assert np.isfinite(
            observation[
                "depth"
            ]
        )

        assert np.isfinite(
            observation[
                "heading"
            ]
        )

        assert np.all(
            np.isfinite(
                observation[
                    "dvl"
                ]
            )
        )

        assert np.all(
            np.isfinite(
                observation[
                    "gyroscope"
                ]
            )
        )

        assert np.all(
            np.isfinite(
                observation[
                    "accelerometer"
                ]
            )
        )

    assert simulator.time == (
        pytest.approx(
            10.0,
            abs=1e-12,
        )
    )
