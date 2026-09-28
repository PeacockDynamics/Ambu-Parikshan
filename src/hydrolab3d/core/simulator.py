"""Core HydroLab-3D simulator kernel."""

from __future__ import annotations

import copy

import numpy as np

from hydrolab3d.sensors import (
    accelerometer_measurement,
    depth_measurement,
    dvl_measurement,
    gyroscope_measurement,
    heading_measurement,
)
from hydrolab3d.vehicles.uuv import (
    integrate_six_dof_euler,
)


def _copy_configuration(value):
    """Recursively copy mutable simulator configuration."""
    if isinstance(value, np.ndarray):
        return value.copy()

    if isinstance(value, dict):
        return {
            key: _copy_configuration(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _copy_configuration(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _copy_configuration(item)
            for item in value
        )

    return copy.deepcopy(value)


class Simulator:
    """
    Lightweight HydroLab-3D simulator kernel.

    The kernel owns simulator truth, deterministic model time,
    vehicle/environment configuration, and a reproducible
    sensor-observation layer.

    Vehicle propagation is delegated entirely to the production
    six-degree-of-freedom integrator.
    """

    def __init__(
        self,
        *,
        initial_pose,
        initial_body_velocity,
        dt,
        dynamics_kwargs,
        sensor_config=None,
        sensor_seed=None,
    ):
        initial_pose = np.asarray(
            initial_pose,
            dtype=float,
        )

        initial_body_velocity = np.asarray(
            initial_body_velocity,
            dtype=float,
        )

        dt = float(dt)

        if initial_pose.shape != (6,):
            raise ValueError(
                "initial_pose must have shape (6,)"
            )

        if initial_body_velocity.shape != (6,):
            raise ValueError(
                "initial_body_velocity must have shape (6,)"
            )

        if not np.all(
            np.isfinite(initial_pose)
        ):
            raise ValueError(
                "initial_pose must contain only finite values"
            )

        if not np.all(
            np.isfinite(initial_body_velocity)
        ):
            raise ValueError(
                "initial_body_velocity must contain only finite values"
            )

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "dt must be finite and strictly positive"
            )

        if not isinstance(
            dynamics_kwargs,
            dict,
        ):
            raise TypeError(
                "dynamics_kwargs must be a dictionary"
            )

        if sensor_config is None:
            sensor_config = {}

        if not isinstance(
            sensor_config,
            dict,
        ):
            raise TypeError(
                "sensor_config must be a dictionary"
            )

        self._initial_pose = (
            initial_pose.copy()
        )

        self._initial_body_velocity = (
            initial_body_velocity.copy()
        )

        self._dt = dt

        self._dynamics_kwargs = (
            _copy_configuration(
                dynamics_kwargs
            )
        )

        self._sensor_config = (
            _copy_configuration(
                sensor_config
            )
        )

        self._sensor_seed = copy.deepcopy(
            sensor_seed
        )

        self._pose = None
        self._body_velocity = None
        self._time = None
        self._diagnostics = None
        self._rng = None

        self.reset()


    @property
    def dt(self):
        """Return the fixed simulator timestep."""
        return self._dt


    @property
    def time(self):
        """Return current deterministic simulation time."""
        return self._time


    def state(self):
        """Return an independent ground-truth state snapshot."""
        return {
            "time": float(self._time),
            "pose": self._pose.copy(),
            "body_velocity": (
                self._body_velocity.copy()
            ),
        }


    @property
    def diagnostics(self):
        """Return an independent copy of the latest validated step diagnostics.

        This read-only inspection surface is intentionally separate from truth
        state and does not participate in propagation or observation creation.
        """
        return (
            None
            if self._diagnostics is None
            else _copy_configuration(self._diagnostics)
        )


    def reset(self):
        """
        Restore initial physical truth and sensor RNG state.
        """
        self._pose = (
            self._initial_pose.copy()
        )

        self._body_velocity = (
            self._initial_body_velocity.copy()
        )

        self._time = 0.0
        self._diagnostics = None

        self._rng = np.random.default_rng(
            self._sensor_seed
        )

        return self.state()


    def _sensor_settings(
        self,
        sensor_name,
    ):
        settings = (
            self._sensor_config.get(
                sensor_name,
                {},
            )
        )

        if not isinstance(
            settings,
            dict,
        ):
            raise TypeError(
                f"sensor_config['{sensor_name}'] "
                "must be a dictionary"
            )

        return _copy_configuration(
            settings
        )


    def _generate_observation(
        self,
        body_acceleration,
    ):
        """
        Generate post-step sensor measurements from truth.
        """
        body_acceleration = np.asarray(
            body_acceleration,
            dtype=float,
        )

        if body_acceleration.shape != (6,):
            raise ValueError(
                "body_acceleration must have shape (6,)"
            )

        if not np.all(
            np.isfinite(body_acceleration)
        ):
            raise ValueError(
                "body_acceleration must contain only finite values"
            )

        current_velocity_world = np.asarray(
            self._dynamics_kwargs[
                "current_velocity_world"
            ],
            dtype=float,
        )

        depth_settings = (
            self._sensor_settings(
                "depth"
            )
        )

        heading_settings = (
            self._sensor_settings(
                "heading"
            )
        )

        dvl_settings = (
            self._sensor_settings(
                "dvl"
            )
        )

        gyro_settings = (
            self._sensor_settings(
                "gyroscope"
            )
        )

        accel_settings = (
            self._sensor_settings(
                "accelerometer"
            )
        )

        return {
            "depth": float(
                depth_measurement(
                    self._pose,
                    rng=self._rng,
                    **depth_settings,
                )
            ),
            "heading": float(
                heading_measurement(
                    self._pose,
                    rng=self._rng,
                    **heading_settings,
                )
            ),
            "dvl": np.asarray(
                dvl_measurement(
                    self._pose,
                    self._body_velocity,
                    current_velocity_world,
                    rng=self._rng,
                    **dvl_settings,
                ),
                dtype=float,
            ).copy(),
            "gyroscope": np.asarray(
                gyroscope_measurement(
                    self._body_velocity,
                    rng=self._rng,
                    **gyro_settings,
                ),
                dtype=float,
            ).copy(),
            "accelerometer": np.asarray(
                accelerometer_measurement(
                    self._pose,
                    self._body_velocity,
                    body_acceleration,
                    rng=self._rng,
                    **accel_settings,
                ),
                dtype=float,
            ).copy(),
        }


    def step(self, action):
        """
        Advance exactly one physics timestep.

        Failure semantics
        -----------------
        Action validation happens before propagation.  Sensor observations are
        generated after the explicit-Euler physical state and time commit, as
        established by the Notebook 10 kernel.  Consequently an observation
        validation error can be raised after truth has advanced; callers must
        not assume ``step`` is transactional for sensor failures.

        Parameters
        ----------
        action : array-like, shape (6,)
            Generalized body-frame actuator wrench
            [Fx, Fy, Fz, Mx, My, Mz].

        Returns
        -------
        dict
            Dictionary containing copied ground truth and post-step
            sensor observations.
        """
        action = np.asarray(
            action,
            dtype=float,
        )

        if action.shape != (6,):
            raise ValueError(
                "action must have shape (6,)"
            )

        if not np.all(
            np.isfinite(action)
        ):
            raise ValueError(
                "action must contain only finite values"
            )

        step_kwargs = (
            _copy_configuration(
                self._dynamics_kwargs
            )
        )

        step_kwargs[
            "actuator_wrench"
        ] = action.copy()

        (
            next_pose,
            next_body_velocity,
            diagnostics,
        ) = integrate_six_dof_euler(
            self._pose,
            self._body_velocity,
            self._dt,
            **step_kwargs,
        )

        self._pose = np.asarray(
            next_pose,
            dtype=float,
        ).copy()

        self._body_velocity = np.asarray(
            next_body_velocity,
            dtype=float,
        ).copy()

        self._time += self._dt

        self._diagnostics = (
            _copy_configuration(
                diagnostics
            )
        )

        observation = (
            self._generate_observation(
                self._diagnostics[
                    "nu_dot"
                ]
            )
        )

        return {
            "truth": self.state(),
            "observation": (
                _copy_configuration(
                    observation
                )
            ),
        }
