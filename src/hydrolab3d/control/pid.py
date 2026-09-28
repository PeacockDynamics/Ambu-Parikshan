"""Minimal classical feedback controllers for HydroLab-3D."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PIDController:
    """Minimal scalar PID controller.

    Parameters
    ----------
    kp, ki, kd : float
        Proportional, integral, and derivative gains.

    output_limits : tuple[float, float]
        Minimum and maximum controller output.

    integral_limits : tuple[float, float] or None
        Optional bounds on accumulated integral state.
    """

    kp: float
    ki: float
    kd: float
    output_limits: tuple = (-1.0, 1.0)
    integral_limits: tuple | None = None

    def __post_init__(self):
        self.kp = float(self.kp)
        self.ki = float(self.ki)
        self.kd = float(self.kd)

        output_min, output_max = self.output_limits

        self.output_min = float(output_min)
        self.output_max = float(output_max)

        if self.output_min > self.output_max:
            raise ValueError(
                "output_limits must satisfy minimum <= maximum"
            )

        if self.integral_limits is None:
            self.integral_min = None
            self.integral_max = None
        else:
            integral_min, integral_max = self.integral_limits

            self.integral_min = float(integral_min)
            self.integral_max = float(integral_max)

            if self.integral_min > self.integral_max:
                raise ValueError(
                    "integral_limits must satisfy minimum <= maximum"
                )

        self.reset()

    def reset(self):
        """Reset accumulated PID state."""

        self.integral = 0.0
        self.previous_error = None

    def update(self, error, dt):
        """Advance the PID controller by one timestep.

        Parameters
        ----------
        error : float
            Current scalar control error.

        dt : float
            Positive controller timestep [s].

        Returns
        -------
        output : float
            Saturated controller output.

        diagnostics : dict
            PID terms and controller internal state.
        """

        error = float(error)
        dt = float(dt)

        if not np.isfinite(error):
            raise ValueError(
                "error must be finite"
            )

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "dt must be finite and strictly positive"
            )

        proportional = (
            self.kp
            * error
        )

        self.integral += (
            error
            * dt
        )

        if self.integral_limits is not None:
            self.integral = float(
                np.clip(
                    self.integral,
                    self.integral_min,
                    self.integral_max,
                )
            )

        integral_term = (
            self.ki
            * self.integral
        )

        if self.previous_error is None:
            error_derivative = 0.0
        else:
            error_derivative = (
                error
                - self.previous_error
            ) / dt

        derivative = (
            self.kd
            * error_derivative
        )

        unsaturated_output = (
            proportional
            + integral_term
            + derivative
        )

        output = float(
            np.clip(
                unsaturated_output,
                self.output_min,
                self.output_max,
            )
        )

        self.previous_error = (
            error
        )

        diagnostics = {
            "error":
                error,

            "proportional":
                proportional,

            "integral_state":
                self.integral,

            "integral":
                integral_term,

            "error_derivative":
                error_derivative,

            "derivative":
                derivative,

            "unsaturated_output":
                unsaturated_output,

            "output":
                output,

            "saturated":
                not np.isclose(
                    output,
                    unsaturated_output,
                ),
        }

        return (
            output,
            diagnostics,
        )
