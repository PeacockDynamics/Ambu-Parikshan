"""Vehicle models for HydroLab-3D."""

from .uuv import (
    current_generalized_acceleration,
    integrate_six_dof_euler,
    simulate_six_dof,
    six_dof_state_derivative,
)
from .faults import apply_wrench_fault

__all__ = [
    "current_generalized_acceleration",
    "integrate_six_dof_euler",
    "simulate_six_dof",
    "six_dof_state_derivative",
    "apply_wrench_fault",
]
