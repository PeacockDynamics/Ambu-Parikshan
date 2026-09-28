"""Deterministic actuator fault primitives, independent of rendering."""

import numpy as np


def apply_wrench_fault(action, *, scale=1.0, stuck_action=None, failed_axes=()):
    """Apply explicit degradation, stuck command, and axis-failure semantics."""
    action = np.asarray(action, dtype=float)
    if action.shape != (6,) or not np.all(np.isfinite(action)):
        raise ValueError("action must be a finite vector with shape (6,)")
    scale = float(scale)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    result = action.copy() if stuck_action is None else np.asarray(stuck_action, dtype=float).copy()
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError("stuck_action must be a finite vector with shape (6,)")
    result *= scale
    for axis in failed_axes:
        if not isinstance(axis, int) or not 0 <= axis < 6:
            raise ValueError("failed_axes must contain indices from 0 through 5")
        result[axis] = 0.0
    return result
