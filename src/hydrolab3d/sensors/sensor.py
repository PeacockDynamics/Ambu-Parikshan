"""Common sensor-measurement utilities for HydroLab-3D."""

import numpy as np


def measure_with_bias_noise(
    truth,
    *,
    bias=0.0,
    noise_std=0.0,
    rng=None,
):
    """
    Apply deterministic bias and Gaussian measurement noise.

    Parameters
    ----------
    truth : array-like
        Ideal scalar or one-dimensional measurement.

    bias : scalar or array-like, optional
        Deterministic additive bias.

    noise_std : scalar or array-like, optional
        Gaussian noise standard deviation.

    rng : numpy.random.Generator or None, optional
        Explicit random generator. Required when noise is nonzero.

    Returns
    -------
    np.ndarray
        Measured value after bias and Gaussian noise.

    Raises
    ------
    ValueError
        If inputs have invalid shape, contain non-finite values,
        are not broadcast-compatible, or noise_std is negative.

    TypeError
        If stochastic noise is requested without a valid
        numpy.random.Generator.
    """
    truth = np.asarray(
        truth,
        dtype=float,
    )

    bias = np.asarray(
        bias,
        dtype=float,
    )

    noise_std = np.asarray(
        noise_std,
        dtype=float,
    )

    if truth.ndim > 1:
        raise ValueError(
            "truth must be a scalar or one-dimensional array"
        )

    if not np.all(np.isfinite(truth)):
        raise ValueError(
            "truth must contain only finite values"
        )

    if not np.all(np.isfinite(bias)):
        raise ValueError(
            "bias must contain only finite values"
        )

    if not np.all(np.isfinite(noise_std)):
        raise ValueError(
            "noise_std must contain only finite values"
        )

    if np.any(noise_std < 0.0):
        raise ValueError(
            "noise_std must be non-negative"
        )

    try:
        truth_b, bias_b, std_b = np.broadcast_arrays(
            truth,
            bias,
            noise_std,
        )
    except ValueError as exc:
        raise ValueError(
            "truth, bias, and noise_std must be "
            "broadcast-compatible"
        ) from exc

    if np.any(std_b > 0.0):
        if rng is None:
            raise ValueError(
                "rng must be provided when noise_std is nonzero"
            )

        if not isinstance(
            rng,
            np.random.Generator,
        ):
            raise TypeError(
                "rng must be a numpy.random.Generator"
            )

        noise = rng.normal(
            loc=0.0,
            scale=std_b,
            size=truth_b.shape,
        )
    else:
        noise = np.zeros_like(
            truth_b,
            dtype=float,
        )

    return (
        truth_b
        + bias_b
        + noise
    )
