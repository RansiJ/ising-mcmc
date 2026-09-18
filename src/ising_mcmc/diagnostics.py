"""Small diagnostics for correlated scalar MCMC series."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def autocorrelation(
    values: ArrayLike,
    max_lag: int | None = None,
) -> NDArray[np.float64]:
    """Return the normalized sample autocorrelation from lag zero onward."""

    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size < 2:
        raise ValueError("values must be a one-dimensional series of length >= 2.")
    if not np.all(np.isfinite(series)):
        raise ValueError("values must contain only finite numbers.")

    if max_lag is None:
        max_lag = series.size - 1
    if isinstance(max_lag, bool) or not isinstance(max_lag, (int, np.integer)):
        raise TypeError("max_lag must be an integer.")
    if not 0 <= max_lag < series.size:
        raise ValueError("max_lag must satisfy 0 <= max_lag < len(values).")

    centered = series - np.mean(series)
    variance = float(np.dot(centered, centered) / series.size)
    if variance == 0.0:
        raise ValueError("autocorrelation is undefined for a constant series.")

    result = np.empty(max_lag + 1, dtype=np.float64)
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        result[lag] = np.mean(centered[:-lag] * centered[lag:]) / variance
    return result


def integrated_autocorrelation_time(
    values: ArrayLike,
    max_lag: int | None = None,
) -> float:
    """Estimate integrated autocorrelation time using the initial positive sum."""

    series = np.asarray(values, dtype=np.float64)
    if max_lag is None:
        max_lag = min(series.size - 1, max(1, series.size // 4))
    correlations = autocorrelation(series, max_lag=max_lag)
    positive = correlations[1:]
    first_nonpositive = np.flatnonzero(positive <= 0.0)
    if first_nonpositive.size:
        positive = positive[: int(first_nonpositive[0])]
    return float(max(1.0, 1.0 + 2.0 * np.sum(positive)))


def effective_sample_size(
    values: ArrayLike,
    max_lag: int | None = None,
) -> float:
    """Estimate the number of independent samples represented by a series."""

    series = np.asarray(values, dtype=np.float64)
    tau = integrated_autocorrelation_time(series, max_lag=max_lag)
    return float(series.size / tau)
