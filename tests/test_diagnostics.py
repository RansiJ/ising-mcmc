import numpy as np
import pytest

from ising_mcmc.diagnostics import (
    autocorrelation,
    effective_sample_size,
    integrated_autocorrelation_time,
)


def test_autocorrelation_has_unit_lag_zero():
    correlations = autocorrelation([1.0, -1.0, 2.0, 0.0], max_lag=2)
    assert correlations.shape == (3,)
    assert correlations[0] == pytest.approx(1.0)


def test_alternating_series_truncates_at_first_negative_correlation():
    series = np.tile([1.0, -1.0], 20)
    assert integrated_autocorrelation_time(series, max_lag=10) == 1.0
    assert effective_sample_size(series, max_lag=10) == len(series)


def test_constant_series_has_no_defined_autocorrelation():
    with pytest.raises(ValueError):
        autocorrelation(np.ones(10))
