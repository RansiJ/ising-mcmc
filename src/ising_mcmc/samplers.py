"""Single-spin MCMC updates for the two-dimensional Ising model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .model import delta_energy, neighbour_sum, validate_lattice


def _validate_inputs(
    lattice: NDArray[np.integer],
    beta: float,
    J: float,
    h: float,
    rng: np.random.Generator,
) -> None:
    validate_lattice(lattice)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative.")
    if not np.isfinite(J) or not np.isfinite(h):
        raise ValueError("J and h must be finite.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator.")


def metropolis_sweep(
    lattice: NDArray[np.integer],
    beta: float,
    rng: np.random.Generator,
    J: float = 1.0,
    h: float = 0.0,
) -> None:
    """Apply ``L**2`` random-site Metropolis updates in place.

    Sites are sampled with replacement, so one sweep is a conventional unit of
    computational effort rather than a guarantee that every site is visited.
    """

    _validate_inputs(lattice, beta, J, h, rng)
    L = lattice.shape[0]

    for _ in range(lattice.size):
        i = int(rng.integers(L))
        j = int(rng.integers(L))
        energy_change = delta_energy(lattice, i, j, J=J, h=h)
        if energy_change <= 0.0 or rng.random() < np.exp(-beta * energy_change):
            lattice[i, j] *= -1


def gibbs_sweep(
    lattice: NDArray[np.integer],
    beta: float,
    rng: np.random.Generator,
    J: float = 1.0,
    h: float = 0.0,
) -> None:
    """Apply ``L**2`` random-site heat-bath/Gibbs updates in place."""

    _validate_inputs(lattice, beta, J, h, rng)
    L = lattice.shape[0]

    for _ in range(lattice.size):
        i = int(rng.integers(L))
        j = int(rng.integers(L))
        local_field = J * neighbour_sum(lattice, i, j) + h
        exponent = 2.0 * beta * local_field

        # Algebraically identical to 1 / (1 + exp(-exponent)), but stable for
        # unusually large fields supplied by a caller.
        if exponent >= 0.0:
            probability_up = 1.0 / (1.0 + np.exp(-exponent))
        else:
            exp_exponent = np.exp(exponent)
            probability_up = exp_exponent / (1.0 + exp_exponent)

        lattice[i, j] = 1 if rng.random() < probability_up else -1
