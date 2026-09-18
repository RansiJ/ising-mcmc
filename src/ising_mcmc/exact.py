"""Exact finite-lattice reference calculations for the 2D Ising model.

The routines in this module enumerate every spin configuration and are intended
only for small lattices.  Their purpose is validation: future MCMC samplers can
be checked against an equilibrium distribution that does not depend on MCMC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .model import MIN_LATTICE_SIZE


@dataclass(frozen=True)
class ExactDistribution:
    """Complete Boltzmann distribution for a small finite lattice."""

    states: NDArray[np.int8]
    energies: NDArray[np.float64]
    magnetizations: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    log_partition: float


@dataclass(frozen=True)
class ExactStatistics:
    """Equilibrium moments derived from exact enumeration."""

    L: int
    beta: float
    J: float
    h: float
    log_partition: float
    mean_energy: float
    mean_energy_squared: float
    mean_magnetization: float
    mean_abs_magnetization: float
    mean_magnetization_squared: float

    @property
    def n_sites(self) -> int:
        return self.L * self.L

    @property
    def energy_density(self) -> float:
        return self.mean_energy / self.n_sites

    @property
    def abs_magnetization_density(self) -> float:
        return self.mean_abs_magnetization / self.n_sites

    @property
    def heat_capacity_per_spin(self) -> float:
        variance = self.mean_energy_squared - self.mean_energy**2
        return self.beta**2 * variance / self.n_sites

    @property
    def susceptibility_per_spin(self) -> float:
        variance = self.mean_magnetization_squared - self.mean_magnetization**2
        return self.beta * variance / self.n_sites


def _validate_exact_inputs(L: int, beta: float, max_sites: int) -> None:
    if isinstance(L, bool) or not isinstance(L, (int, np.integer)):
        raise TypeError("L must be an integer.")
    if L < MIN_LATTICE_SIZE:
        raise ValueError(f"L must be at least {MIN_LATTICE_SIZE}.")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative.")
    if isinstance(max_sites, bool) or not isinstance(max_sites, (int, np.integer)):
        raise TypeError("max_sites must be an integer.")
    if L * L > max_sites:
        raise ValueError(
            "Exact enumeration is intentionally restricted to small lattices; "
            f"received {L * L} sites with max_sites={max_sites}."
        )


def enumerate_states(L: int, max_sites: int = 16) -> NDArray[np.int8]:
    """Enumerate all ``2**(L*L)`` spin configurations for a small lattice."""

    _validate_exact_inputs(L=L, beta=0.0, max_sites=max_sites)
    n_sites = L * L
    labels = np.arange(1 << n_sites, dtype=np.uint64)[:, None]
    bit_positions = np.arange(n_sites, dtype=np.uint64)[None, :]
    bits = (labels >> bit_positions) & np.uint64(1)
    spins = (2 * bits.astype(np.int8) - 1).reshape(-1, L, L)
    return spins


def _energies_and_magnetizations(
    states: NDArray[np.int8],
    J: float,
    h: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Vectorized Hamiltonian evaluation independent of the MCMC implementation."""

    right = np.roll(states, shift=-1, axis=2)
    down = np.roll(states, shift=-1, axis=1)
    interaction = np.sum(states * right + states * down, axis=(1, 2), dtype=np.int64)
    magnetizations = np.sum(states, axis=(1, 2), dtype=np.int64)
    energies = (-J * interaction - h * magnetizations).astype(np.float64)
    return energies, magnetizations


def exact_distribution(
    L: int,
    beta: float,
    J: float = 1.0,
    h: float = 0.0,
    max_sites: int = 16,
) -> ExactDistribution:
    """Return the normalized finite-volume Boltzmann distribution."""

    _validate_exact_inputs(L=L, beta=beta, max_sites=max_sites)
    if not np.isfinite(J) or not np.isfinite(h):
        raise ValueError("J and h must be finite.")

    states = enumerate_states(L=L, max_sites=max_sites)
    energies, magnetizations = _energies_and_magnetizations(states, J=J, h=h)

    log_weights = -beta * energies
    shift = float(np.max(log_weights))
    shifted_weights = np.exp(log_weights - shift)
    normalizer = float(np.sum(shifted_weights))
    probabilities = shifted_weights / normalizer
    log_partition = shift + float(np.log(normalizer))

    return ExactDistribution(
        states=states,
        energies=energies,
        magnetizations=magnetizations,
        probabilities=probabilities.astype(np.float64),
        log_partition=log_partition,
    )


def exact_statistics(
    L: int,
    beta: float,
    J: float = 1.0,
    h: float = 0.0,
    max_sites: int = 16,
) -> ExactStatistics:
    """Compute exact equilibrium moments for a small finite lattice."""

    distribution = exact_distribution(
        L=L,
        beta=beta,
        J=J,
        h=h,
        max_sites=max_sites,
    )
    p = distribution.probabilities
    E = distribution.energies
    M = distribution.magnetizations.astype(np.float64)

    return ExactStatistics(
        L=L,
        beta=float(beta),
        J=float(J),
        h=float(h),
        log_partition=distribution.log_partition,
        mean_energy=float(np.sum(p * E)),
        mean_energy_squared=float(np.sum(p * E**2)),
        mean_magnetization=float(np.sum(p * M)),
        mean_abs_magnetization=float(np.sum(p * np.abs(M))),
        mean_magnetization_squared=float(np.sum(p * M**2)),
    )
