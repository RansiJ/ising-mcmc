"""Core definitions for the two-dimensional ferromagnetic Ising model.

The model uses a square ``L x L`` lattice with periodic boundary conditions and
spins in ``{-1, +1}``.  The Hamiltonian is

    H(s) = -J * sum_<ij> s_i s_j - h * sum_i s_i,

where each nearest-neighbour bond is counted exactly once.

This module deliberately contains no MCMC logic.  Keeping the physical model
separate from the samplers makes it possible to test energy changes and
observables independently before introducing stochastic transition kernels.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

SpinLattice = NDArray[np.int8]
MIN_LATTICE_SIZE: Final[int] = 2


def _validate_size(L: int) -> None:
    if isinstance(L, bool) or not isinstance(L, (int, np.integer)):
        raise TypeError("L must be an integer.")
    if L < MIN_LATTICE_SIZE:
        raise ValueError(f"L must be at least {MIN_LATTICE_SIZE}.")


def _validate_lattice_shape(lattice: NDArray[np.integer]) -> None:
    """Validate only lattice geometry; this check is O(1)."""

    array = np.asarray(lattice)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("lattice must be a square two-dimensional array.")
    _validate_size(array.shape[0])


def validate_lattice(lattice: NDArray[np.integer]) -> None:
    """Validate the representation used throughout the project.

    Parameters
    ----------
    lattice:
        Square two-dimensional array whose entries are exactly ``-1`` or ``+1``.

    Raises
    ------
    ValueError
        If the array is not square, is too small, or contains invalid spins.
    """

    array = np.asarray(lattice)
    _validate_lattice_shape(array)
    if not np.all((array == -1) | (array == 1)):
        raise ValueError("lattice entries must be either -1 or +1.")


def random_lattice(L: int, rng: np.random.Generator) -> SpinLattice:
    """Generate a reproducible random spin configuration."""

    _validate_size(L)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator.")
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))


def ordered_lattice(L: int, spin: int = 1) -> SpinLattice:
    """Generate a fully ordered configuration with all spins equal."""

    _validate_size(L)
    if spin not in (-1, 1):
        raise ValueError("spin must be either -1 or +1.")
    return np.full((L, L), spin, dtype=np.int8)


def neighbour_sum(lattice: NDArray[np.integer], i: int, j: int) -> int:
    """Return the four-neighbour spin sum at ``(i, j)`` with periodic boundaries."""

    _validate_lattice_shape(lattice)
    L = lattice.shape[0]
    if not (0 <= i < L and 0 <= j < L):
        raise IndexError("spin index is outside the lattice.")

    return int(
        lattice[(i + 1) % L, j]
        + lattice[(i - 1) % L, j]
        + lattice[i, (j + 1) % L]
        + lattice[i, (j - 1) % L]
    )


def total_energy(
    lattice: NDArray[np.integer],
    J: float = 1.0,
    h: float = 0.0,
) -> float:
    """Compute the Hamiltonian of a configuration.

    Only the right and downward neighbours are included in the interaction sum,
    so every periodic nearest-neighbour bond is counted once.
    """

    validate_lattice(lattice)
    array = np.asarray(lattice, dtype=np.int64)

    right = np.roll(array, shift=-1, axis=1)
    down = np.roll(array, shift=-1, axis=0)
    interaction_sum = np.sum(array * right + array * down, dtype=np.int64)
    field_sum = np.sum(array, dtype=np.int64)

    return float(-J * interaction_sum - h * field_sum)


def delta_energy(
    lattice: NDArray[np.integer],
    i: int,
    j: int,
    J: float = 1.0,
    h: float = 0.0,
) -> float:
    """Return the exact energy change produced by flipping one spin."""

    _validate_lattice_shape(lattice)
    L = lattice.shape[0]
    if not (0 <= i < L and 0 <= j < L):
        raise IndexError("spin index is outside the lattice.")

    spin = int(lattice[i, j])
    if spin not in (-1, 1):
        raise ValueError("selected lattice entry must be either -1 or +1.")
    local_field = J * neighbour_sum(lattice, i, j) + h
    return float(2.0 * spin * local_field)


def magnetization(lattice: NDArray[np.integer]) -> int:
    """Return total magnetization ``M = sum_i s_i``."""

    validate_lattice(lattice)
    return int(np.sum(lattice, dtype=np.int64))


def energy_density(
    lattice: NDArray[np.integer],
    J: float = 1.0,
    h: float = 0.0,
) -> float:
    """Return energy per spin."""

    validate_lattice(lattice)
    return total_energy(lattice, J=J, h=h) / lattice.size


def magnetization_density(lattice: NDArray[np.integer]) -> float:
    """Return magnetization per spin."""

    validate_lattice(lattice)
    return magnetization(lattice) / lattice.size
