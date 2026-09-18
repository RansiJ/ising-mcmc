import numpy as np
import pytest

from ising_mcmc.model import (
    delta_energy,
    magnetization,
    ordered_lattice,
    random_lattice,
    total_energy,
    validate_lattice,
)


def test_random_lattice_is_reproducible_with_generator_seed():
    first = random_lattice(5, np.random.default_rng(2026))
    second = random_lattice(5, np.random.default_rng(2026))
    np.testing.assert_array_equal(first, second)


def test_ordered_state_energy_matches_bond_count():
    L = 4
    J = 1.7
    h = -0.2
    lattice = ordered_lattice(L, spin=1)
    n_sites = L * L
    expected = -J * (2 * n_sites) - h * n_sites
    assert total_energy(lattice, J=J, h=h) == pytest.approx(expected)


@pytest.mark.parametrize("L", [2, 3, 5])
def test_delta_energy_matches_direct_energy_difference(L):
    rng = np.random.default_rng(741 + L)
    lattice = random_lattice(L, rng)
    J = 0.83
    h = -0.17

    for i in range(L):
        for j in range(L):
            before = total_energy(lattice, J=J, h=h)
            expected_delta = delta_energy(lattice, i, j, J=J, h=h)

            flipped = lattice.copy()
            flipped[i, j] *= -1
            after = total_energy(flipped, J=J, h=h)

            assert after - before == pytest.approx(expected_delta)


def test_magnetization_changes_by_twice_flipped_spin():
    rng = np.random.default_rng(91)
    lattice = random_lattice(4, rng)
    i, j = 1, 2
    spin_before = int(lattice[i, j])
    m_before = magnetization(lattice)

    lattice[i, j] *= -1
    m_after = magnetization(lattice)

    assert m_after - m_before == -2 * spin_before


@pytest.mark.parametrize(
    "bad_lattice",
    [
        np.ones((2, 3), dtype=np.int8),
        np.zeros((3, 3), dtype=np.int8),
        np.ones((1, 1), dtype=np.int8),
    ],
)
def test_invalid_lattices_are_rejected(bad_lattice):
    with pytest.raises(ValueError):
        validate_lattice(bad_lattice)
