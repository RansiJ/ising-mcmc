import numpy as np
import pytest

from ising_mcmc.model import delta_energy, ordered_lattice, random_lattice
from ising_mcmc.samplers import gibbs_sweep, metropolis_sweep


@pytest.mark.parametrize("sweep", [metropolis_sweep, gibbs_sweep])
def test_sweep_is_reproducible_with_generator_seed(sweep):
    initial = random_lattice(5, np.random.default_rng(11))
    first = initial.copy()
    second = initial.copy()

    sweep(first, beta=0.43, rng=np.random.default_rng(2026))
    sweep(second, beta=0.43, rng=np.random.default_rng(2026))

    np.testing.assert_array_equal(first, second)


def test_metropolis_at_beta_zero_accepts_every_proposal():
    lattice = ordered_lattice(3)
    expected = lattice.copy()
    expected_rng = np.random.default_rng(83)
    for _ in range(lattice.size):
        i = int(expected_rng.integers(3))
        j = int(expected_rng.integers(3))
        if delta_energy(expected, i, j) > 0.0:
            expected_rng.random()
        expected[i, j] *= -1

    metropolis_sweep(lattice, beta=0.0, rng=np.random.default_rng(83))

    np.testing.assert_array_equal(lattice, expected)


def test_gibbs_strong_positive_field_sets_selected_spins_up():
    lattice = ordered_lattice(4, spin=-1)
    gibbs_sweep(
        lattice,
        beta=10.0,
        rng=np.random.default_rng(5),
        J=0.0,
        h=10.0,
    )
    assert np.count_nonzero(lattice == 1) > 0
    assert set(np.unique(lattice)) <= {-1, 1}


@pytest.mark.parametrize("sweep", [metropolis_sweep, gibbs_sweep])
def test_sweep_rejects_invalid_beta_and_rng(sweep):
    lattice = ordered_lattice(2)
    with pytest.raises(ValueError):
        sweep(lattice, beta=-0.1, rng=np.random.default_rng(1))
    with pytest.raises(TypeError):
        sweep(lattice, beta=0.2, rng=None)
