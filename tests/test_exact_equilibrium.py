import numpy as np
import pytest

from ising_mcmc.exact import enumerate_states, exact_distribution, exact_statistics
from ising_mcmc.model import total_energy


def test_enumeration_contains_every_binary_spin_state():
    L = 2
    states = enumerate_states(L)
    assert states.shape == (2 ** (L * L), L, L)
    assert np.unique(states.reshape(states.shape[0], -1), axis=0).shape[0] == states.shape[0]
    assert set(np.unique(states)) == {-1, 1}


def test_beta_zero_distribution_is_uniform_and_normalized():
    distribution = exact_distribution(L=2, beta=0.0)
    expected = np.full_like(
        distribution.probabilities,
        1.0 / distribution.probabilities.size,
    )
    np.testing.assert_allclose(distribution.probabilities, expected)
    assert np.sum(distribution.probabilities) == pytest.approx(1.0)


def test_zero_field_distribution_has_spin_flip_symmetry():
    stats = exact_statistics(L=3, beta=0.44, J=1.0, h=0.0)
    assert stats.mean_magnetization == pytest.approx(0.0, abs=1e-13)


def test_vectorized_exact_energy_agrees_with_core_hamiltonian():
    distribution = exact_distribution(L=2, beta=0.31, J=0.91, h=0.13)
    reference = np.array(
        [total_energy(state, J=0.91, h=0.13) for state in distribution.states]
    )
    np.testing.assert_allclose(distribution.energies, reference)


def test_exact_statistics_are_moments_of_returned_distribution():
    distribution = exact_distribution(L=3, beta=0.38, J=1.0, h=0.07)
    stats = exact_statistics(L=3, beta=0.38, J=1.0, h=0.07)

    p = distribution.probabilities
    E = distribution.energies
    M = distribution.magnetizations

    assert stats.mean_energy == pytest.approx(np.sum(p * E))
    assert stats.mean_energy_squared == pytest.approx(np.sum(p * E**2))
    assert stats.mean_magnetization == pytest.approx(np.sum(p * M))
    assert stats.mean_abs_magnetization == pytest.approx(np.sum(p * np.abs(M)))
    assert stats.mean_magnetization_squared == pytest.approx(np.sum(p * M**2))


def test_exact_enumeration_guard_blocks_large_state_spaces():
    with pytest.raises(ValueError):
        enumerate_states(L=5)
