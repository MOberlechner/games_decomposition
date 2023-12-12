"""
This module tests the decomposition
"""
import numpy as np
import pytest

from decomposition.payoff import Payoff
from decomposition.structure import Structure

settings = [[2, 2], [3, 3], [4, 4], [2, 3], [3, 4], [2, 2, 2], [2, 3, 4], [6, 6]]


@pytest.mark.parametrize("n_actions", settings)
def test_decomposition_random(n_actions):

    for _ in range(5):

        # create structure for setting
        n_agents = len(n_actions)
        structure = Structure(n_actions, save_load=False)

        # compute decomposition for random instance
        payoff_vector = np.random.rand(n_agents * np.prod(n_actions))
        payoff = Payoff(structure, payoff_vector)
        assert np.allclose(payoff.payoff_vector, payoff_vector)
        assert np.allclose(
            payoff.payoff_vector, payoff.uN + payoff.uH + payoff.uP
        ), f"decomposition does not sum up to original payoff ({n_actions})"

        payoff_vector_H = payoff.uH.copy()
        payoff_vector_P = payoff.uP.copy()

        # check decomposition of harmonic game
        payoff = Payoff(structure, payoff_vector_H)
        assert np.allclose(payoff_vector_H, payoff.uH)
        assert np.allclose(payoff.uP, 0.0)
        assert np.isclose(payoff.compute_potentialness(), 0.0)

        # check decomposition of potential game
        payoff = Payoff(structure, payoff_vector_P)
        assert np.allclose(payoff_vector_P, payoff.uP)
        assert np.allclose(payoff.uH, 0.0)
        assert np.isclose(payoff.compute_potentialness(), 1.0)
