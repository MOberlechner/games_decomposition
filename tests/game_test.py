"""
This module tests the decomposition
"""
import numpy as np
import pytest

from games_decomposition.game import Game
from games_decomposition.structure import Structure


def test_basics():
    game = Game([2, 3, 4], save_load=False)
    assert isinstance(game, Game)
    assert game.n_agents == 3
    assert game.agents == [0, 1, 2]


def test_matrices_to_vector():
    # 2 x 2 game
    game = Game([2, 2], save_load=False)
    payoff_matrices = [np.array([[1, 2], [3, 4]]), np.array([[-1, -2], [-3, -4]])]
    vector = [1, 2, 3, 4, -1, -2, -3, -4]
    assert game.matrices_to_vector(payoff_matrices) == vector

    # 3 x 2 game
    game = Game([3, 2], save_load=False)
    payoff_matrices = [
        np.array([[1, 2], [3, 4], [5, 6]]),
        np.array([[-1, -2], [-3, -4], [-5, -6]]),
    ]
    vector = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]
    assert game.matrices_to_vector(payoff_matrices) == vector


settings = [[2, 2], [3, 3], [2, 3], [3, 4], [2, 2, 2], [2, 3, 4]]


@pytest.mark.parametrize("n_actions", settings)
def test_vector_to_matrix(n_actions):
    game = Game(n_actions, save_load=False)
    matrix = np.random.rand(*n_actions)
    vector = game.matrices_to_vector([matrix] * len(n_actions))
    assert np.array_equal(
        matrix, game.vector_to_matrices(vector)[0]
    ), f"setting {n_actions}"
