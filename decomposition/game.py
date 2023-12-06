from typing import List

import numpy as np

from decomposition.payoff import Payoff
from decomposition.structure import Structure


class Game:
    """Class that handles payoff matrices and access decomposition functionality"""

    def __init__(self, n_actions: List[int], save_load: bool = True, path: str = None):
        """initialize game class

        Args:
            n_actions (List[int]): list with number of actions for each player
            save_load (bool, optional): save/load computed structure to save time. Defaults to True.
            path (str): path where structure is saved
        """
        self.n_agents = len(n_actions)
        self.agents = list(range(self.n_agents))
        self.n_actions = n_actions
        self.actions = list(range(self.n_agents))
        self.structure = Structure(self.n_actions, save_load=save_load, path=path)

    def __repr__(self) -> str:
        return f"Game({self.n_actions})"

    def compute_decomposition_matrix(self, payoff_matrices: List[np.ndarray]):
        """compute Hodge decomposition given payoff matrices"""
        # test input
        assert self.n_agents == len(payoff_matrices)
        assert self.n_actions == list(payoff_matrices[0].shape)

        # decomposition
        payoff_vector = self.matrices_to_vector(payoff_matrices)
        self.compute_decomposition(payoff_vector)

    def compute_decomposition(self, payoff_vector: List[float]):
        """compute Hodge decomposition given payoff vector"""
        # test input
        assert len(payoff_vector) == self.n_agents * np.prod(
            self.n_actions
        ), f"expected {self.n_agents * np.prod(self.n_actions)} entries, found {len(payoff_vector)}"

        # compute decomposition
        self.payoff = Payoff(self.structure, payoff_vector)

        # save results to matrix game:
        self.payoff_matrices = self.vector_to_matrices(payoff_vector)
        self.payoff_matrices_P = self.vector_to_matrices(self.payoff.uP)
        self.payoff_matrices_H = self.vector_to_matrices(self.payoff.uH)
        self.payoff_matrices_N = self.vector_to_matrices(self.payoff.uN)
        self.potential = self.payoff.potential
        self.potentialness = self.payoff.potentialness
        self.decomposition_computed = True

    def create_game_potentialness(potentialness: float) -> List[np.ndarray]:
        """create a new game with given level of potentialness as convex combination of components uH and uP"""
        if not self.decomposition_computed:
            print("compute decomposition first!")
            return None
        else:
            new_payoff = self.payoff.create_payoff_potentialness(potentialness)
            return self.vector_to_matrices(new_payoff)

    def matrices_to_vector(self, matrices: List[np.ndarray]) -> list:
        """transform given payoff matrices to payoff vector

        Args:
            matrices (List[np.ndarray]): list of (payoff) matrices

        Returns:
            list: list of all entries of (payoff) matrices
        """
        return list(np.hstack([m.flatten() for m in matrices]))

    def vector_to_matrices(self, vector: list) -> List[np.ndarray]:
        """transform given vector to (payoff) matrices

        Args:
            vector (list): list of all entries of (payoff) matrices
            n_actions (List[int]): list with number of actions for each player

        Returns:
            List[np.ndarray]: list of (payoff) matrices
        """
        return [
            vector[
                agent * np.prod(self.n_actions) : (agent + 1) * np.prod(self.n_actions)
            ]
            for agent in self.agents
        ]