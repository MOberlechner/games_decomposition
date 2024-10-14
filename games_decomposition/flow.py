import numpy as np

from games_decomposition.structure import Structure


class Flow:
    def __init__(self, structure: Structure, payoff_vector: list):
        """_summary_

        Args:
            game (Game): instance of Game class
            payoff_vector (list): list of payoffs
        """

        self.structure = structure
        self.payoff_vector = payoff_vector

        self.Du, self.DuP, self.DuH = self.decompose_flow()
        self.potentialness = self.compute_potentialness()

    def decompose_flow(self):
        """compute hodge decomposition without"""
        u = self.payoff_vector

        e = self.structure.exact_projection
        PWC = self.structure.pwc_matrix

        Du = PWC @ u
        DuP = e @ Du.T
        DuH = Du - DuP.T

        result = [Du, DuP, DuH]
        return [np.array(u).flatten() for u in result]

    def compute_potentialness(self):
        """compute |uP| / ( |uP|+|uH|)"""
        return np.linalg.norm(self.DuP) / (
            np.linalg.norm(self.DuP) + np.linalg.norm(self.DuH)
        )
