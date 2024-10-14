import numpy as np

from games_decomposition.structure import Structure


class Payoff:
    def __init__(self, structure: Structure, payoff_vector: list):
        """_summary_

        Args:
            game (Game): instance of Game class
            payoff_vector (list): list of payoffs
        """

        self.structure = structure
        self.payoff_vector = payoff_vector

        self.uN, self.uP, self.uH, self.potential = self.decompose_payoff()
        self.u_strategic = self.uP + self.uH
        self.potentialness = self.compute_potentialness()

    def decompose_payoff(self):
        """compute hodge decomposition"""

        u = self.payoff_vector

        PI = self.structure.normalization_projection
        e = self.structure.exact_projection

        PWC_pinv = self.structure.pwc_matrix_pinv
        PWC = self.structure.pwc_matrix
        delta_0_pinv = self.structure.coboundary_0_matrix_pinv
        delta_0 = self.structure.coboundary_0_matrix

        uN = u - PI @ u
        uP = PWC_pinv @ e @ PWC @ u
        uH = u - uN - uP
        phi = delta_0_pinv @ PWC @ u

        result = [uN, uP, uH, phi]
        return [np.array(u).flatten() for u in result]

    def compute_potentialness(self):
        """compute |uP| / ( |uP|+|uH|)"""
        return np.linalg.norm(self.uP) / (
            np.linalg.norm(self.uP) + np.linalg.norm(self.uH)
        )

    def determine_alpha(self, potentialness: float) -> float:
        """determina alpha to get level of potentialness"""
        if potentialness is None:
            return 1 / 2
        else:
            assert 0 <= potentialness <= 1
            return (
                potentialness
                * np.linalg.norm(self.uH)
                / (
                    (1 - potentialness) * np.linalg.norm(self.uP)
                    + potentialness * np.linalg.norm(self.uH)
                )
            )

    def create_payoff_potentialness(self, potentialness: float) -> np.ndarray:
        """create new game  with given level of potentialness
        game is created as convex combination of uP and uH: (1-alpha) * uH + alpha * uP,
        where the alpha is choosen such that given level of potentialness is achieved

        Args:
            potentialness (float): given level of potentialness

        Returns:
            np.ndarray: payoff_vector
        """
        assert 0 <= potentialness <= 1
        alpha = self.determine_alpha(potentialness)
        return (1 - alpha) * self.uH + alpha * self.uP
