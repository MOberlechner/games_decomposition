import itertools
import math as mt
import os
from itertools import combinations
from time import time
from typing import List

import numpy as np
import numpy.linalg as npla


def two_tuples_differ_for_one_element(tuple_one: tuple, tuple_two: tuple) -> bool:
    """check if two tuples differ e

    Args:
        tuple_one (tuple): node 1
        tuple_two (tuple): node 2

    Returns:
        bool: tuples differ in exactly one entry
    """
    assert len(tuple_one) == len(tuple_two), "Tuples must have the same length"
    count_zeros = 0
    for i in range(len(tuple_one)):
        if tuple_one[i] == tuple_two[i]:
            count_zeros += 1
    return count_zeros == len(tuple_one) - 1


def different_index(edge):
    a, b = edge
    i = 0
    while a[i] == b[i]:
        i += 1
    return i


class Player:
    """
    Player class. Each player has
    - name i
    - number of strategies Ai
    - strategies [1, 2, ..., Ai]
    """

    def __init__(self, num_strategies, player_name):
        self.num_strategies = num_strategies
        self.strategies = [i for i in range(1, num_strategies + 1)]
        self.player_name = player_name


class Structure:
    def __init__(
        self,
        num_strategies_for_player: List[int],
        flow_only: bool = False,
    ):
        """create structure induced by game for decomposition

        Args:
            num_strategies_for_player (List[int]): contains number of actions of each player
            save_load (bool, optional): save/load matrices. Defaults to True.
            path (str, optional): specify path to project. If None it looks in the current path for the name "games_decomposition". If the projects name is different, this might not work. Defaults to None.
        """
        self.label_game = f"Structure({num_strategies_for_player})"
        self.num_strategies_for_player = num_strategies_for_player
        self.num_players = len(num_strategies_for_player)
        self.flow_only = flow_only

    def create_structure(self, flow_only: bool = False):
        """Create structure, i.e., matrices that depend only on the number of agents and actions,
        and not on the payoffs

        Args:
            flow_only (bool, optional): For the flow decomposition we only need the pwc matrix. Defaults to False.
        """
        t0 = time()

        t0_init = time()
        # List of Player instances
        self.players = []
        for i in range(self.num_players):
            self.players.append(Player(self.num_strategies_for_player[i], i + 1))

        # A, that is number of strategis profiles, that is number of nodes of the response graph
        self.num_strategy_profiles = int(mt.prod(self.num_strategies_for_player))

        # AN, that is dimension of payoff space
        self.num_payoffs = self.num_strategy_profiles * self.num_players

        # Curly_A set, that is set of strategies profiles, of cardinality A
        # e.g. for 2x2 game it looks like [(1,1), (1,2), (2,1), (2,2)]
        self.strategy_profiles = list(
            itertools.product(*[p.strategies for p in self.players])
        )

        self.payoff_basis = self.get_payoff_basis()
        t1_init = time()

        t0_response_graph = time()
        # Make Response Graph
        self.make_response_graph()
        t1_response_graph = time()

        # Simplicial complexes terminology
        self.dim_C0 = self.num_strategy_profiles
        self.dim_C1 = self.num_edges
        self.dim_C0N = self.num_payoffs

        # PWC MATRIX C0N --> C1
        t0_pwc_matrix = time()
        self.pwc_matrix = self.make_pwc_matrix()
        t1_pwc_matrix = time()

        # MATRIX coboundary 0 map: d_0: C^0 --> C^1
        t0_coboundary_0_matrix = time()
        self.coboundary_0_matrix = self.make_coboundary_0_matrix()
        t1_coboundary_0_matrix = time()

        # pinv(δ_0): C^1 --> C^0
        t0_coboundary_0_matrix_pinv = time()
        self.coboundary_0_matrix_pinv = npla.pinv(self.coboundary_0_matrix)
        t1_coboundary_0_matrix_pinv = time()

        # e: C1 --> C1 projection onto exact
        t0_exact_projection = time()
        self.exact_projection = np.matmul(
            self.coboundary_0_matrix, self.coboundary_0_matrix_pinv
        )
        t1_exact_projection = time()

        if not flow_only:
            # Moore-Penrose pseudo-inverse of pwc
            t0_pwc_matrix_pinv = time()
            self.pwc_matrix_pinv = npla.pinv(self.pwc_matrix)
            t1_pwc_matrix_pinv = time()

            # PI: C0N --> C0N projection onto Euclidean orthogonal complement of ker(δ_0^N)
            t0_normalization_projection = time()
            self.normalization_projection = np.matmul(
                self.pwc_matrix_pinv, self.pwc_matrix
            )
            t1_normalization_projection = time()

            t0_potential = time()
            self.potential = np.matmul(self.coboundary_0_matrix_pinv, self.pwc_matrix)
            t1_potential = time()
            t1 = time()

        self.time = {
            "total": t1 - t0,
            "init": t1_init - t0_init,
            "response_graph": t1_response_graph - t0_response_graph,
            "pwc_matrix": t1_pwc_matrix - t0_pwc_matrix,
            "coboundary_0_matrix": t1_coboundary_0_matrix - t0_coboundary_0_matrix,
            "coboundary_0_matrix_pinv": t1_coboundary_0_matrix_pinv
            - t0_coboundary_0_matrix_pinv,
            "exact_projection": t1_exact_projection - t0_exact_projection,
            "pwc_matrix_pinv": t1_pwc_matrix_pinv - t0_pwc_matrix_pinv,
            "normalization_projection": t1_normalization_projection
            - t0_normalization_projection,
            "potential": t1_potential - t0_potential,
        }
        self.dim = {
            "dim_CO": self.dim_C0,
            "dim_CON": self.dim_C0N,
            "dim_C1": self.dim_C1,
            "pwc_matrix": self.pwc_matrix.shape,
            "pwc_matrix_pinv": self.pwc_matrix_pinv.shape,
            "coboundary_0_matrix": self.coboundary_0_matrix.shape,
            "coboundary_0_matrix_pinv": self.coboundary_0_matrix_pinv.shape,
            "exact_projection": self.exact_projection.shape,
            "normalization_projection": self.normalization_projection.shape,
        }

    def get_payoff_basis(self):
        """create basis of (C^0)^N of cardinality AN, i.e. basis of vector space of payoffs
        Its elements are e = (i,a) for i in N for a in A

        e.g. for 2x2 it looks like
        [[0, (1, 1)], [0, (1, 2)], [0, (2, 1)], [0, (2, 2)],
         [1, (1, 1)], [1, (1, 2)], [1, (2, 1)], [1, (2, 2)]]
        """
        return [
            (i.player_name - 1, a) for i in self.players for a in self.strategy_profiles
        ]

    # Make response graph dictionary, unoriented
    def make_response_graph(self):
        """Create response graph"""

        self.num_nodes = self.num_strategy_profiles
        self.num_edges_per_node = sum(self.num_strategies_for_player) - self.num_players
        self.num_edges = int(self.num_nodes * self.num_edges_per_node / 2)

        self.nodes = self.strategy_profiles  # list [a, b, c]
        self.graph = self.make_graph()  # dictionary graph[a] = [b, c, d]
        self.edges = self.make_edges()  # list [ [a,b], [c,d] ]
        self.sort_elementary_chains(self.edges)

    def make_graph(self):
        """
        Format: dictionary such that each key is a node and each value is the list of connected nodes
        e.g. graph[a] = [b, c, d] where a, b, c, d are nodes and [ab], [ac], [ad] are edges
        """
        graph = {}

        for s1 in self.strategy_profiles:
            unilateral_displacements = [
                s2
                for s2 in self.strategy_profiles
                if two_tuples_differ_for_one_element(s1, s2)
            ]
            graph[s1] = unilateral_displacements

        return graph

    def make_edges(self):
        """Create all undirected edges in this graph
        Format: list of lists, e.g. [ [a,b], [c,d] ] with [a,b] and [c,d] edges
        """
        edges = []
        for node1, neighbors in self.graph.items():
            for node2 in neighbors:
                if [node1, node2] not in edges and [node2, node1] not in edges:
                    edges.append([node1, node2])
        assert len(edges) == self.num_edges
        return edges

    def make_pwc_matrix(self):
        """Matrix of pwc: C^O^N --> C^1"""
        A = np.zeros([int(self.dim_C1), int(self.dim_C0N)])

        for row in range(int(self.dim_C1)):
            edge = self.edges[row]
            i = different_index(edge)

            minus_column = self.payoff_basis.index((i, edge[0]))
            plus_column = self.payoff_basis.index((i, edge[1]))
            A[row][minus_column] = -1
            A[row][plus_column] = +1

        return A

    def make_coboundary_0_matrix(self):
        """Matrix of d_0: C^0 --> C^1"""

        # Start with transpose
        A = np.zeros([int(self.dim_C1), int(self.dim_C0)])

        for row in range(int(self.dim_C1)):
            basis_edge = self.edges[row]

            minus_node, plus_node = basis_edge
            minus_column = self.nodes.index(minus_node)
            plus_column = self.nodes.index(plus_node)
            A[row][minus_column] = -1
            A[row][plus_column] = +1

        return A

    def sort_elementary_chains(self, list_of_simplices):
        for simplex in list_of_simplices:
            simplex.sort()

    def print_time(self):
        if hasattr(self, "time"):
            print("Runtime")
            print("-" * 41)
            for key, value in self.time.items():
                print(f"{key}".ljust(30), f"{value:.2f}s".rjust(10))
            print("")

    def print_dim(self):
        if hasattr(self, "dim"):
            print("Dimensions")
            print("-" * 51)
            for key, value in self.dim.items():
                print(f"{key}".ljust(30), f"{value}".rjust(20))

    # ----------------------------------------------------------------------------------- #
    #         Methods that that allow computed components to be stored/loaded             #
    # ----------------------------------------------------------------------------------- #

    def set_path(self, save_load: bool, path: str):
        """set path to save or load computed matrices

        Args:
            save_load (bool): save/load matrices
            path (str): directory where to save
        """
        if not save_load:
            return

        # set path
        if path is None:
            path = os.path.join(*os.getcwd())
        self.path = os.path.join(path, "data_decomp/structure/", self.label_game)

        # create directory (if does not exist)
        os.makedirs(self.path, exist_ok=True)

    def structure_exists(self):
        """check if a game exists"""
        for matrix in self.relevant_matrices:
            if not os.path.exists(os.path.join(self.path, f"{matrix}.npy")):
                return False
        return True

    def save_structure(self):
        """Save relevant matrices of game"""
        for matrix in self.relevant_matrices:
            np.save(os.path.join(self.path, matrix), getattr(self, matrix))

    def load_structure(self):
        """Load relevant matrices of saved game"""
        for matrix in self.relevant_matrices:
            try:
                setattr(self, matrix, np.load(os.path.join(self.path, f"{matrix}.npy")))
            except:
                print("File not found. Matrices are computed instead.")
                self.create_game()


if __name__ == "__main__":
    structure = Structure([8, 8, 8])
    structure.create_structure()
    structure.print_time()
    structure.print_dim()
