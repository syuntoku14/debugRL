import numpy as np
from tqdm import tqdm
from .core import Solver
from debug_rl.utils import (
    eps_greedy_policy,
    softmax_policy,
)


class OracleSolver(Solver):
    def solve(self):
        """
        Do value iteration from the end of the episode.

        Returns:
            SxA matrix
        """
        self.init_history()
        values = np.zeros((self.dS, self.dA))  # SxA
        error = np.inf
        # start training
        for k in tqdm(range(self.solve_options["num_trains"])):
            new_values = self.backup(values)  # SxA
            error = np.abs(values - new_values).max()
            values = new_values
            self.record_performance(k, values)
            if error < 1e-9:
                break
        self.record_performance(k, values, force=True)

        print("Iteration finished with maximum error: ", error)


class OracleViSolver(OracleSolver):
    def backup(self, curr_q_val):
        # Bellman Operator to update q values
        discount = self.solve_options["discount"]
        curr_v_val = np.max(curr_q_val, axis=-1)  # S
        prev_q = self.trr_rew_sum \
            + discount*(self.transition_matrix *
                        curr_v_val).reshape(self.dS, self.dA)
        return prev_q

    def compute_policy(self, q_values, eps_greedy=0.0):
        # return epsilon-greedy policy
        return eps_greedy_policy(q_values, eps_greedy=eps_greedy)


class OracleCviSolver(OracleSolver):
    """
    Solver of conservative value iteration.
    This implementation is based on the paper:
    http://proceedings.mlr.press/v89/kozuno19a.html.
    """

    def backup(self, curr_pref):
        discount = self.solve_options["discount"]
        alpha = self.solve_options["alpha"]
        beta = self.solve_options["beta"]
        mP = self.max_operator(curr_pref, beta)
        prev_preference = \
            alpha * (curr_pref - mP.reshape(-1, 1)) \
            + self.trr_rew_sum \
            + discount*(self.transition_matrix * mP).reshape(self.dS, self.dA)
        return prev_preference

    def compute_policy(self, preference):
        # return softmax policy
        beta = self.solve_options["beta"]
        return softmax_policy(preference, beta=beta)
