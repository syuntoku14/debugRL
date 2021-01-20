import numpy as np
from tqdm import tqdm
from .core import Solver
from debug_rl.utils import (
    eps_greedy_policy,
    softmax_policy,
)


class OracleSolver(Solver):
    def run(self, num_steps=10000):
        """
        Do value iteration from the end of the episode.
        """
        values = self.values  # SxA
        # start training
        for _ in tqdm(range(num_steps)):
            new_values = self.backup(values)  # SxA
            error = np.abs(values - new_values).max()
            values = new_values
            self.record_performance(values)
            if error < 1e-9:
                break
            self.step += 1

        print("Iteration finished with maximum error: ", error)


class OracleViSolver(OracleSolver):
    def backup(self, curr_q_val):
        # Bellman Operator to update q values
        discount = self.solve_options["discount"]
        curr_v_val = np.max(curr_q_val, axis=-1)  # S
        prev_q = self.env.trans_rew_sum \
            + discount*(self.env.transition_matrix *
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
            + self.env.trans_rew_sum \
            + discount*(self.env.transition_matrix * mP).reshape(self.dS, self.dA)
        return prev_preference

    def compute_policy(self, preference):
        # return softmax policy
        beta = self.solve_options["beta"]
        return softmax_policy(preference, beta=beta)
