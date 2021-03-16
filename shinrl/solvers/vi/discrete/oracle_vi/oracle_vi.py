import numpy as np
from tqdm import tqdm
from .core import Solver
from shinrl.utils import (
    eps_greedy_policy,
    softmax_policy,
)


class OracleSolver(Solver):
    def run(self, num_steps=10000):
        values = self.tb_values  # SxA

        for _ in tqdm(range(num_steps)):
            self.record_history(np.asarray(values))

            # ----- update table -----
            new_values = self.backup(values)  # SxA
            error = np.abs(values - new_values).max()
            values = new_values
            self.record_scalar("Error", error)

            self.step += 1

    def record_history(self, values):
        self.record_array("Values", values)
        self.set_tb_policy(values)
        if self.step % self.solve_options["evaluation_interval"] == 0:
            expected_return = self.env.compute_expected_return(self.tb_policy)
            self.record_scalar("Return", expected_return, tag="Policy")


class OracleViSolver(OracleSolver):
    def backup(self, curr_q_val):
        discount = self.solve_options["discount"]
        curr_v_val = np.max(curr_q_val, axis=-1)  # S
        prev_q = self.env.reward_matrix \
            + discount*(self.env.transition_matrix *
                        curr_v_val).reshape(self.dS, self.dA)
        return prev_q

    def set_tb_policy(self, values):
        policy = eps_greedy_policy(values, eps_greedy=0.0)
        self.record_array("Policy", policy)


class OracleCviSolver(OracleSolver):
    """
    This implementation is based on the paper:
    http://proceedings.mlr.press/v89/kozuno19a.html.
    """

    def backup(self, curr_pref):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha, beta = kl_coef / (er_coef+kl_coef), 1 / (er_coef+kl_coef)
        mP = self.max_operator(curr_pref, beta)
        prev_preference = \
            alpha * (curr_pref - mP.reshape(-1, 1)) \
            + self.env.reward_matrix \
            + discount*(self.env.transition_matrix *
                        mP).reshape(self.dS, self.dA)
        return prev_preference

    def set_tb_policy(self, values):
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        beta = 1 / (er_coef+kl_coef)
        policy = softmax_policy(values, beta=beta)
        self.record_array("Policy", policy)