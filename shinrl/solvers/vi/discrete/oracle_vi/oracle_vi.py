import numpy as np
from tqdm import tqdm

from shinrl.utils import eps_greedy_policy, softmax_policy

from .core import Solver


class OracleSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_history()
            self.update()
            self.add_noise()
            self.step += 1

    def add_noise(self):
        # to analyze the error tolerance
        noise_scale = self.solve_options["noise_scale"]
        values = self.tb_values
        values = values + np.random.randn(*values.shape) * noise_scale
        self.record_array("Values", values)

    def record_history(self):
        if self.step % self.solve_options["evaluation_interval"] == 0:
            expected_return = self.env.compute_expected_return(self.tb_policy)
            self.record_scalar("Return", expected_return, tag="Policy")


class OracleViSolver(OracleSolver):
    def update(self):
        # update q
        curr_q_val = self.tb_values
        discount = self.solve_options["discount"]
        curr_v_val = np.max(curr_q_val, axis=-1)  # S
        prev_q = self.env.reward_matrix + discount * (
            self.env.transition_matrix * curr_v_val
        ).reshape(self.dS, self.dA)
        self.record_array("Values", np.asarray(prev_q))

        # set greedy policy
        policy = eps_greedy_policy(self.tb_values, eps_greedy=0.0)
        self.record_array("Policy", policy)


class OracleCviSolver(OracleSolver):
    """
    Implementation of Conservative Value Iteration (CVI).
    See http://proceedings.mlr.press/v89/kozuno19a.html.
    """

    def update(self):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha, beta = kl_coef / (er_coef + kl_coef), 1 / (er_coef + kl_coef)
        curr_pref = self.tb_values

        # update preference
        mP = self.max_operator(curr_pref, beta)
        prev_preference = (
            alpha * (curr_pref - mP.reshape(-1, 1))
            + self.env.reward_matrix
            + discount * (self.env.transition_matrix * mP).reshape(self.dS, self.dA)
        )
        self.record_array("Values", np.asarray(prev_preference))

        # set greedy policy
        policy = softmax_policy(self.tb_values, beta=beta)
        self.record_array("Policy", policy)


class OracleMviSolver(OracleSolver):
    """
    Implementation of Munchausen Value Iteration (MVI).
    MVI is equivalent to CVI.
    See https://arxiv.org/abs/2007.14430.
    """

    def update(self):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        # alpha controls the ratio of kl and entropy coefficients
        # tau is the base coefficient.
        alpha = kl_coef / (kl_coef + er_coef)
        tau = kl_coef + er_coef

        # update q
        policy = softmax_policy(self.tb_values, beta=1 / tau)
        curr_q = self.tb_values
        next_max = np.sum(policy * (curr_q - tau * np.log(policy)), axis=-1)
        prev_q = (
            self.env.reward_matrix
            + alpha * tau * np.log(policy)
            + discount
            * (self.env.transition_matrix * next_max).reshape(self.dS, self.dA)
        )
        self.record_array("Values", np.asarray(prev_q))

        # set policy
        policy = softmax_policy(self.tb_values, beta=1 / tau)
        self.record_array("Policy", policy)
