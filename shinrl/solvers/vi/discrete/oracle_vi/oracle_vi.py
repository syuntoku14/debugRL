import numpy as np
from scipy import special
from tqdm import tqdm

from shinrl import utils

from .core import Solver


class OracleSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_history()
            new_values = self.compute_new_values()
            self.update(new_values)
            self.step += 1

    def update(self, new_values):
        # to analyze the error tolerance
        noise_scale = self.solve_options["noise_scale"]
        new_values = new_values + np.random.randn(*new_values.shape) * noise_scale
        values = self.tb_values
        values = values + self.solve_options["lr"] * (new_values - values)
        self.record_array("Values", values)
        self.set_tb_policy()

    def record_history(self):
        if self.step % self.solve_options["evaluation_interval"] == 0:
            expected_return = self.env.compute_expected_return(self.tb_policy)
            self.record_scalar("Return", expected_return, tag="Policy")


class OracleViSolver(OracleSolver):
    def compute_new_values(self):
        q = self.tb_values
        discount = self.solve_options["discount"]
        v = np.max(q, axis=-1)  # S
        new_q = self.env.reward_matrix + discount * (
            self.env.transition_matrix * v
        ).reshape(self.dS, self.dA)
        return np.asarray(new_q)

    def set_tb_policy(self):
        # set greedy policy
        policy = utils.eps_greedy_policy(self.tb_values, eps_greedy=0.0)
        self.record_array("Policy", policy)


class OracleCviSolver(OracleSolver):
    """
    Implementation of Conservative Value Iteration (CVI).
    See http://proceedings.mlr.press/v89/kozuno19a.html.
    """

    def compute_new_values(self):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha, beta = kl_coef / (er_coef + kl_coef), 1 / (er_coef + kl_coef)
        p = self.tb_values

        # update preference
        mP = self.max_operator(p, beta)
        new_p = (
            alpha * (p - mP.reshape(-1, 1))
            + self.env.reward_matrix
            + discount * (self.env.transition_matrix * mP).reshape(self.dS, self.dA)
        )
        return np.asarray(new_p)

    def set_tb_policy(self):
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        beta = 1 / (er_coef + kl_coef)
        policy = utils.softmax_policy(self.tb_values, beta=beta)
        self.record_array("Policy", policy)


class OracleMviSolver(OracleSolver):
    """
    Implementation of Munchausen Value Iteration (MVI).
    MVI is equivalent to CVI.
    See https://arxiv.org/abs/2007.14430.
    """

    def compute_new_values(self):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        # alpha controls the ratio of kl and entropy coefficients
        # tau is the base coefficient.
        alpha = kl_coef / (kl_coef + er_coef)
        tau = kl_coef + er_coef

        # update q
        policy = utils.softmax_policy(self.tb_values, beta=1 / tau)
        q = self.tb_values
        next_max = np.sum(policy * (q - tau * np.log(policy)), axis=-1)
        new_q = (
            self.env.reward_matrix
            + alpha * tau * np.log(policy)
            + discount
            * (self.env.transition_matrix * next_max).reshape(self.dS, self.dA)
        )
        return np.asarray(new_q)

    def set_tb_policy(self):
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        tau = kl_coef + er_coef
        policy = utils.softmax_policy(self.tb_values, beta=1 / tau)
        self.record_array("Policy", policy)


