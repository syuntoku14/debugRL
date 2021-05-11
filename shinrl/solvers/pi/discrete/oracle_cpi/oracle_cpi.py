import numpy as np
from tqdm import tqdm

from .core import Solver


class OracleCpiSolver(Solver):
    """
    Implementation of Conservative Policy Iteration (CPI)
    See http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.7.7601
    """

    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_history()

            self.evaluate_policy()
            self.add_noise()
            if self.solve_options["constant_mix_rate"] < 0:
                self.update_mix_rate()
            else:
                self.mix_rate = self.solve_options["constant_mix_rate"]
            self.improve_policy()
            self.step += 1

    def add_noise(self):
        # to analyze the error tolerance
        noise_scale = self.solve_options["noise_scale"]
        values = self.tb_values
        values = values + np.random.randn(*values.shape) * noise_scale
        self.record_array("Values", values)

    def update_mix_rate(self):
        discount = self.solve_options["discount"]
        r_max = self.env.reward_matrix.max()
        maxes = self.tb_values.argmax(axis=-1)
        gr_policy = np.zeros_like(self.tb_policy)
        gr_policy[np.arange(len(gr_policy)), maxes] = 1
        q = self.env.compute_action_values(self.tb_policy, discount)
        adv = q - np.sum(self.tb_policy * q, axis=-1, keepdims=True)
        visit = self.env.compute_visitation(self.tb_policy, discount).sum(
            axis=(0, 2)
        )  # S

        A_ = np.sum(gr_policy * (adv), axis=-1)  # S
        A = np.sum(visit * A_)
        TVD = np.abs(gr_policy - self.tb_policy).sum(axis=-1).max()
        Delta = A_.max() - A_.min()
        if self.solve_options["use_spi"]:
            self.mix_rate = (
                A * (1 - discount) ** 2 / (discount * TVD * Delta) if A > 0 else 0.0
            )
        else:
            self.mix_rate = (
                A * (1 - discount) ** 3 / (discount * 8 * r_max) if A > 0 else 0.0
            )
        self.record_scalar("MixRate", self.mix_rate)

    def evaluate_policy(self):
        discount = self.solve_options["discount"]
        curr_q = self.tb_values  # SxA
        curr_v = np.sum(self.tb_policy * (curr_q), axis=-1)  # S
        values = self.env.reward_matrix + discount * (
            self.env.transition_matrix * curr_v
        ).reshape(self.dS, self.dA)
        self.set_tb_values(values)

    def improve_policy(self):
        maxes = self.tb_values.max(axis=-1, keepdims=True)
        gr_policy = (self.tb_values == maxes).astype(float)
        policy = self.tb_policy * (1 - self.mix_rate) + gr_policy * self.mix_rate
        policy = policy / np.sum(policy, axis=-1, keepdims=True)
        self.set_tb_policy(policy)

    def record_history(self):
        if self.step % self.solve_options["evaluation_interval"] == 0:
            expected_return = self.env.compute_expected_return(self.tb_policy)
            self.record_scalar("Return", expected_return, tag="Policy")
            kl = np.sum(
                self.tb_policy * np.log(self.tb_policy / self.tb_prev_policy + 1e-6),
                axis=-1,
            ).max()
            self.record_scalar("maxKL", kl)
