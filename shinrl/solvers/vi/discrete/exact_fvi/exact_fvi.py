import numpy as np
import torch
from shinrl import utils
from torch.nn import functional as F
from tqdm import tqdm

from .core import Solver


class ExactFittedSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_history()
            self.add_noise()
            self.update_network(self.compute_target())
            self.set_tb_values_policy()
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

    def update_network(self, target):
        target = torch.tensor(target, dtype=torch.float32, device=self.device)
        for _ in range(self.solve_options["proj_iters"]):
            values = self.value_network(self.all_obss)
            loss = self.critic_loss(target, values)
            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()


class ExactFittedViSolver(ExactFittedSolver):
    def compute_target(self):
        discount = self.solve_options["discount"]
        curr_v_val = np.max(self.tb_values, axis=-1)  # S
        prev_q = self.env.reward_matrix + discount * (
            self.env.transition_matrix * curr_v_val
        ).reshape(self.dS, self.dA)
        return np.asarray(prev_q)

    def set_tb_values_policy(self):
        q_values = (
            self.value_network(self.all_obss)
            .reshape(self.dS, self.dA)
            .detach()
            .cpu()
            .numpy()
        )
        policy = utils.eps_greedy_policy(q_values, eps_greedy=0.0)
        self.record_array("Values", q_values)
        self.record_array("Policy", policy)


class ExactFittedCviSolver(ExactFittedSolver):
    def compute_target(self):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha, beta = kl_coef / (er_coef + kl_coef), 1 / (er_coef + kl_coef)
        mP = self.max_operator(self.tb_values, beta)
        prev_preference = (
            alpha * (self.tb_values - mP.reshape(-1, 1))
            + self.env.reward_matrix
            + discount * (self.env.transition_matrix * mP).reshape(self.dS, self.dA)
        )
        return np.asarray(prev_preference)

    def set_tb_values_policy(self):
        preference = (
            self.value_network(self.all_obss)
            .reshape(self.dS, self.dA)
            .detach()
            .cpu()
            .numpy()
        )
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        beta = 1 / (er_coef + kl_coef)
        policy = utils.softmax_policy(preference, beta=beta)
        self.record_array("Values", preference)
        self.record_array("Policy", policy)


class ExactFittedMviSolver(ExactFittedSolver):
    def compute_target(self):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha = kl_coef / (kl_coef + er_coef)
        tau = kl_coef + er_coef

        # update q
        policy = utils.softmax_policy(self.tb_values, beta=1 / tau)
        curr_q = self.tb_values
        x = np.sum(policy * (curr_q - tau * np.log(policy)), axis=-1)
        prev_q = (
            self.env.reward_matrix
            + alpha * tau * np.log(policy)
            + discount * (self.env.transition_matrix * x).reshape(self.dS, self.dA)
        )
        return np.asarray(prev_q)

    def set_tb_values_policy(self):
        preference = (
            self.value_network(self.all_obss)
            .reshape(self.dS, self.dA)
            .detach()
            .cpu()
            .numpy()
        )
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        tau = er_coef + kl_coef
        policy = utils.softmax_policy(preference, beta=1 / tau)
        self.record_array("Values", preference)
        self.record_array("Policy", policy)
