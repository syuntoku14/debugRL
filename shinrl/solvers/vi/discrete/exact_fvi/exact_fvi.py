import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F
from .core import Solver
from shinrl import utils


class ExactFittedSolver(Solver):
    def run(self, num_steps=10000):
        prev_values = self.value_network(self.all_obss).detach().cpu().numpy()
        for _ in tqdm(range(num_steps)):
            self.set_tb_values_policy()
            self.record_history()

            # ----- update table -----
            values = self.value_network(self.all_obss).detach().cpu().numpy()
            error = np.abs(prev_values - values).max()
            prev_values = values
            self.record_scalar("Error", error)

            target = self.backup(values)  # SxA
            target = torch.tensor(
                target, dtype=torch.float32, device=self.device)
            self.update_network(target)
            self.step += 1

    def record_history(self):
        if self.step % self.solve_options["evaluation_interval"] == 0:
            expected_return = self.env.compute_expected_return(self.tb_policy)
            self.record_scalar("Return", expected_return, tag="Policy")


class ExactFittedViSolver(ExactFittedSolver):
    def backup(self, curr_q_val):
        # Bellman Operator to update q values
        discount = self.solve_options["discount"]
        curr_v_val = np.max(curr_q_val, axis=-1)  # S
        prev_q = self.env.reward_matrix \
            + discount*(self.env.transition_matrix *
                        curr_v_val).reshape(self.dS, self.dA)
        return prev_q

    def set_tb_values_policy(self):
        q_values = self.value_network(self.all_obss).reshape(
            self.dS, self.dA).detach().cpu().numpy()
        policy = utils.eps_greedy_policy(q_values, eps_greedy=0.0)
        self.record_array("Values", q_values)
        self.record_array("Policy", policy)


class ExactFittedCviSolver(ExactFittedSolver):
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

    def set_tb_values_policy(self):
        preference = self.value_network(self.all_obss).reshape(
            self.dS, self.dA).detach().cpu().numpy()
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        beta = 1 / (er_coef+kl_coef)
        policy = utils.softmax_policy(preference, beta=beta)
        self.record_array("Values", preference)
        self.record_array("Policy", policy)
