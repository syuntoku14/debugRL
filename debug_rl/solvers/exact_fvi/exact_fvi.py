import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F
from .core import Solver
from debug_rl.utils import (
    eps_greedy_policy,
    softmax_policy
)


class ExactFittedSolver(Solver):
    def run(self, num_steps=10000):
        prev_values = self.value_network(self.all_obss).detach().cpu().numpy()
        for _ in tqdm(range(num_steps)):
            self.record_performance()

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

    def update_network(self, target):
        values = self.value_network(self.all_obss)
        loss = self.critic_loss(target, values)
        self.value_optimizer.zero_grad()
        loss.backward()
        if self.solve_options["clip_grad"]:
            for param in self.value_network.parameters():
                param.grad.data.clamp_(-1, 1)
        self.value_optimizer.step()
        return loss.detach().cpu().item()


class ExactFittedViSolver(ExactFittedSolver):
    def backup(self, curr_q_val):
        # Bellman Operator to update q values
        discount = self.solve_options["discount"]
        curr_v_val = np.max(curr_q_val, axis=-1)  # S
        prev_q = self.env.reward_matrix \
            + discount*(self.env.transition_matrix *
                        curr_v_val).reshape(self.dS, self.dA)
        return prev_q

    def to_policy(self, q_values, eps_greedy=0.0):
        return eps_greedy_policy(q_values, eps_greedy=eps_greedy)


class ExactFittedCviSolver(ExactFittedSolver):
    def backup(self, curr_pref):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha, beta = kl_coef / (er_coef+kl_coef), 1 / (er_coef+kl_coef)
        mP = self.max_operator(curr_pref, beta)
        prev_preference = \
            alpha * (curr_pref - mP.reshape(-1, 1)) \
            + self.env.reward_matrix \
            + discount*(self.env.transition_matrix * mP).reshape(self.dS, self.dA)
        return prev_preference

    def to_policy(self, preference):
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        beta = 1 / (er_coef+kl_coef)
        return softmax_policy(preference, beta=beta)
