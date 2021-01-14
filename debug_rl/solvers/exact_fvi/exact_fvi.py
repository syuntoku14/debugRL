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
    def solve(self):
        """
        Do value iteration from the end of the episode with a value approximator.

        Returns:
            values: SxA matrix
        """
        self.all_obss = torch.tensor(self.env.all_observations,
                                     dtype=torch.float32, device=self.device)
        error = np.inf
        values = np.zeros((self.dS, self.dA))  # SxA
        # start training
        for k in tqdm(range(self.solve_options["num_trains"])):
            target = self.backup(values)  # SxA
            target = torch.tensor(
                target, dtype=torch.float32, device=self.device)
            self.update_network(target)
            projected = self.value_network(self.all_obss).detach().cpu().numpy()
            error = np.abs(values - projected).max()
            values = projected
            self.record_performance(k, values)
            if error < 1e-7:
                break
        self.record_performance(k, values, force=True)

        print("Iteration finished with maximum error: ", error)

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

    def reset_weights(self):
        for layer in self.value_network:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class ExactFittedViSolver(ExactFittedSolver):
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


class ExactFittedCviSolver(ExactFittedSolver):
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
