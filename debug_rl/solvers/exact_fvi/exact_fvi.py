import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F
from .core import Solver
from debug_rl.utils import (
    get_all_observations,
    eps_greedy_policy,
    softmax_policy
)


class ExactFittedSolver(Solver):
    def solve(self):
        """
        Do value iteration from the end of the episode with a value approximator.
        Store the results and backup to self.values

        Returns:
            values: SxA matrix
        """
        values = np.zeros((self.dS, self.dA))  # SxA
        obss = get_all_observations(self.env)
        obss = torch.tensor(obss, dtype=torch.float32, device=self.device)
        error = np.inf
        # start training
        for _ in tqdm(range(self.solve_options["num_trains"])):
            target = self.backup(values)  # SxA
            target = torch.tensor(
                target, dtype=torch.float32, device=self.device)
            self.update_network(target, obss)
            projected = self.value_network(obss).detach().cpu().numpy()
            error = np.abs(values - projected).max()
            values = projected
            if error < 1e-7:
                break

        print("Iteration finished with maximum error: ", error)
        self.values = values
        return self.values

    def update_network(self, target, obss):
        values = self.value_network(obss)
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
        prev_q = self.trr_rew_sum \
            + discount*(self.transition_matrix *
                        curr_v_val).reshape(self.dS, self.dA)
        return prev_q

    def compute_policy(self, q_values, eps_greedy=0.0):
        # return epsilon-greedy policy
        policy_probs = eps_greedy_policy(q_values, eps_greedy=eps_greedy)
        self.policy = policy_probs
        return self.policy


class ExactFittedCviSolver(ExactFittedSolver):
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
        policy_probs = softmax_policy(preference, beta=beta)
        self.policy = policy_probs
        return self.policy
