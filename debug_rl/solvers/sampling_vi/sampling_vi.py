import numpy as np
import torch
from tqdm import tqdm
from .core import Solver
from debug_rl.utils import (
    collect_samples,
    make_replay_buffer,
    squeeze_trajectory,
    compute_epsilon,
    eps_greedy_policy,
    softmax_policy
)


class SamplingSolver(Solver):
    def run(self, num_steps=10000):
        values = self.values  # SxA
        for _ in tqdm(range(num_steps)):
            # ------ collect samples by the current policy ------
            policy = self.compute_policy(values)
            trajectory = collect_samples(
                self.env, policy, self.solve_options["num_samples"])

            # ----- record performance -----
            self.record_performance(values, policy)

            # ----- update values -----
            for state, act, next_state, rew in zip(
                    trajectory["state"], trajectory["act"],
                    trajectory["next_state"], trajectory["rew"]):
                curr_value = values[state, act]
                target = self.backup(
                    values, state, act, next_state, rew)
                lr = self.solve_options["lr"]
                values[state, act] = \
                    (1-lr) * curr_value + lr*target
            self.step += 1


class SamplingViSolver(SamplingSolver):
    def backup(self, q_values, state, act, next_state, rew):
        # Bellman Operator to update q values SxA
        discount = self.solve_options["discount"]
        next_v = np.max(q_values[next_state], axis=-1)
        target = rew + discount * next_v
        return target

    def compute_policy(self, q_values):
        # return epsilon-greedy policy
        eps_greedy = compute_epsilon(
            self.step, self.solve_options["eps_start"],
            self.solve_options["eps_end"],
            self.solve_options["eps_decay"])
        return eps_greedy_policy(q_values, eps_greedy=eps_greedy)


class SamplingCviSolver(SamplingSolver):
    def backup(self, pref, state, act, next_state, rew):
        discount = self.solve_options["discount"]
        alpha, beta = self.solve_options["alpha"], self.solve_options["beta"]
        curr_pref = pref[state, act]
        curr_max = self.max_operator(pref[state], beta)
        next_max = self.max_operator(pref[next_state], beta)
        target = alpha * (curr_pref - curr_max) \
            + (rew + discount*next_max)
        return target

    def compute_policy(self, preference):
        # return softmax policy
        beta = self.solve_options["beta"]
        return softmax_policy(preference, beta=beta)
