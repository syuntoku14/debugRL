import numpy as np
import torch
from tqdm import tqdm
from .core import Solver
from rlowan.utils import (
    compute_epsilon,
    eps_greedy_policy,
    softmax_policy,
    collect_samples,
    get_tb_action,
    make_replay_buffer,
)


class SamplingSolver(Solver):
    def run(self, num_steps=10000):
        values = self.tb_values  # SxA

        for _ in tqdm(range(num_steps)):
            self.record_history(values)

            # ----- collect samples using the current policy -----
            trajectory = collect_samples(
                self.env, get_tb_action, self.solve_options["num_samples"],
                policy=self.tb_policy)

            # ----- update values -----
            for state, act, next_state, rew in zip(
                    trajectory["state"], trajectory["act"],
                    trajectory["next_state"], trajectory["rew"]):
                curr_value = values[state, act]
                target = self.backup(values, state, act, next_state, rew)
                lr = self.solve_options["lr"]
                values[state, act] = (1-lr) * curr_value + lr*target
            self.step += 1

    def record_history(self, values):
        self.record_array("Values", values)
        self.set_tb_policy(values)
        if self.step % self.solve_options["record_performance_interval"] == 0:
            expected_return = self.env.compute_expected_return(self.tb_policy)
            self.record_scalar("Return", expected_return, tag="Policy")


class SamplingViSolver(SamplingSolver):
    def backup(self, q_values, state, act, next_state, rew):
        # Bellman Operator to update q values SxA
        discount = self.solve_options["discount"]
        next_v = np.max(q_values[next_state], axis=-1)
        target = rew + discount*next_v
        return target

    def set_tb_policy(self, values):
        # return epsilon-greedy policy
        eps_greedy = compute_epsilon(
            self.step, self.solve_options["eps_start"],
            self.solve_options["eps_end"],
            self.solve_options["eps_decay"])
        policy = eps_greedy_policy(values, eps_greedy=eps_greedy)
        self.record_array("Policy", policy)


class SamplingCviSolver(SamplingSolver):
    def backup(self, pref, state, act, next_state, rew):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha, beta = kl_coef / (er_coef+kl_coef), 1 / (er_coef+kl_coef)
        curr_pref = pref[state, act]
        curr_max = self.max_operator(pref[state], beta)
        next_max = self.max_operator(pref[next_state], beta)
        target = alpha * (curr_pref-curr_max) + (rew+discount*next_max)
        return target

    def set_tb_policy(self, preference):
        # return softmax policy
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        beta = 1 / (er_coef+kl_coef)
        policy = softmax_policy(preference, beta=beta)
        self.record_array("Policy", policy)
