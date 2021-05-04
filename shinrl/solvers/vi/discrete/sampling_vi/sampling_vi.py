import numpy as np
import torch
from tqdm import tqdm

from shinrl import utils

from .core import Solver


class SamplingSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.set_tb_policy()
            self.record_history()
            trajectory = utils.collect_samples(
                self.env,
                utils.get_tb_action,
                self.solve_options["num_samples"],
                policy=self.tb_policy,
            )

            for state, act, next_state, rew in zip(
                trajectory["state"],
                trajectory["act"],
                trajectory["next_state"],
                trajectory["rew"],
            ):
                self.update(state, act, next_state, rew)

            self.step += 1

    def record_history(self):
        if self.step % self.solve_options["evaluation_interval"] == 0:
            expected_return = self.env.compute_expected_return(self.tb_policy)
            self.record_scalar("Return", expected_return, tag="Policy")


class SamplingViSolver(SamplingSolver):
    def update(self, state, act, next_state, rew):
        # update table
        lr = self.solve_options["lr"]
        discount = self.solve_options["discount"]
        next_v = np.max(self.tb_values[next_state], axis=-1)
        target = rew + discount * next_v
        curr_q = self.tb_values[state, act]
        self.tb_values[state, act] = (1 - lr) * curr_q + lr * target

    def set_tb_policy(self):
        # set epsilon-greedy policy
        eps_greedy = utils.compute_epsilon(
            self.step,
            self.solve_options["eps_start"],
            self.solve_options["eps_end"],
            self.solve_options["eps_decay"],
        )
        policy = utils.eps_greedy_policy(self.tb_values, eps_greedy=eps_greedy)
        self.record_array("Policy", policy)


class SamplingCviSolver(SamplingSolver):
    def update(self, state, act, next_state, rew):
        lr = self.solve_options["lr"]
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha, beta = kl_coef / (er_coef + kl_coef), 1 / (er_coef + kl_coef)
        curr_pref = self.tb_values[state, act]
        curr_max = self.max_operator(self.tb_values[state], beta)
        next_max = self.max_operator(self.tb_values[next_state], beta)
        target = alpha * (curr_pref - curr_max) + (rew + discount * next_max)
        self.tb_values[state, act] = (1 - lr) * curr_pref + lr * target

    def set_tb_policy(self):
        # set softmax policy
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        policy = utils.softmax_policy(self.tb_values, beta=1 / (er_coef + kl_coef))
        self.record_array("Policy", policy)


class SamplingMviSolver(SamplingSolver):
    """
    This may not work due to insufficient exploration.
    """

    def update(self, state: int, act: int, next_state: int, rew: float) -> None:
        lr = self.solve_options["lr"]
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha, tau = kl_coef / (kl_coef + er_coef), kl_coef + er_coef

        policy = utils.softmax_policy(self.tb_values, beta=1 / tau)
        curr_policy, next_policy = policy[state, act], policy[next_state]
        curr_q, next_q = self.tb_values[state, act], self.tb_values[next_state]
        next_max = np.sum(next_policy * (next_q - tau * np.log(next_policy)), axis=-1)
        target = rew + alpha * tau * np.log(curr_policy) + discount * next_max
        self.tb_values[state, act] = (1 - lr) * curr_q + lr * target

    def set_tb_policy(self) -> None:
        # set softmax policy
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        tau = kl_coef + er_coef
        policy = utils.softmax_policy(self.tb_values, beta=1 / tau)
        self.record_array("Policy", policy)
