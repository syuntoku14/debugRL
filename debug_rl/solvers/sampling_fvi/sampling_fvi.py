from scipy.special import rel_entr
from scipy import stats
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .core import Solver
from debug_rl.envs.base import TabularEnv
from debug_rl.utils import (
    trajectory_to_tensor,
    squeeze_trajectory,
    collect_samples,
    compute_epsilon,
    eps_greedy_policy,
    softmax_policy
)


class SamplingFittedSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            values = self.value_network(self.all_obss).reshape(
                self.dS, self.dA).detach().cpu().numpy()
            policy = self.compute_policy(values)
            self.record_performance(policy)

            # ------ collect samples using the current policy ------
            trajectory = collect_samples(
                self.env, policy, self.solve_options["num_samples"])

            # ----- generate mini-batch from the replay_buffer -----
            self.buffer.add(**trajectory)
            trajectory = self.buffer.sample(
                self.solve_options["minibatch_size"])
            trajectory = squeeze_trajectory(trajectory)
            tensor_traj = trajectory_to_tensor(trajectory, self.device)

            # ----- update values -----
            target = self.backup(tensor_traj)
            loss = self.update_network(
                target, self.value_network, self.value_optimizer,
                obss=tensor_traj["obs"], actions=tensor_traj["act"])
            self.record_scalar("LossCritic", loss)

            # ----- update target network -----
            if (self.step+1) % self.solve_options["target_update_interval"] == 0:
                self.target_value_network.load_state_dict(
                    self.value_network.state_dict())
            self.step += 1


class SamplingFittedViSolver(SamplingFittedSolver):
    # SamplingFittedViSolver is the same as DQN
    def backup(self, tensor_traj):
        discount = self.solve_options["discount"]
        next_obss, rews, dones = tensor_traj["next_obs"], tensor_traj[
            "rew"], tensor_traj["done"]

        with torch.no_grad():
            next_q_val = self.target_value_network(next_obss)  # SxA
            next_vval, _ = torch.max(next_q_val, dim=-1)  # S
            new_q = rews + discount*next_vval*(~dones)
        return new_q.squeeze()  # S

    def compute_policy(self, q_values):
        eps_greedy = compute_epsilon(
            self.step, self.solve_options["eps_start"],
            self.solve_options["eps_end"],
            self.solve_options["eps_decay"])
        return eps_greedy_policy(q_values, eps_greedy=eps_greedy)


class SamplingFittedCviSolver(SamplingFittedSolver):
    def backup(self, tensor_traj):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha, beta = kl_coef / (er_coef+kl_coef), 1 / (er_coef+kl_coef)
        obss, actions, next_obss, rews, dones = tensor_traj["obs"], tensor_traj[
            "act"], tensor_traj["next_obs"], tensor_traj["rew"], tensor_traj["done"]

        with torch.no_grad():
            curr_preference = self.target_value_network(obss)
            next_preference = self.target_value_network(next_obss)  # SxA
            curr_max = self.max_operator(curr_preference, beta)  # S
            next_max = self.max_operator(next_preference, beta)  # S

            curr_preference = curr_preference.gather(
                1, actions.reshape(-1, 1)).squeeze()  # S

            new_preference = \
                alpha * (curr_preference.squeeze() - curr_max.squeeze()) \
                + rews + discount*next_max*(~dones)
        return new_preference.squeeze()  # S

    def compute_policy(self, preference):
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        beta = 1 / (er_coef+kl_coef)
        return softmax_policy(preference, beta=beta)
