from copy import deepcopy
from scipy import stats
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .core import Solver
from debug_rl.utils import (
    trajectory_to_tensor,
    squeeze_trajectory,
    collect_samples,
)


class SacSolver(Solver):
    def solve(self):
        self.init_history()
        tensor_all_obss = torch.tensor(
            self.all_obss, dtype=torch.float32, device=self.device)
        preference = self.policy_network(tensor_all_obss).reshape(
            self.dS, self.dA).detach().cpu().numpy()

        self.env.reset()
        # Collect random samples in advance
        policy = self.compute_policy(preference)
        trajectory = collect_samples(
            self.env, policy, self.solve_options["minibatch_size"], self.all_obss)
        self.buffer.add(**trajectory)

        # start training
        for k in tqdm(range(self.solve_options["num_trains"])):
            # ------ collect samples by the current policy ------
            policy = self.compute_policy(preference)
            eval_policy = policy
            trajectory = collect_samples(
                self.env, policy, self.solve_options["num_samples"], self.all_obss)
            # ----- generate mini-batch from the replay_buffer -----
            self.buffer.add(**trajectory)
            trajectory = self.buffer.sample(
                self.solve_options["minibatch_size"])
            trajectory = squeeze_trajectory(trajectory)

            # ----- record performance -----
            self.record_performance(k, tensor_all_obss)

            # ----- update q network -----
            tensor_traj = trajectory_to_tensor(trajectory, self.device)
            target = self.backup(tensor_traj)
            value_loss = self.update_network(
                target, self.value_network, self.value_optimizer,
                self.critic_loss, obss=tensor_traj["obs"], actions=tensor_traj["act"])
            self.update_network(
                target, self.value_network2, self.value_optimizer2,
                self.critic_loss, obss=tensor_traj["obs"],
                actions=tensor_traj["act"])
            self.record_scalar("value loss", value_loss)

            # ----- update policy network -----
            target = self.policy_backup(tensor_traj)  # BxA
            policy_loss = self.update_network(
                target, self.policy_network, self.policy_optimizer,
                self.policy_loss_fn, obss=tensor_traj["obs"])
            self.record_scalar("policy loss", policy_loss)

            preference = self.policy_network(tensor_all_obss).reshape(
                self.dS, self.dA).detach().cpu().numpy()

            # ----- update target network -----
            # soft update
            self.update_target_network(self.value_network,
                                       self.target_value_network, self.solve_options["polyak"])
            self.update_target_network(self.value_network2,
                                       self.target_value_network2, self.solve_options["polyak"])
        self.values = preference
        return preference

    # This implementation does not use reparameterization trick
    def backup(self, tensor_traj):
        discount = self.solve_options["discount"]
        alpha = self.solve_options["alpha"]
        beta = self.solve_options["beta"]
        sigma = (1-alpha) / beta

        obss = tensor_traj["obs"]
        actions = tensor_traj["act"]
        next_obss = tensor_traj["next_obs"]
        rews = tensor_traj["rew"]
        dones = tensor_traj["done"]

        with torch.no_grad():
            # compute q target
            next_q = self.target_value_network(next_obss)  # BxA
            next_q2 = self.target_value_network2(next_obss)  # BxA
            next_q = torch.min(next_q, next_q2)  # BxA
            policy = torch.softmax(
                self.policy_network(next_obss), dim=-1)  # BxA
            policy = policy + (policy == 0.0).float() * \
                1e-8  # avoid numerical instability
            log_policy = torch.log(policy)
            next_v = torch.sum(
                policy * (next_q - sigma*log_policy), dim=-1)  # B
            target_q = rews + discount*next_v*(~dones)
        return target_q.squeeze()  # Q

    def policy_backup(self, tensor_traj):
        obss = tensor_traj["obs"]
        alpha = self.solve_options["alpha"]
        beta = self.solve_options["beta"]
        sigma = (1-alpha) / beta

        with torch.no_grad():
            # compute policy target
            curr_q = self.value_network(obss)  # BxA
            curr_q2 = self.value_network2(obss)  # BxA
            curr_q = torch.min(curr_q, curr_q)  # BxA
            policy = torch.softmax(curr_q/sigma, dim=-1)
            policy = policy + (policy == 0.0).float() * \
                1e-8  # avoid numerical instability
        return policy
