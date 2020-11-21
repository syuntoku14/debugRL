import gym
from copy import deepcopy
from scipy.special import rel_entr
from scipy import stats
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
import itertools
from .core import Solver
from debug_rl.utils import (
    trajectory_to_tensor,
    squeeze_trajectory,
    collect_samples,
    make_replay_buffer,
)


class SacContinuousSolver(Solver):
    # This assumes one-dimensional action_space
    def solve(self):
        assert isinstance(self.env.action_space, gym.spaces.Box)
        self.init_history()
        tensor_all_obss = torch.tensor(
            self.all_obss, dtype=torch.float32, device=self.device)  # dS x obs_dim
        tensor_all_actions = torch.tensor(
            self.all_actions, dtype=torch.float32,
            device=self.device).repeat(self.dS, 1).reshape(self.dS, self.dA)  # dS x dA
        dist = self.policy_network.compute_pi_distribution(tensor_all_obss)
        log_policy = dist.log_prob(tensor_all_actions) \
            - (2*(np.log(2) - tensor_all_actions - F.softplus(-2*tensor_all_actions)))
        policy_probs = torch.softmax(log_policy, dim=-1).reshape(
            self.dS, self.dA).detach().cpu().numpy() 
        prev_policy = None
        self.env.reset()

        # Collect random samples in advance
        policy = self.compute_policy(policy_probs)
        trajectory = collect_samples(
            self.env, policy, self.solve_options["minibatch_size"], self.all_obss)
        self.buffer.add(**trajectory)

        # start training
        for k in tqdm(range(self.solve_options["num_trains"])):
            # ------ collect samples by the current policy ------
            prev_policy = deepcopy(policy)
            policy = self.compute_policy(policy_probs)
            trajectory = collect_samples(
                self.env, policy, self.solve_options["num_samples"], self.all_obss)
            self.buffer.add(**trajectory)

            # # ----- record performance -----
            self.record_performance(k, tensor_all_obss, tensor_all_actions)
            kl = rel_entr(policy, prev_policy).sum(axis=-1)
            entropy = stats.entropy(policy, axis=1)
            self.record_scalar("KL divergence", kl.mean(), x=k)
            self.record_scalar("Entropy", entropy.mean(), x=k)

            # ----- update q network -----
            trajectory = self.buffer.sample(
                self.solve_options["minibatch_size"])
            trajectory = squeeze_trajectory(trajectory)
            tensor_traj = trajectory_to_tensor(trajectory, self.device, is_discrete=False)
            tensor_traj["act"] = tensor_traj["act"].unsqueeze(-1)
            target = self.backup(tensor_traj)
            value_loss = self.update_network(
                target, self.value_network, self.value_optimizer,
                self.critic_loss, obss=tensor_traj["obs"], actions=tensor_traj["act"])
            self.record_scalar("value loss", value_loss)
            self.update_network(
                target, self.value_network2, self.value_optimizer2,
                self.critic_loss, obss=tensor_traj["obs"],
                actions=tensor_traj["act"])

            # ----- update policy network -----
            # update policy
            policy_loss = self.update_policy(tensor_traj)
            self.record_scalar("policy loss", policy_loss)

            dist = self.policy_network.compute_pi_distribution(tensor_all_obss)
            log_policy = dist.log_prob(tensor_all_actions) \
                - (2*(np.log(2) - tensor_all_actions - F.softplus(-2*tensor_all_actions)))
            policy_probs = torch.softmax(log_policy, dim=-1).reshape(
                self.dS, self.dA).detach().cpu().numpy() 

            # ----- update target networks with constant polyak -----
            polyak = self.solve_options["polyak"]
            self.update_target_network(self.value_network,
                                       self.target_value_network, polyak)
            self.update_target_network(self.value_network2,
                                       self.target_value_network2, polyak)

    def backup(self, tensor_traj):
        discount = self.solve_options["discount"]
        sigma = self.solve_options["sigma"]

        next_obss = tensor_traj["next_obs"]
        rews = tensor_traj["rew"]
        dones = tensor_traj["done"]

        with torch.no_grad():
            # compute q target
            raw_pi2, logp_pi2 = self.policy_network(next_obss, return_raw=True)
            # Target Q-values
            pi2 = self.policy_network.squash_action(raw_pi2)
            q1_pi_targ = self.target_value_network(next_obss, pi2).squeeze(-1)
            q2_pi_targ = self.target_value_network2(next_obss, pi2).squeeze(-1)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = rews + discount * (~dones) * (q_pi_targ - sigma * logp_pi2)
            self.record_scalar("Backup logp_pi2", logp_pi2.mean())
            self.record_scalar("Backup q_pi_targ", q_pi_targ.mean())
            self.record_scalar("Backup rews", rews.mean())
        return backup.squeeze()  # B

    def update_policy(self, tensor_traj):
        sigma = self.solve_options["sigma"]
        obss = tensor_traj["obs"]

        raw_pi, logp_pi = self.policy_network(obss, return_raw=True)
        pi = self.policy_network.squash_action(raw_pi)

        q1_pi = self.value_network(obss, pi).squeeze(-1)
        q2_pi = self.value_network2(obss, pi).squeeze(-1)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss = (logp_pi - 1/sigma*q_pi).mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss
