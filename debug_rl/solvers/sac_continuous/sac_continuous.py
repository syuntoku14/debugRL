import gym
from copy import deepcopy
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
    # This implementation assumes one-dimensional action_space
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_performance()

            # ------ collect samples using the current policy ------
            policy_probs = self.make_policy_probs()
            policy = self.compute_policy(policy_probs)
            trajectory = collect_samples(
                self.env, policy, self.solve_options["num_samples"])
            self.buffer.add(**trajectory)

            # ----- generate mini-batch from the replay_buffer -----
            trajectory = self.buffer.sample(
                self.solve_options["minibatch_size"])
            trajectory = squeeze_trajectory(trajectory)
            tensor_traj = trajectory_to_tensor(
                trajectory, self.device, is_discrete=False)
            tensor_traj["act"] = tensor_traj["act"].unsqueeze(-1)

            # ----- update q network -----
            critic_loss = self.update_critic(
                self.value_network, self.value_optimizer, tensor_traj)
            self.record_scalar("LossCritic", critic_loss)
            critic_loss2 = self.update_critic(
                self.value_network2, self.value_optimizer2, tensor_traj)
            self.record_scalar("LossCritic2", critic_loss2)

            # ----- update policy network -----
            actor_loss = self.update_actor(tensor_traj)
            self.record_scalar("LossActor", actor_loss)

            # ----- update target networks with constant polyak -----
            polyak = self.solve_options["polyak"]
            self.update_target_network(self.value_network,
                                       self.target_value_network, polyak)
            self.update_target_network(self.value_network2,
                                       self.target_value_network2, polyak)
            self.step += 1

    def update_critic(self, network, optimizer, tensor_traj):
        discount = self.solve_options["discount"]
        sigma = self.solve_options["sigma"]

        obss, actions, next_obss, rews, dones = tensor_traj["obs"], tensor_traj[
            "act"], tensor_traj["next_obs"], tensor_traj["rew"], tensor_traj["done"]

        with torch.no_grad():
            # compute q target
            raw_pi2, logp_pi2 = self.policy_network(next_obss, return_raw=True)
            pi2 = self.policy_network.squash_action(raw_pi2)
            q1_pi_targ = self.target_value_network(next_obss, pi2).squeeze(-1)
            q2_pi_targ = self.target_value_network2(next_obss, pi2).squeeze(-1)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            target = rews + discount * \
                (~dones) * (q_pi_targ - sigma * logp_pi2)

        values = network(obss, actions).squeeze()
        loss = self.critic_loss(target, values)
        optimizer.zero_grad()
        loss.backward()
        if self.solve_options["clip_grad"]:
            for param in network.parameters():
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return loss.detach().cpu().item()

    def update_actor(self, tensor_traj):
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

    def update_target_network(self, network, target_network, polyak=-1.0):
        if polyak < 0:
            # hard update
            target_network.load_state_dict(network.state_dict())
        else:
            # soft update
            with torch.no_grad():
                for p, p_targ in zip(network.parameters(), target_network.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
