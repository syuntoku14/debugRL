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
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_performance()

            # ------ collect samples by the current policy ------
            preference = self.policy_network(self.all_obss).reshape(
                self.dS, self.dA).detach().cpu().numpy()
            policy = self.compute_policy(preference)
            trajectory = collect_samples(
                self.env, policy, self.solve_options["num_samples"])
            self.buffer.add(**trajectory)

            # ----- generate mini-batch from the replay_buffer -----
            trajectory = self.buffer.sample(
                self.solve_options["minibatch_size"])
            trajectory = squeeze_trajectory(trajectory)
            tensor_traj = trajectory_to_tensor(trajectory, self.device)

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

            # ----- update target network -----
            self.update_target_network(self.value_network,
                                       self.target_value_network, self.solve_options["polyak"])
            self.update_target_network(self.value_network2,
                                       self.target_value_network2, self.solve_options["polyak"])
            self.step += 1

    # This implementation does not use reparameterization trick
    def update_critic(self, network, optimizer, tensor_traj):
        discount = self.solve_options["discount"]
        sigma = self.solve_options["sigma"]

        obss, actions, next_obss, rews, dones = tensor_traj["obs"], tensor_traj[
            "act"], tensor_traj["next_obs"], tensor_traj["rew"], tensor_traj["done"]

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
            target = rews + discount*next_v*(~dones)

        values = network(obss)
        values = values.gather(1, actions.reshape(-1, 1)).squeeze()
        loss = self.critic_loss(target, values)
        optimizer.zero_grad()
        loss.backward()
        if self.solve_options["clip_grad"]:
            for param in network.parameters():
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return loss.detach().cpu().item()

    def update_actor(self, tensor_traj):
        obss = tensor_traj["obs"]
        sigma = self.solve_options["sigma"]

        with torch.no_grad():
            # compute policy target
            curr_q = self.value_network(obss)  # BxA
            curr_q2 = self.value_network2(obss)  # BxA
            curr_q = torch.min(curr_q, curr_q)  # BxA
            target = torch.softmax(curr_q/sigma, dim=-1)
            target = target + (target == 0.0).float() * \
                1e-8  # avoid numerical instability

        preference = self.policy_network(obss)
        policy = torch.softmax(preference, dim=-1)  # BxA
        policy = policy + (policy == 0.0).float() * \
            1e-8  # avoid numerical instability
        log_policy = torch.log(policy)  # BxA
        loss = torch.sum(policy * (log_policy-target.log()),
                         dim=-1, keepdim=True).mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        if self.solve_options["clip_grad"]:
            for param in self.policy_network.parameters():
                param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()
        return loss.detach().cpu().item()

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
