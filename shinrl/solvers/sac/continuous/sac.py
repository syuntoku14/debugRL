from copy import deepcopy

import gym
import numpy as np
import torch
from shinrl import utils
from torch.nn import functional as F
from tqdm import tqdm

from .core import Solver


class SacSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            if self.is_tabular:
                self.set_tb_values_policy()
            self.record_history()

            # ------ collect samples using the current policy ------
            trajectory = self.collect_samples(self.solve_options["num_samples"])
            self.buffer.add(**trajectory)

            # ----- generate mini-batch from the replay_buffer -----
            trajectory = self.buffer.sample(self.solve_options["minibatch_size"])
            tensor_traj = utils.trajectory_to_tensor(trajectory, self.device)
            if len(tensor_traj["act"].shape) == 1:
                tensor_traj["act"] = tensor_traj["act"].unsqueeze(-1)

            # ----- update q network -----
            critic_loss, critic_loss2 = self.update_critic(tensor_traj)
            self.record_scalar("LossCritic", critic_loss)
            self.record_scalar("LossCritic2", critic_loss2)

            # ----- update policy network -----
            actor_loss = self.update_actor(tensor_traj)
            self.record_scalar("LossActor", actor_loss)

            # ----- update target networks with constant polyak -----
            polyak = self.solve_options["polyak"]
            with torch.no_grad():
                for p, p_targ in zip(
                    self.value_network.parameters(),
                    self.target_value_network.parameters(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(
                    self.value_network2.parameters(),
                    self.target_value_network2.parameters(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
            self.step += 1

    def update_critic(self, tensor_traj):
        discount = self.solve_options["discount"]
        er_coef = self.solve_options["er_coef"]

        obss, actions, next_obss, rews, dones, timeouts = (
            tensor_traj["obs"],
            tensor_traj["act"],
            tensor_traj["next_obs"],
            tensor_traj["rew"],
            tensor_traj["done"],
            tensor_traj["timeout"],
        )
        # Ignore the "done" signal if it comes from hitting the time
        dones = dones * (~timeouts)

        with torch.no_grad():
            # compute q target
            pi2, logp_pi2 = self.policy_network(next_obss)
            q1_pi_targ = self.target_value_network(next_obss, pi2).squeeze(-1)
            q2_pi_targ = self.target_value_network2(next_obss, pi2).squeeze(-1)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            target = rews + discount * (q_pi_targ - er_coef * logp_pi2) * (~dones)

        q1 = self.value_network(obss, actions).squeeze()
        q2 = self.value_network2(obss, actions).squeeze()
        loss1 = self.critic_loss(target, q1)
        loss2 = self.critic_loss(target, q2)
        loss = loss1 + loss2
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        return loss1.detach().cpu().item(), loss2.detach().cpu().item()

    def update_actor(self, tensor_traj):
        er_coef = self.solve_options["er_coef"]
        obss = tensor_traj["obs"]

        pi, logp_pi = self.policy_network(obss)
        q1_pi = self.value_network(obss, pi).squeeze(-1)
        q2_pi = self.value_network2(obss, pi).squeeze(-1)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss = (logp_pi - q_pi / er_coef).mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss.detach().cpu().numpy()

    def get_action_gym(self, env):
        if self.buffer.get_stored_size() < self.solve_options["num_random_samples"]:
            return self.env.action_space.sample(), 0.0
        with torch.no_grad():
            obs = torch.as_tensor(env.obs, dtype=torch.float32, device=self.device)
            pi, logp_pi = self.policy_network(obs)
            return pi.detach().cpu().numpy(), logp_pi.detach().cpu().numpy()

    def set_tb_values_policy(self):
        tensor_all_obss = torch.repeat_interleave(
            self.all_obss, self.dA, 0
        )  # (SxA) x obss
        tensor_all_actions = self.all_actions.reshape(-1, 1)
        q1 = self.value_network(tensor_all_obss, tensor_all_actions)
        q2 = self.value_network2(tensor_all_obss, tensor_all_actions)
        values = torch.min(q1, q2).reshape(self.dS, self.dA)
        self.record_array("Values", values)

        dist = self.policy_network.compute_pi_distribution(self.all_obss)
        log_policy = dist.log_prob(self.all_actions) - (
            2 * (np.log(2) - self.all_actions - F.softplus(-2 * self.all_actions))
        )
        policy_probs = torch.softmax(log_policy, dim=-1).reshape(self.dS, self.dA)
        self.record_array("Policy", policy_probs)

    def record_history(self):
        if self.step % self.solve_options["evaluation_interval"] == 0:
            if self.is_tabular:
                expected_return = self.env.compute_expected_return(self.tb_policy)
                self.record_scalar("Return", expected_return, tag="Policy")
            else:
                n_episodes = self.solve_options["gym_evaluation_episodes"]
                traj = utils.collect_samples(
                    self.env, self.get_action_gym, num_episodes=n_episodes
                )
                expected_return = traj["rew"].sum() / n_episodes
                self.record_scalar("Return", expected_return, tag="Policy")
