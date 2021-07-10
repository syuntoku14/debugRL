from copy import deepcopy

import gym
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from shinrl import utils

from .core import Solver


class SacSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_history()

            # ----- generate mini-batch from replay_buffer -----
            utils.collect_samples(
                self.env,
                self.get_action,
                num_samples=self.solve_options["num_samples"],
                buffer=self.buffer,
            )
            trajectory = self.buffer.sample(self.solve_options["minibatch_size"])
            tensor_traj = utils.trajectory_to_tensor(trajectory, self.device)

            # ----- update networks -----
            self.update_critic(tensor_traj)
            self.update_actor(tensor_traj)

            # ----- update target networks with constant polyak -----
            polyak = self.solve_options["polyak"]
            with torch.no_grad():
                for p, p_targ, p2, p_targ2 in zip(
                    self.value_network.parameters(),
                    self.target_value_network.parameters(),
                    self.value_network2.parameters(),
                    self.target_value_network2.parameters(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                    p_targ2.data.mul_(polyak)
                    p_targ2.data.add_((1 - polyak) * p2.data)
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
        loss = self.critic_loss(target, q1) + self.critic_loss(target, q2)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def update_actor(self, tensor_traj):
        er_coef = self.solve_options["er_coef"]
        obss = tensor_traj["obs"]

        pi, logp_pi = self.policy_network(obss)
        q_pi = torch.min(
            self.value_network(obss, pi).squeeze(-1),
            self.value_network2(obss, pi).squeeze(-1),
        )

        # Entropy-regularized policy loss
        loss = (logp_pi - q_pi / er_coef).mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def get_action(self, env):
        with torch.no_grad():
            obs = torch.as_tensor(env.obs, dtype=torch.float32, device=self.device)
            pi, logp_pi = self.policy_network(obs)
            return pi.detach().cpu().numpy(), logp_pi.detach().cpu().numpy()

    def record_history(self):
        if self.step % self.solve_options["evaluation_interval"] == 0:
            n_episodes = self.solve_options["gym_evaluation_episodes"]
            traj = utils.collect_samples(
                self.env, self.get_action, num_episodes=n_episodes
            )
            expected_return = traj["rew"].sum() / n_episodes
            self.record_scalar("Return", expected_return, tag="Policy")