import random
from copy import deepcopy

import numpy as np
import torch
from scipy import stats
from torch.nn import functional as F
from tqdm import tqdm

from shinrl import utils

from .core import Solver


class SacSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_history()

            # ----- generate mini-batch  -----
            utils.collect_samples(
                self.env,
                self.explore,
                num_samples=self.solve_options["num_samples"],
                buffer=self.buffer,
            )
            trajectory = self.buffer.sample(self.solve_options["minibatch_size"])
            tensor_traj = utils.trajectory_to_tensor(trajectory, self.device)

            # ----- update networks -----
            self.update_critic(tensor_traj)
            self.update_actor(tensor_traj)
            if (self.step + 1) % self.solve_options["target_update_interval"] == 0:
                self.target_value_network.load_state_dict(
                    self.value_network.state_dict()
                )
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
            next_q = self.target_value_network(next_obss)  # BxA
            policy = torch.softmax(self.policy_network(next_obss), dim=-1)  # BxA
            policy = (
                policy + (policy == 0.0).float() * 1e-8
            )  # avoid numerical instability
            log_policy = torch.log(policy)
            next_v = torch.sum(policy * (next_q - er_coef * log_policy), dim=-1)  # B
            target = rews + discount * next_v * (~dones)

        q = self.value_network(obss).gather(1, actions.reshape(-1, 1)).squeeze()
        loss = self.critic_loss(target, q)

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def update_actor(self, tensor_traj):
        obss = tensor_traj["obs"]
        er_coef = self.solve_options["er_coef"]

        with torch.no_grad():
            # compute policy target
            curr_q = self.value_network(obss)  # BxA
            target = torch.softmax(curr_q / er_coef, dim=-1)
            target = (
                target + (target == 0.0).float() * 1e-8
            )  # avoid numerical instability

        preference = self.policy_network(obss)
        policy = torch.softmax(preference, dim=-1)  # BxA
        policy = policy + (policy == 0.0).float() * 1e-8  # avoid numerical instability
        log_policy = torch.log(policy)  # BxA
        loss = torch.sum(
            policy * (log_policy - target.log()), dim=-1, keepdim=True
        ).mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss.detach().cpu().item()

    def explore(self, env):
        with torch.no_grad():
            eps = random.random()
            eps_thresh = utils.compute_epsilon(
                self.step,
                self.solve_options["eps_decay"],
                self.solve_options["eps_warmup"],
                self.solve_options["eps_end"],
            )
            if eps > 1.0:
                obs = torch.tensor(
                    env.obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                action = self.policy_network(obs).max(1)[1].item()
            else:
                action = self.env.action_space.sample()
            return action, 0.0

    def exploit(self, env):
        with torch.no_grad():
            obs = torch.as_tensor(
                env.obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            prefs = self.policy_network(obs).detach().cpu().numpy()
            probs = utils.softmax_policy(prefs).reshape(-1)
            log_probs = np.log(probs)
            action = np.random.choice(np.arange(0, env.action_space.n), p=probs)
            log_prob = log_probs[action]
            return action, log_prob

    def record_history(self):
        if self.step % self.solve_options["evaluation_interval"] == 0:
            n_episodes = self.solve_options["gym_evaluation_episodes"]
            traj = utils.collect_samples(
                self.env, self.exploit, num_episodes=n_episodes
            )
            expected_return = traj["rew"].sum() / n_episodes
            self.record_scalar("Return", expected_return, tag="Policy")
