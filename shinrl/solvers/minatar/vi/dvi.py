import random

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from shinrl import utils

from .core import Solver


class DeepVISolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_history()

            # ----- generate mini-batch -----
            utils.collect_samples(
                self.env,
                self.explore,
                num_samples=self.solve_options["num_samples"],
                buffer=self.buffer,
            )
            trajectory = self.buffer.sample(self.solve_options["minibatch_size"])
            tensor_traj = utils.trajectory_to_tensor(trajectory, self.device)

            # ----- update -----
            self.update_network(tensor_traj)
            if (self.step + 1) % self.solve_options["target_update_interval"] == 0:
                self.target_value_network.load_state_dict(
                    self.value_network.state_dict()
                )
            self.step += 1

    def record_history(self) -> None:
        if self.step % self.solve_options["evaluation_interval"] == 0:
            n_episodes = self.solve_options["gym_evaluation_episodes"]
            traj = utils.collect_samples(
                self.env, self.exploit, num_episodes=n_episodes
            )
            expected_return = traj["rew"].sum() / n_episodes
            self.record_scalar("Return", expected_return, tag="Policy")

    def update_network(self, tensor_traj) -> None:
        target = self.compute_target(tensor_traj)
        obss, actions = tensor_traj["obs"], tensor_traj["act"]
        values = self.value_network(obss).gather(1, actions.reshape(-1, 1)).squeeze()
        loss = self.critic_loss(target, values)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        self.record_scalar("LossCritic", loss.detach().cpu().item())

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
                action = self.value_network(obs).max(1)[1].item()
            else:
                action = self.env.action_space.sample()
            return action, 0.0


class DqnSolver(DeepVISolver):
    def compute_target(self, tensor_traj):
        discount = self.solve_options["discount"]
        next_obss, rews, dones, timeouts = (
            tensor_traj["next_obs"],
            tensor_traj["rew"],
            tensor_traj["done"],
            tensor_traj["timeout"],
        )
        # Ignore the "done" signal if it comes from hitting the time
        dones = dones * (~timeouts)

        with torch.no_grad():
            next_q_val = self.target_value_network(next_obss)  # SxA
            next_vval, _ = torch.max(next_q_val, dim=-1)  # S
            new_q = rews + discount * next_vval * (~dones)
        return new_q.squeeze()  # S

    def exploit(self, env):
        with torch.no_grad():
            obs = torch.as_tensor(
                env.obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action = self.value_network(obs).max(1)[1].item()
            return action, 0.0


class MDqnSolver(DeepVISolver):
    """MDqnSolver
    Implementation of Munchausen DQN: https://arxiv.org/abs/2007.14430
    """

    def compute_log_policy(self, val, tau):
        # to avoid numerical problems
        val = val - val.max(dim=-1, keepdim=True)[0]
        logp = val / tau - torch.logsumexp(val / tau, dim=-1, keepdim=True)
        return torch.clamp(logp, min=-10, max=1)

    def compute_target(self, tensor_traj):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha, tau = kl_coef / (kl_coef + er_coef), kl_coef + er_coef
        obss, actions, next_obss, rews, dones, timeouts = (
            tensor_traj["obs"],
            tensor_traj["act"],
            tensor_traj["next_obs"],
            tensor_traj["rew"],
            tensor_traj["done"],
            tensor_traj["timeout"],
        )
        dones = dones * (~timeouts)

        with torch.no_grad():
            curr_preference = self.target_value_network(obss)
            curr_policy_log = (
                self.compute_log_policy(curr_preference, tau)
                .gather(1, actions.reshape(-1, 1))
                .squeeze()
            )
            next_preference = self.target_value_network(next_obss)  # SxA
            next_policy = F.softmax(next_preference / tau, dim=-1)  # SxA
            next_policy_log = self.compute_log_policy(next_preference, tau)
            next_max = (next_policy * (next_preference - tau * next_policy_log)).sum(
                dim=-1
            )
            new_preference = (
                rews + alpha * tau * curr_policy_log + discount * next_max * (~dones)
            )
        return new_preference.squeeze()  # S

    def exploit(self, env):
        with torch.no_grad():
            obs = torch.as_tensor(
                env.obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            preference = self.value_network(obs).detach().cpu().numpy()  # 1 x dA
            er_coef, kl_coef = (
                self.solve_options["er_coef"],
                self.solve_options["kl_coef"],
            )
            tau = er_coef + kl_coef
            probs = utils.softmax_policy(preference, beta=1 / tau).reshape(-1)
            log_probs = np.log(probs)
            action = np.random.choice(np.arange(0, env.action_space.n), p=probs)
            log_prob = log_probs[action]
            return action, log_prob
