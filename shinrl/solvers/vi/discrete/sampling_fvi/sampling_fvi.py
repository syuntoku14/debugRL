import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from shinrl import utils

from .core import Solver


class SamplingFittedSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            if self.is_tabular:
                self.set_tb_values_policy()
            self.record_history()

            # ----- generate mini-batch from the replay_buffer -----
            self.buffer.add(**self.collect_samples(self.solve_options["num_samples"]))
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
            if self.is_tabular:
                expected_return = self.env.compute_expected_return(self.tb_policy)
                self.record_scalar("Return", expected_return, tag="Policy")
                aval = self.env.compute_action_values(
                    self.tb_policy, self.solve_options["discount"]
                )
                self.record_scalar("QError", ((aval - self.tb_values) ** 2).mean())
            else:
                n_episodes = self.solve_options["gym_evaluation_episodes"]
                traj = utils.collect_samples(
                    self.env, self.get_action_gym, num_episodes=n_episodes
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


class SamplingFittedViSolver(SamplingFittedSolver):
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

    def set_tb_values_policy(self):
        q_values = (
            self.value_network(self.all_obss)
            .reshape(self.dS, self.dA)
            .detach()
            .cpu()
            .numpy()
        )
        eps_greedy = utils.compute_epsilon(
            self.step,
            self.solve_options["eps_start"],
            self.solve_options["eps_end"],
            self.solve_options["eps_decay"],
        )
        policy = utils.eps_greedy_policy(q_values, eps_greedy=eps_greedy)
        self.record_array("Values", q_values)
        self.record_array("Policy", policy)

    def get_action_gym(self, env):
        obs = torch.as_tensor(
            env.obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        vals = self.value_network(obs).detach().cpu().numpy()  # 1 x dA
        eps_greedy = utils.compute_epsilon(
            self.step,
            self.solve_options["eps_start"],
            self.solve_options["eps_end"],
            self.solve_options["eps_decay"],
        )
        probs = utils.eps_greedy_policy(vals, eps_greedy=eps_greedy).reshape(-1)
        log_probs = np.log(probs)
        action = np.random.choice(np.arange(0, env.action_space.n), p=probs)
        log_prob = log_probs[action]
        return action, log_prob


class SamplingFittedCviSolver(SamplingFittedSolver):
    def compute_target(self, tensor_traj):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha, beta = kl_coef / (er_coef + kl_coef), 1 / (er_coef + kl_coef)
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
            next_preference = self.target_value_network(next_obss)  # SxA
            curr_max = self.max_operator(curr_preference, beta)  # S
            next_max = self.max_operator(next_preference, beta)  # S

            curr_preference = curr_preference.gather(
                1, actions.reshape(-1, 1)
            ).squeeze()  # S

            new_preference = (
                alpha * (curr_preference.squeeze() - curr_max.squeeze())
                + rews
                + discount * next_max * (~dones)
            )
        return new_preference.squeeze()  # S

    def set_tb_values_policy(self):
        preference = (
            self.value_network(self.all_obss)
            .reshape(self.dS, self.dA)
            .detach()
            .cpu()
            .numpy()
        )
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        beta = 1 / (er_coef + kl_coef)
        policy = utils.softmax_policy(preference, beta=beta)
        self.record_array("Values", preference)
        self.record_array("Policy", policy)

    def get_action_gym(self, env):
        obs = torch.as_tensor(
            env.obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        preference = self.value_network(obs).detach().cpu().numpy()  # 1 x dA
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        beta = 1 / (er_coef + kl_coef)
        probs = utils.softmax_policy(preference, beta=beta).reshape(-1)
        log_probs = np.log(probs)
        action = np.random.choice(np.arange(0, env.action_space.n), p=probs)
        log_prob = log_probs[action]
        return action, log_prob


class SamplingFittedMviSolver(SamplingFittedSolver):
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
            next_preference = self.target_value_network(next_obss)  # SxA
            curr_policy = (
                F.softmax(curr_preference / tau, dim=-1)
                .gather(1, actions.reshape(-1, 1))
                .squeeze()
            )  # S
            next_policy = F.softmax(next_preference / tau, dim=-1)  # SxA
            next_max = (next_policy * (next_preference - tau * next_policy.log())).sum(
                dim=-1
            )

            new_preference = (
                rews + alpha * tau * curr_policy.log() + discount * next_max * (~dones)
            )
        return new_preference.squeeze()  # S

    def set_tb_values_policy(self):
        preference = (
            self.value_network(self.all_obss)
            .reshape(self.dS, self.dA)
            .detach()
            .cpu()
            .numpy()
        )
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        tau = er_coef + kl_coef
        policy = utils.softmax_policy(preference, beta=1 / tau)
        self.record_array("Values", preference)
        self.record_array("Policy", policy)

    def get_action_gym(self, env):
        obs = torch.as_tensor(
            env.obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        preference = self.value_network(obs).detach().cpu().numpy()  # 1 x dA
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        tau = er_coef + kl_coef
        probs = utils.softmax_policy(preference, beta=1 / tau).reshape(-1)
        log_probs = np.log(probs)
        action = np.random.choice(np.arange(0, env.action_space.n), p=probs)
        log_prob = log_probs[action]
        return action, log_prob
