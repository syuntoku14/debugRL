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
    eps_greedy_policy,
    compute_epsilon,
    softmax_policy)


class SamplingFittedSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_performance()

            # ----- generate mini-batch from the replay_buffer -----
            trajectory = self.collect_samples(self.solve_options["num_samples"])
            self.buffer.add(**trajectory)
            trajectory = self.buffer.sample(
                self.solve_options["minibatch_size"])
            tensor_traj = trajectory_to_tensor(trajectory, self.device)

            # ----- update values -----
            target = self.backup(tensor_traj)
            loss = self.update_network(
                target, obss=tensor_traj["obs"], actions=tensor_traj["act"])
            self.record_scalar("LossCritic", loss)

            # ----- update target network -----
            if (self.step+1) % self.solve_options["target_update_interval"] == 0:
                self.target_value_network.load_state_dict(
                    self.value_network.state_dict())
            self.step += 1

    def update_network(self, target, obss, actions):
        values = self.value_network(obss)
        values = values.gather(1, actions.reshape(-1, 1)).squeeze()
        loss = self.critic_loss(target, values)
        self.value_optimizer.zero_grad()
        loss.backward()
        if self.solve_options["clip_grad"]:
            for param in network.parameters():
                param.grad.data.clamp_(-1, 1)
        self.value_optimizer.step()
        return loss.detach().cpu().item()


class SamplingFittedViSolver(SamplingFittedSolver):
    def backup(self, tensor_traj):
        discount = self.solve_options["discount"]
        next_obss, rews, dones, timeouts = tensor_traj["next_obs"], tensor_traj[
            "rew"], tensor_traj["done"], tensor_traj["timeout"]
        # Ignore the "done" signal if it comes from hitting the time
        dones = dones * (~timeouts)

        with torch.no_grad():
            next_q_val = self.target_value_network(next_obss)  # SxA
            next_vval, _ = torch.max(next_q_val, dim=-1)  # S
            new_q = rews + discount*next_vval*(~dones)
        return new_q.squeeze()  # S

    def set_policy(self):
        q_values = self.value_network(self.all_obss).reshape(
            self.dS, self.dA).detach().cpu().numpy()
        eps_greedy = compute_epsilon(
            self.step, self.solve_options["eps_start"],
            self.solve_options["eps_end"],
            self.solve_options["eps_decay"])
        policy = eps_greedy_policy(q_values, eps_greedy=eps_greedy)
        self.record_array("Policy", policy)

    def policy(self, env):
        if not hasattr(env, 'obs'):
            env.obs = env.reset()
        obs = torch.as_tensor(env.obs, dtype=torch.float32).unsqueeze(0)
        vals = net(obs).detach().cpu().numpy()  # 1 x dA
        eps_greedy = compute_epsilon(
            self.step, self.solve_options["eps_start"],
            self.solve_options["eps_end"],
            self.solve_options["eps_decay"])
        probs = eps_greedy_policy(vals, eps_greedy=eps_greedy).reshape(-1)
        log_probs = np.log(probs)
        action = np.random.choice(np.arange(0, env.action_space.n), p=probs)
        log_prob = log_probs[action]
        return action, log_prob


class SamplingFittedCviSolver(SamplingFittedSolver):
    def backup(self, tensor_traj):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha, beta = kl_coef / (er_coef+kl_coef), 1 / (er_coef+kl_coef)
        obss, actions, next_obss, rews, dones, timeouts = tensor_traj["obs"], tensor_traj[
            "act"], tensor_traj["next_obs"], tensor_traj["rew"], tensor_traj["done"], tensor_traj["timeout"]

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

    def set_policy(self):
        preference = self.value_network(self.all_obss).reshape(
            self.dS, self.dA).detach().cpu().numpy()
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        beta = 1 / (er_coef+kl_coef)
        policy = softmax_policy(preference, beta=beta)
        self.record_array("Policy", policy)

    def get_action(self, env):
        if not hasattr(env, 'obs'):
            env.obs = env.reset()
        obs = torch.as_tensor(env.obs, dtype=torch.float32).unsqueeze(0)
        preference = self.value_network(obs).detach().cpu().numpy()  # 1 x dA
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        beta = 1 / (er_coef+kl_coef)
        probs = softmax_policy(preference, beta=beta).reshape(-1)
        log_probs = np.log(probs)
        action = np.random.choice(np.arange(0, env.action_space.n), p=probs)
        log_prob = log_probs[action]
        return action, log_prob
