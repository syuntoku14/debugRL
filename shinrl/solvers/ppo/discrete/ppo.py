from copy import deepcopy
from scipy import stats
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .core import Solver
from shinrl import utils


class PpoSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            if self.is_tabular:
                self.set_tb_values_policy()
            self.record_history()

            # ------ collect samples using the current policy ------
            trajectory = self.collect_samples(
                self.solve_options["num_samples"])
            trajectory = self.compute_coef(trajectory)
            tensor_traj = utils.trajectory_to_tensor(trajectory, self.device)

            # ----- update networks -----
            actor_loss = self.update_actor(tensor_traj)
            self.record_scalar("LossActor", actor_loss)
            critic_loss = self.update_critic(tensor_traj)
            self.record_scalar("LossCritic", critic_loss)

            self.step += 1

    def compute_coef(self, traj):
        discount, lam = self.solve_options["discount"], self.solve_options["td_lam"]
        act, rew, done = traj["act"], traj["rew"], traj["done"]
        obs = torch.tensor(
            traj["obs"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(
            traj["next_obs"], dtype=torch.float32, device=self.device)
        values = self.value_network(obs).squeeze(1).detach().cpu().numpy()
        next_values = self.value_network(
            next_obs).squeeze(1).detach().cpu().numpy()
        td_err = rew + discount * next_values - values
        Q, GAE = [], []

        if self.is_tabular:
            state, next_state = traj["state"], traj["next_state"]
            Q_table = self.env.compute_action_values(
                self.tb_policy, discount)  # SxA
            oracle_Q, oracle_V = [], []

        ret = values[len(rew)-1]
        gae = td_err[len(rew)-1]
        for step in reversed(range(len(rew))):
            gae = td_err[step] if done[step] else td_err[step] + \
                discount*lam*gae
            ret = values[step] if done[step] else rew[step] + discount*ret
            GAE.insert(0, gae)
            Q.insert(0, ret)
            if self.is_tabular:
                oracle_Q.insert(0, Q_table[state[step], act[step]])
                oracle_V.insert(
                    0, np.sum(self.tb_policy[state[step]] * Q_table[state[step]]))
        Q, GAE = np.array(Q), np.array(GAE)
        if self.is_tabular:
            oracle_Q, oracle_V = np.array(oracle_Q), np.array(oracle_V)
            self.record_scalar("ReturnError", np.mean((Q-oracle_Q)**2))
            self.record_scalar("AdvantageError", np.mean(
                ((Q-values) - (oracle_Q-oracle_V))**2), tag="Q-V")
            self.record_scalar("AdvantageError", np.mean(
                (GAE - (oracle_Q-oracle_V))**2), tag="GAE")

        traj["ret"] = Q
        traj["coef"] = GAE
        return traj

    def update_actor(self, tensor_traj):
        clip_ratio = self.solve_options["clip_ratio"]
        obss, actions, log_probs = tensor_traj["obs"], tensor_traj["act"], tensor_traj["log_prob"]
        advantage = tensor_traj["coef"]

        preference = self.policy_network(obss)
        policy = torch.softmax(preference, dim=-1)  # BxA
        entropy = torch.sum(policy * policy.log(), dim=-1)  # B
        log_policy = policy.gather(
            1, actions.reshape(-1, 1)).squeeze().log()  # B
        ratio = torch.exp(log_policy - log_probs)  # B
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advantage

        loss = -(torch.min(ratio * advantage, clip_adv)
                 - self.solve_options["entropy_coef"]*entropy).mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss.detach().cpu().item()

    def update_critic(self, tensor_traj):
        obss = tensor_traj["obs"]
        returns = tensor_traj["ret"]
        values = self.value_network(obss).squeeze(-1)
        loss = 0.5 * self.critic_loss(values, returns)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        return loss.detach().cpu().item()

    def set_tb_values_policy(self):
        values = self.value_network(self.all_obss).reshape(
            self.dS, 1).repeat(1, self.dA).detach().cpu().numpy()
        self.record_array("Values", values)
        preference = self.policy_network(self.all_obss).reshape(
            self.dS, self.dA).detach().cpu().numpy()
        policy = utils.softmax_policy(preference)
        self.record_array("Policy", policy)

    def get_action_gym(self, env):
        obs = torch.as_tensor(env.obs, dtype=torch.float32, device=self.device)
        prefs = self.policy_network(obs).detach().cpu().numpy()
        probs = utils.softmax_policy(prefs)
        log_probs = np.log(probs)
        action = np.random.choice(np.arange(0, env.action_space.n), p=probs)
        log_prob = log_probs[action]
        return action, log_prob

    def collect_samples(self, num_samples):
        if self.is_tabular:
            return utils.collect_samples(
                self.env, utils.get_tb_action, self.solve_options["num_samples"],
                policy=self.tb_policy)
        else:
            return utils.collect_samples(
                self.env, self.get_action_gym, self.solve_options["num_samples"])

    def record_history(self):
        if self.step % self.solve_options["evaluation_interval"] == 0:
            if self.is_tabular:
                expected_return = \
                    self.env.compute_expected_return(self.tb_policy)
                self.record_scalar("Return", expected_return, tag="Policy")
            else:
                n_episodes = self.solve_options["num_episodes_gym_record"]
                traj = utils.collect_samples(
                    self.env, self.get_action_gym, num_episodes=n_episodes)
                expected_return = traj["rew"].sum() / n_episodes
                self.record_scalar("Return", expected_return, tag="Policy")
