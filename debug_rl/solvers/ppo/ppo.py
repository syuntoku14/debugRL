from copy import deepcopy
from scipy import stats
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .core import Solver
from debug_rl.utils import (
    trajectory_to_tensor,
    collect_samples,
)


class PpoSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_performance()

            # ------ collect samples using the current policy ------
            preference = self.policy_network(self.all_obss).reshape(
                self.dS, self.dA).detach().cpu().numpy()
            policy = self.compute_policy(preference)
            trajectory = collect_samples(
                self.env, policy, self.solve_options["num_samples"])
            trajectory = self.compute_coef(trajectory, policy)
            tensor_traj = trajectory_to_tensor(trajectory, self.device)

            # ----- update networks -----
            actor_loss = self.update_actor(tensor_traj)
            self.record_scalar("LossActor", actor_loss)
            critic_loss = self.update_critic(tensor_traj)
            self.record_scalar("LossCritic", critic_loss)

            self.step += 1

    def compute_coef(self, traj, policy):
        discount, lam = self.solve_options["discount"], self.solve_options["td_lam"]
        state, next_state, act, rew, done = traj["state"], traj[
            "next_state"], traj["act"], traj["rew"], traj["done"]
        obs = torch.tensor(
            traj["obs"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(
            traj["next_obs"], dtype=torch.float32, device=self.device)
        values = self.value_network(obs).squeeze(1).detach().cpu().numpy()
        next_values = self.value_network(next_obs).squeeze(1).detach().cpu().numpy()
        td_err = rew + discount * next_values - values
        Q_table = self.env.compute_action_values(policy, discount)  # SxA

        # last_val != 0.0 since TabularEnv returns done only at timeout
        def compute_last_val(step):  
            return values[step]

        def compute_last_adv(step):
            return td_err[step]

        Q, GAE, oracle_Q, oracle_V = [], [], [], []
        ret = compute_last_val(len(rew)-1)
        gae = compute_last_adv(len(rew)-1)
        for step in reversed(range(len(rew))):
            gae = compute_last_adv(step) if done[step] else td_err[step] + discount*lam*gae 
            ret = compute_last_val(step) if done[step] else rew[step] + discount*ret 
            GAE.insert(0, gae)
            Q.insert(0, ret)
            oracle_Q.insert(0, Q_table[state[step], act[step]])
            oracle_V.insert(0, np.sum(policy[state[step]] * Q_table[state[step]]))
        Q, GAE, oracle_Q, oracle_V = np.array(Q), np.array(GAE), np.array(oracle_Q), np.array(oracle_V)
        self.record_scalar("ReturnError", np.mean((Q-oracle_Q)**2))
        self.record_scalar("AdvantageError", np.mean(((Q-values) - (oracle_Q-oracle_V))**2), tag="Q-V")
        self.record_scalar("AdvantageError", np.mean((GAE - (oracle_Q-oracle_V))**2), tag="GAE")

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
