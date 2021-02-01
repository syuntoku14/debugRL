from copy import deepcopy
from scipy import stats
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .core import Solver
from debug_rl.utils import (
    collect_samples,
    squeeze_trajectory,
    trajectory_to_tensor
)


class IpgSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_performance()

            # ------ collect samples using the current policy ------
            preference = self.policy_network(
                self.all_obss).reshape(self.dS, self.dA).detach().cpu().numpy()
            policy = self.compute_policy(preference)
            trajectory = collect_samples(
                self.env, policy, num_samples=self.solve_options["num_samples"])
            self.buffer.add(**trajectory)

            # ----- compute on-policy gradient coefficient -----
            trajectory = self.compute_coef(trajectory, policy)
            trajectory = squeeze_trajectory(trajectory)
            tensor_traj = trajectory_to_tensor(trajectory, self.device)

            # ----- generate mini-batch from the replay_buffer -----
            off_trajectory = self.buffer.sample(
                self.solve_options["minibatch_size"])
            off_trajectory = squeeze_trajectory(off_trajectory)
            tensor_off_traj = trajectory_to_tensor(off_trajectory, self.device)

            # ----- update networks -----
            on_actor_loss, off_actor_loss = self.update_actor(tensor_traj, tensor_off_traj)
            self.record_scalar("LossActor", on_actor_loss, tag="on")
            self.record_scalar("LossActor", off_actor_loss, tag="off")
            baseline_loss = self.update_baseline(tensor_traj)
            self.record_scalar("LossBaseline", baseline_loss)
            critic_loss = self.update_critic(tensor_traj)
            self.record_scalar("LossCritic", critic_loss)

            # ----- update target network -----
            if (self.step+1) % self.solve_options["target_update_interval"] == 0:
                self.target_value_network.load_state_dict(
                    self.value_network.state_dict())
            self.step += 1

    def compute_coef(self, traj, policy):
        discount, lam = self.solve_options["discount"], self.solve_options["td_lam"]
        Q_table = self.env.compute_action_values(policy, discount)  # SxA

        state, next_state, act, rew, done = traj["state"], traj[
            "next_state"], traj["act"], traj["rew"], traj["done"]
        obs = torch.tensor(
            traj["obs"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(
            traj["next_obs"], dtype=torch.float32, device=self.device)
        values = self.baseline_network(obs).squeeze(1).detach().cpu().numpy()
        next_values = self.baseline_network(next_obs).squeeze(1).detach().cpu().numpy()
        td_err = rew + discount * next_values - values

        # last_val != 0.0 since TabularEnv returns done only at timeout
        def compute_last_val(step):  
            if self.solve_options["last_val"] == "oracle":
                return Q_table[state[step], act[step]]
            elif self.solve_options["last_val"] == "critic":
                return values[step]
            else:
                raise ValueError

        def compute_last_adv(step):
            if self.solve_options["last_val"] == "oracle":
                return 0.0
            elif self.solve_options["last_val"] == "critic":
                return td_err[step]
            else:
                raise ValueError

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
        traj["adv"] = GAE
        if self.solve_options["coef"] == "Q":
            traj["coef"] = Q
        elif self.solve_options["coef"] == "A":
            traj["coef"] = Q - values
        elif self.solve_options["coef"] == "GAE":
            traj["coef"] = GAE
        else:
            raise ValueError
        return traj

    def update_actor(self, tensor_traj, tensor_off_traj):
        discount = self.solve_options["discount"]
        obss, actions, coef = tensor_traj["obs"], tensor_traj["act"], tensor_traj["coef"]

        preference = self.policy_network(obss)
        policy = torch.softmax(preference, dim=-1)  # BxA
        log_a = policy.gather(1, actions.reshape(-1, 1)).squeeze().log()  # B

        # on-policy loss
        on_loss = - (1-self.solve_options["on_off_coef"]) * (coef*log_a).mean()

        # off-policy loss
        off_obss = tensor_off_traj["obs"]
        off_policy = torch.softmax(
            self.policy_network(off_obss), dim=-1)  # BxA
        off_policy = off_policy + (off_policy == 0.0).float() * \
            1e-8  # avoid numerical instability
        values = self.value_network(off_obss).detach()  # BxA
        off_loss = -self.solve_options["on_off_coef"] * ((values * off_policy).sum(1)).mean()

        # entropy bonus
        off_entropy = - (off_policy * off_policy.log()).sum(1).mean()
        er_loss = - self.solve_options["er_coef"] * off_entropy

        loss = on_loss + off_loss + er_loss
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return on_loss.detach().cpu().item(), off_loss.detach().cpu().item()

    def update_critic(self, tensor_traj):
        discount = self.solve_options["discount"]
        obss, next_obss, act, rews, dones, timeouts = tensor_traj["obs"], tensor_traj[
            "next_obs"], tensor_traj["act"], tensor_traj["rew"], tensor_traj["done"], tensor_traj["timeout"]
        # Ignore the "done" signal if it comes from hitting the time
        dones = dones * (~timeouts)

        with torch.no_grad():
            next_policy = torch.softmax(
                self.policy_network(next_obss), dim=-1)  # BxA
            next_q_val = self.target_value_network(next_obss)  # SxA
            next_vval = (next_q_val * next_policy).sum(dim=-1)  # S
            new_q = rews + discount*next_vval*(~dones)
        values = self.value_network(obss).gather(
            1, act.reshape(-1, 1)).squeeze()
        loss = self.critic_loss(values, new_q)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        return loss.detach().cpu().item()

    def update_baseline(self, tensor_traj):
        obss, act, returns = tensor_traj["obs"], tensor_traj["act"], tensor_traj["ret"]
        values = self.baseline_network(obss).squeeze()
        loss = self.critic_loss(values, returns)
        self.baseline_optimizer.zero_grad()
        loss.backward()
        self.baseline_optimizer.step()
        return loss.detach().cpu().item()
