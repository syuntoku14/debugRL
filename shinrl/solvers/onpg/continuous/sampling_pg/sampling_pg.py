import numpy as np
import torch
from tqdm import tqdm

from shinrl import utils

from .core import Solver


class SamplingPgSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            if self.is_tabular:
                self.set_tb_values_policy()
            self.record_history()

            # ------ collect samples using the current policy ------
            trajectory = self.collect_samples(self.solve_options["num_samples"])
            with torch.no_grad():
                trajectory = self.compute_coef(trajectory)

            # ----- update networks -----
            for traj in self.iter(trajectory):
                tensor_traj = utils.trajectory_to_tensor(traj, self.device)
                if len(tensor_traj["act"].shape) == 1:
                    tensor_traj["act"] = tensor_traj["act"].unsqueeze(-1)
                actor_loss, critic_loss = self.update(tensor_traj)
            self.record_scalar("LossActor", actor_loss)
            self.record_scalar("LossCritic", critic_loss)

            self.step += 1

    def compute_coef(self, traj):
        discount, lam = self.solve_options["discount"], self.solve_options["td_lam"]
        act, rew, done = traj["act"], traj["rew"], traj["done"]
        obs = torch.tensor(traj["obs"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(
            traj["next_obs"], dtype=torch.float32, device=self.device
        )
        values = self.value_network(obs).squeeze(1).detach().cpu().numpy()
        next_values = self.value_network(next_obs).squeeze(1).detach().cpu().numpy()
        td_err = rew + discount * next_values - values

        if self.is_tabular:
            state, next_state = traj["state"], traj["next_state"]
            Q_table = self.env.compute_action_values(self.tb_policy, discount)  # SxA
            oracle_Q, rets = np.zeros_like(rew), np.zeros_like(rew)
            oracle_Q, oracle_V = np.zeros_like(rew), np.zeros_like(rew)

        gae = 0.0
        ret = 0.0 if done[-1] else next_values[-1]
        gaes, rets = np.zeros_like(rew), np.zeros_like(rew)
        for step in reversed(range(len(rew))):
            gae = td_err[step] + discount * lam * gae * (~done[step])
            ret = rew[step] + discount * ret * (~done[step])
            gaes[step], rets[step] = gae, ret
            if self.is_tabular:
                oracle_Q[step] = Q_table[
                    state[step], self.env.discretize_action(act[step])
                ]
                oracle_V[step] = np.sum(
                    self.tb_policy[state[step]] * Q_table[state[step]]
                )
        if self.is_tabular:
            self.record_scalar("ReturnError", np.mean((rets - oracle_Q) ** 2))
            self.record_scalar(
                "AdvantageError",
                np.mean(((rets - values) - (oracle_Q - oracle_V)) ** 2),
                tag="Q-V",
            )
            self.record_scalar(
                "AdvantageError",
                np.mean((gaes - (oracle_Q - oracle_V)) ** 2),
                tag="GAE",
            )

        traj["ret"], traj["adv"] = rets, gaes
        if self.solve_options["coef"] == "Q":
            traj["coef"] = rets
        elif self.solve_options["coef"] == "A":
            traj["coef"] = rets - values
        elif self.solve_options["coef"] == "GAE":
            traj["coef"] = gaes
        else:
            raise ValueError
        return traj

    def iter(self, traj):
        batch_size = traj["ret"].shape[0]
        minibatch_size = batch_size // self.solve_options["num_minibatches"]
        for _ in range(self.solve_options["num_minibatches"]):
            rand_ids = np.random.randint(0, batch_size, minibatch_size)
            yield {key: val[rand_ids] for key, val in traj.items()}

    def update(self, tensor_traj):
        obss, actions, old_log_probs = (
            tensor_traj["obs"],
            tensor_traj["act"],
            tensor_traj["log_prob"],
        )
        returns, coef = tensor_traj["ret"], tensor_traj["coef"]

        dist = self.policy_network.compute_pi_distribution(obss)
        log_probs = self.policy_network.compute_logp_pi(dist, actions)
        ratio = torch.exp(log_probs - old_log_probs)  # B
        actor_loss = -(ratio * coef).mean()

        values = self.value_network(obss).squeeze(-1)
        critic_loss = self.critic_loss(values, returns)

        loss = actor_loss + self.solve_options["vf_coef"] * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.solve_options["clip_grad"]:
            for param in self.params:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return actor_loss.detach().cpu().item(), critic_loss.detach().cpu().item()

    def set_tb_values_policy(self):
        values = (
            self.value_network(self.all_obss)
            .reshape(self.dS, 1)
            .repeat(1, self.dA)
            .detach()
            .cpu()
            .numpy()
        )
        self.record_array("Values", values)
        dist = self.policy_network.compute_pi_distribution(self.all_obss)
        log_policy = dist.log_prob(self.all_actions)
        policy_probs = torch.softmax(log_policy, dim=-1).reshape(self.dS, self.dA)
        self.record_array("Policy", policy_probs)

    def get_action_gym(self, env):
        with torch.no_grad():
            obs = torch.as_tensor(env.obs, dtype=torch.float32, device=self.device)
            pi, logp_pi = self.policy_network(obs)
            return pi.detach().cpu().numpy(), logp_pi.detach().cpu().numpy()

    def collect_samples(self, num_samples):
        if self.is_tabular:
            return utils.collect_samples(
                self.env,
                utils.get_tb_action,
                self.solve_options["num_samples"],
                policy=self.tb_policy,
            )
        else:
            return utils.collect_samples(
                self.env, self.get_action_gym, self.solve_options["num_samples"]
            )

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


class PpoSolver(SamplingPgSolver):
    def update(self, tensor_traj):
        clip_ratio = self.solve_options["clip_ratio"]
        obss, actions, old_log_probs = (
            tensor_traj["obs"],
            tensor_traj["act"],
            tensor_traj["log_prob"],
        )
        returns, advantage = tensor_traj["ret"], tensor_traj["coef"]

        dist = self.policy_network.compute_pi_distribution(obss)
        log_probs = self.policy_network.compute_logp_pi(dist, actions)
        ratio = torch.exp(log_probs - old_log_probs)  # B
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
        actor_loss = -torch.min(ratio * advantage, clip_adv).mean()

        values = self.value_network(obss).squeeze(-1)
        critic_loss = self.critic_loss(values, returns)

        entropy = dist.entropy().sum(-1).mean()

        loss = (
            actor_loss
            + self.solve_options["vf_coef"] * critic_loss
            - self.solve_options["ent_coef"] * entropy
        )
        self.optimizer.zero_grad()
        loss.backward()
        if self.solve_options["clip_grad"]:
            for param in self.params:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return actor_loss.detach().cpu().item(), critic_loss.detach().cpu().item()
