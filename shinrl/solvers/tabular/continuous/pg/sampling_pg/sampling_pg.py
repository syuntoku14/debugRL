import numpy as np
import torch
from tqdm import tqdm

from shinrl import utils

from .core import Solver


class SamplingPgSolver(Solver):
    def run(self, num_steps=10000):
        assert self.is_tabular
        for _ in tqdm(range(num_steps)):
            self.set_tb_values_policy()
            self.record_history()

            # ------ collect samples ------
            trajectory = self.collect_samples(self.solve_options["num_samples"])
            with torch.no_grad():
                trajectory = self.compute_coef(trajectory)
            if len(trajectory["act"].shape) == 1:
                trajectory["act"] = np.expand_dims(trajectory["act"], axis=-1)

            # ----- update critic -----
            for traj in self.iter(trajectory):
                tensor_traj = utils.trajectory_to_tensor(traj, self.device)
                loss = self.update_critic(tensor_traj)
            self.record_scalar("LossCritic", loss)

            # ----- update actor -----
            for traj in self.iter(trajectory):
                tensor_traj = utils.trajectory_to_tensor(traj, self.device)
                loss = self.update_actor(tensor_traj)
            self.record_scalar("LossActor", loss)

            self.step += 1

    def iter(self, traj):
        batch_size = traj["ret"].shape[0]
        minibatch_size = batch_size // self.solve_options["num_minibatches"]
        for _ in range(self.solve_options["num_minibatches"]):
            rand_ids = np.random.randint(0, batch_size, minibatch_size)
            yield {key: val[rand_ids] for key, val in traj.items()}

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

        gae, ret = 0.0, 0.0 if done[-1] else next_values[-1]
        gaes, rets = np.zeros_like(rew), np.zeros_like(rew)
        for step in reversed(range(len(rew))):
            gae = td_err[step] + discount * lam * gae * (~done[step])
            ret = rew[step] + discount * ret * (~done[step])
            gaes[step], rets[step] = gae, ret
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

    def update_critic(self, tensor_traj):
        obss, returns = tensor_traj["obs"], tensor_traj["ret"]
        values = self.value_network(obss).squeeze(-1)
        loss = self.critic_loss(values, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()

    def update_actor(self, tensor_traj):
        obss, actions, old_log_probs, coef = (
            tensor_traj["obs"],
            tensor_traj["act"],
            tensor_traj["log_prob"],
            tensor_traj["coef"],
        )

        dist = self.policy_network.compute_distribution(obss)
        actions = self.policy_network.unsquash_action(actions)
        log_probs = self.policy_network.compute_logp_pi(dist, actions).sum(-1)
        ratio = torch.exp(log_probs - old_log_probs)  # B
        actor_loss = -(ratio * coef).mean()
        # entropy to encourage exploration
        entropy = self.policy_network.compute_entropy(dist).sum(-1).mean()
        loss = actor_loss - self.solve_options["er_coef"] * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()

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
        dist = self.policy_network.compute_distribution(self.all_obss)
        all_actions = self.policy_network.unsquash_action(self.all_actions)
        log_policy = self.policy_network.compute_logp_pi(dist, all_actions)
        policy = (
            torch.softmax(log_policy, dim=-1)
            .reshape(self.dS, self.dA)
            .detach()
            .cpu()
            .numpy()
        )
        if "array" in self.history["Policy"]:
            kl = (
                policy * (np.log(policy + 1e-20) - np.log(self.tb_policy + 1e-20))
            ).sum(1)
            self.record_scalar("KL", kl.max())
        self.record_array("Policy", policy)

    def get_action_gym(self, env):
        with torch.no_grad():
            obs = torch.as_tensor(env.obs, dtype=torch.float32, device=self.device)
            pi, logp_pi = self.policy_network(obs)
            return pi.detach().cpu().numpy(), logp_pi.sum(-1).detach().cpu().numpy()

    def collect_samples(self, num_samples):
        return utils.collect_samples(
            self.env,
            utils.get_tb_action,
            self.solve_options["num_samples"],
            policy=self.tb_policy,
        )

    def record_history(self):
        if self.step % self.solve_options["evaluation_interval"] == 0:
            expected_return = self.env.compute_expected_return(self.tb_policy)
            self.record_scalar("Return", expected_return, tag="Policy")


class PpoSolver(SamplingPgSolver):
    def update_actor(self, tensor_traj):
        clip_ratio = self.solve_options["clip_ratio"]
        obss, actions, old_log_probs, advantage = (
            tensor_traj["obs"],
            tensor_traj["act"],
            tensor_traj["log_prob"],
            tensor_traj["coef"],
        )

        dist = self.policy_network.compute_distribution(obss)
        actions = self.policy_network.unsquash_action(actions)
        log_probs = self.policy_network.compute_logp_pi(dist, actions).sum(-1)
        ratio = torch.exp(log_probs - old_log_probs)  # B
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
        loss = -torch.min(ratio * advantage, clip_adv).mean()
        # entropy to encourage exploration
        entropy = self.policy_network.compute_entropy(dist).sum(-1).mean()
        loss = loss - self.solve_options["er_coef"] * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()
