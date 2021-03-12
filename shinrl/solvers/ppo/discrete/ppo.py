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
            with torch.no_grad():
                trajectory = self.compute_coef(trajectory)

            # ----- update networks -----
            for traj in self.iter(trajectory):
                tensor_traj = utils.trajectory_to_tensor(traj, self.device)
                actor_loss, critic_loss = self.update(tensor_traj)
            self.record_scalar("LossActor", actor_loss)
            self.record_scalar("LossCritic", critic_loss)

            self.step += 1

    def compute_coef(self, traj):
        discount, lam = self.solve_options["discount"], self.solve_options["td_lam"]
        rew, done, timeout = traj["rew"], traj["done"], traj["timeout"]
        # done = done * (~timeout)
        obs = torch.tensor(
            traj["obs"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(
            traj["next_obs"], dtype=torch.float32, device=self.device)
        values = self.value_network(obs).squeeze(1).detach().cpu().numpy()
        next_values = self.value_network(
            next_obs).squeeze(1).detach().cpu().numpy()
        td_err = rew + discount*next_values*(~done) - values

        gae = 0.
        gaes = np.zeros_like(rew)
        ret = 0.
        rets = np.zeros_like(rew)
        for step in reversed(range(len(rew))):
            gae = td_err[step] + discount*lam*gae*(~done[step])
            ret = rew[step] + discount*ret*(~done[step])
            gaes[step] = gae
            rets[step] = ret
        traj["ret"] = rets
        gaes = (gaes - gaes.mean()) / gaes.std()
        traj["coef"] = gaes
        return traj

    def iter(self, traj):
        batch_size = traj["ret"].shape[0]
        minibatch_size = batch_size // self.solve_options["num_minibatches"]
        for _ in range(self.solve_options["num_minibatches"]):
            rand_ids = np.random.randint(0, batch_size, minibatch_size)
            yield {key: val[rand_ids] for key, val in traj.items()}

    def update(self, tensor_traj):
        clip_ratio = self.solve_options["clip_ratio"]
        obss, actions, old_log_probs = tensor_traj["obs"], tensor_traj["act"], tensor_traj["log_prob"]
        ret, advantage = tensor_traj["ret"], tensor_traj["coef"]

        preference = self.policy_network(obss)
        policy = torch.softmax(preference, dim=-1)  # BxA
        log_probs = policy.gather(
            1, actions.reshape(-1, 1)).squeeze().log()  # B
        ratio = torch.exp(log_probs - old_log_probs)  # B
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advantage
        actor_loss = -torch.min(ratio * advantage, clip_adv).mean()
        entropy = -torch.sum(policy * policy.log(), dim=-1).mean()

        values = self.value_network(obss).squeeze(-1)
        critic_loss = self.critic_loss(values, ret)

        loss = actor_loss + \
            self.solve_options["vf_coef"]*critic_loss - \
            self.solve_options["ent_coef"]*entropy
        self.optimizer.zero_grad()
        loss.backward()
        if self.solve_options["clip_grad"]:
            for param in self.params:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return actor_loss.detach().cpu().item(), critic_loss.detach().cpu().item()

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
