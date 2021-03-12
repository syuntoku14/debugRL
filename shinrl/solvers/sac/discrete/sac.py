from copy import deepcopy
from scipy import stats
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .core import Solver
from shinrl import utils


class SacSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            if self.is_tabular:
                self.set_tb_values_policy()
            self.record_history()

            # ------ collect samples using the current policy ------
            trajectory = self.collect_samples(
                self.solve_options["num_samples"])
            self.buffer.add(**trajectory)

            # ----- generate mini-batch from the replay_buffer -----
            trajectory = self.buffer.sample(
                self.solve_options["minibatch_size"])
            tensor_traj = utils.trajectory_to_tensor(trajectory, self.device)

            # ----- update q network -----
            critic_loss, critic_loss2 = self.update_critic(tensor_traj)
            self.record_scalar("LossCritic", critic_loss)
            self.record_scalar("LossCritic2", critic_loss2)

            # ----- update policy network -----
            actor_loss = self.update_actor(tensor_traj)
            self.record_scalar("LossActor", actor_loss)

            # ----- update target networks with constant polyak -----
            polyak = self.solve_options["polyak"]
            with torch.no_grad():
                for p, p_targ in zip(self.value_network.parameters(), self.target_value_network.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(self.value_network2.parameters(), self.target_value_network2.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
            self.step += 1

    def update_critic(self, tensor_traj):
        discount = self.solve_options["discount"]
        er_coef = self.solve_options["er_coef"]

        obss, actions, next_obss, rews, dones, timeouts = tensor_traj["obs"], tensor_traj[
            "act"], tensor_traj["next_obs"], tensor_traj["rew"], tensor_traj["done"], tensor_traj["timeout"]
        # Ignore the "done" signal if it comes from hitting the time
        dones = dones * (~timeouts)

        with torch.no_grad():
            # compute q target
            next_q = self.target_value_network(next_obss)  # BxA
            next_q2 = self.target_value_network2(next_obss)  # BxA
            next_q = torch.min(next_q, next_q2)  # BxA
            policy = torch.softmax(
                self.policy_network(next_obss), dim=-1)  # BxA
            policy = policy + (policy == 0.0).float() * \
                1e-8  # avoid numerical instability
            log_policy = torch.log(policy)
            next_v = torch.sum(
                policy * (next_q - er_coef*log_policy), dim=-1)  # B
            target = rews + discount*next_v*(~dones)

        q1 = self.value_network(obss)
        q1 = q1.gather(1, actions.reshape(-1, 1)).squeeze()
        loss1 = self.critic_loss(target, q1)
        q2 = self.value_network2(obss)
        q2 = q2.gather(1, actions.reshape(-1, 1)).squeeze()
        loss2 = self.critic_loss(target, q2)
        loss = loss1 + loss2

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        return loss1.detach().cpu().item(), loss2.detach().cpu().item()

    def update_actor(self, tensor_traj):
        obss = tensor_traj["obs"]
        er_coef = self.solve_options["er_coef"]

        with torch.no_grad():
            # compute policy target
            curr_q = self.value_network(obss)  # BxA
            curr_q2 = self.value_network2(obss)  # BxA
            curr_q = torch.min(curr_q, curr_q)  # BxA
            target = torch.softmax(curr_q/er_coef, dim=-1)
            target = target + (target == 0.0).float() * \
                1e-8  # avoid numerical instability

        preference = self.policy_network(obss)
        policy = torch.softmax(preference, dim=-1)  # BxA
        policy = policy + (policy == 0.0).float() * \
            1e-8  # avoid numerical instability
        log_policy = torch.log(policy)  # BxA
        loss = torch.sum(policy * (log_policy-target.log()),
                         dim=-1, keepdim=True).mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss.detach().cpu().item()

    def set_tb_values_policy(self):
        q1 = self.value_network(self.all_obss)
        q2 = self.value_network2(self.all_obss)
        values = torch.min(q1, q2).reshape(self.dS, self.dA)
        self.record_array("Values", values)

        prefs = self.policy_network(self.all_obss).detach().cpu().numpy()
        policy = utils.softmax_policy(prefs)
        self.record_array("Policy", policy)

    def get_action_gym(self, env):
        obs = torch.as_tensor(env.obs, dtype=torch.float32, device=self.device)
        prefs = self.policy_network(obs).detach().cpu().numpy()
        probs = utils.softmax_policy(prefs)
        log_probs = np.log(probs)
        action = np.random.choice(np.arange(0, env.action_space.n), p=probs)
        log_prob = log_probs[action]
        return action, log_prob

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
