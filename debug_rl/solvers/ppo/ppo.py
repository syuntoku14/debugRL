from copy import deepcopy
from scipy import stats
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .core import Solver
from debug_rl.utils import (
    trajectory_to_tensor,
    squeeze_trajectory,
    collect_samples,
)


class PpoSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_performance()

            # ------ collect samples by the current policy ------
            preference = self.policy_network(self.all_obss).reshape(
                self.dS, self.dA).detach().cpu().numpy()
            policy = self.compute_policy(preference)
            trajectory = collect_samples(
                self.env, policy, self.solve_options["num_samples"])
            trajectory = squeeze_trajectory(trajectory)
            gaes, ret = self.compute_gae(trajectory)
            tensor_traj = trajectory_to_tensor(trajectory, self.device)
            tensor_traj.update({"gae": gaes, "ret": ret})

            # ----- update networks -----
            policy_loss = self.update_actor(tensor_traj)
            self.record_scalar("policy loss", policy_loss)
            value_loss = self.update_critic(tensor_traj)
            self.record_scalar("value loss", value_loss)

            self.step += 1

    def compute_gae(self, traj):
        discount = self.solve_options["discount"]
        lam = self.solve_options["lam"]
        obs = torch.tensor(traj["obs"], device=self.device, dtype=torch.float32)
        last_obs = torch.tensor(
            traj["next_obs"], device=self.device, dtype=torch.float32)[-1].unsqueeze(0)
        rew, masks = traj["rew"], ~traj["done"]
        with torch.no_grad():
            ret = []
            gae = 0
            values = self.value_network(
                obs).squeeze(-1).detach().cpu().numpy()  # B
            last_val = self.value_network(
                last_obs).squeeze(-1).detach().cpu().numpy()  # B
            values = np.append(values, last_val)
            for step in reversed(range(len(rew))):
                delta = rew[step] + discount * \
                    values[step+1]*masks[step] - values[step]
                gae = delta + discount*lam*masks[step]*gae
                ret.insert(0, gae+values[step])
            gaes = torch.tensor(
                np.array(ret) - values[:-1], device=self.device, dtype=torch.float32)
            ret = torch.tensor(ret, device=self.device, dtype=torch.float32)
        return gaes, ret

    def update_actor(self, tensor_traj):
        clip_ratio = self.solve_options["clip_ratio"]
        obss, actions, act_prob = tensor_traj["obs"], tensor_traj["act"], tensor_traj["act_prob"]
        advantage = tensor_traj["gae"]

        preference = self.policy_network(obss)
        policy = torch.softmax(preference, dim=-1)  # BxA
        entropy = torch.sum(policy * policy.log(), dim=-1)  # B
        log_policy = policy.gather(
            1, actions.reshape(-1, 1)).squeeze().log()  # B
        log_policy_old = torch.log(act_prob)  # B
        ratio = torch.exp(log_policy - log_policy_old)  # B
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
