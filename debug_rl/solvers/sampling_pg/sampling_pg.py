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


class SamplingPgSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_performance()

            # ------ collect samples using the current policy ------
            preference = self.policy_network(
                self.all_obss).reshape(self.dS, self.dA).detach().cpu().numpy()
            policy = self.compute_policy(preference)
            trajectory = collect_samples(
                self.env, policy, 0, num_episodes=self.solve_options["num_episodes"])
            trajectory = self.compute_coef(trajectory, policy)
            trajectory = squeeze_trajectory(trajectory)
            tensor_traj = trajectory_to_tensor(trajectory, self.device)

            # ----- update networks -----
            actor_loss = self.update_actor(tensor_traj)
            self.record_scalar("LossActor", actor_loss)

            self.step += 1

    def compute_coef(self, traj, policy):
        discount = self.solve_options["discount"]
        state, act, rew, done = traj["state"], traj["act"], traj["rew"], traj["done"]
        Q = []
        oracle_Q = []
        oracle_V = []
        action_values = self.env.compute_action_values(policy, discount)  # SxA

        ret = 0
        for step in reversed(range(len(rew))):
            s, a, r, d = state[step], act[step], rew[step], done[step]
            ret = action_values[s, a] if d else r + discount * ret
            Q.insert(0, ret)
            oracle_Q.insert(0, action_values[s, a])
            oracle_V.insert(0, np.sum(policy[s] * action_values[s]))
        Q = np.array(Q)
        oracle_Q = np.array(oracle_Q)
        oracle_V = np.array(oracle_V)
        self.record_scalar("ReturnError", np.mean((Q-oracle_Q)**2))
        self.record_scalar("AdvantageError", np.mean(((Q-oracle_V) - (oracle_Q-oracle_V))**2))

        if self.solve_options["coef"] == "Q":
            traj["coef"] = Q
        elif self.solve_options["coef"] == "A":
            traj["coef"] = Q - oracle_V
        else:
            raise ValueError("{} is an invalid option.".format(self.solve_options["coef"]))
        return traj

    def update_actor(self, tensor_traj):
        discount = self.solve_options["discount"]
        obss, actions, coef = tensor_traj["obs"], tensor_traj["act"], tensor_traj["coef"]

        preference = self.policy_network(obss)
        policy = torch.softmax(preference, dim=-1)  # BxA
        log_a = policy.gather(1, actions.reshape(-1, 1)).squeeze().log()  # B

        # update policy
        loss = -(coef*log_a).mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss.detach().cpu().item()
