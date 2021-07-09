from copy import deepcopy

import numpy as np
import torch
from scipy import stats
from torch.nn import functional as F
from tqdm import tqdm

from shinrl import utils

from .core import Solver


class ExactPgSolver(Solver):
    def run(self, num_steps=10000):
        assert self.is_tabular
        for _ in tqdm(range(num_steps)):
            self.set_tb_values_policy()
            self.record_history()
            self.update_actor()
            self.step += 1

    def update_actor(self):
        discount = self.solve_options["discount"]
        # compute policy
        dist = self.policy_network.compute_distribution(self.all_obss)
        all_actions = self.policy_network.unsquash_action(self.all_actions)
        log_policy = self.policy_network.compute_logp_pi(dist, all_actions)
        policy = torch.softmax(log_policy, dim=-1).reshape(self.dS, self.dA)
        policy_np = policy.detach().cpu().numpy()
        # compute action values
        action_values = self.env.compute_action_values(policy_np, discount)  # SxA
        action_values = torch.tensor(
            action_values, dtype=torch.float32, device=self.device
        )
        # compute stationary distribution
        visitation = self.env.compute_visitation(policy_np, discount).sum(
            axis=(0, 2)
        )  # S
        visitation = torch.tensor(visitation, dtype=torch.float32, device=self.device)

        if self.solve_options["coef"] == "Q":
            coef = action_values
        elif self.solve_options["coef"] == "A":
            coef = action_values - (action_values * policy).sum(dim=1, keepdims=True)
        else:
            raise ValueError(
                "{} is an invalid option.".format(self.solve_options["coef"])
            )

        # update policy
        loss = -((coef * policy).sum(dim=1) * visitation).sum(dim=0)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        self.record_scalar("LossActor", loss.detach().cpu().item())

    def set_tb_values_policy(self):
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
        values = self.env.compute_action_values(policy)
        self.record_array("Values", values)

    def record_history(self):
        if self.step % self.solve_options["evaluation_interval"] == 0:
            expected_return = self.env.compute_expected_return(self.tb_policy)
            self.record_scalar("Return", expected_return, tag="Policy")
