from copy import deepcopy

import numpy as np
import torch
from scipy import stats
from shinrl import utils
from torch.nn import functional as F
from tqdm import tqdm

from .core import Solver


class ExactPgSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.set_tb_values_policy()
            self.record_history()

            # ----- update networks -----
            actor_loss = self.update_actor()
            self.record_scalar("LossActor", actor_loss)
            self.step += 1

    def update_actor(self):
        discount = self.solve_options["discount"]
        # compute policy
        dist = self.policy_network.compute_pi_distribution(self.all_obss)
        log_policy = dist.log_prob(self.all_actions)
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
        return loss.detach().cpu().item()

    def set_tb_values_policy(self):
        dist = self.policy_network.compute_pi_distribution(self.all_obss)
        log_policy = dist.log_prob(self.all_actions)
        policy = (
            torch.softmax(log_policy, dim=-1)
            .reshape(self.dS, self.dA)
            .detach()
            .cpu()
            .numpy()
        )
        self.record_array("Policy", policy)
        values = self.env.compute_action_values(policy)
        self.record_array("Values", values)

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
