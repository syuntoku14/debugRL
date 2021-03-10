from copy import deepcopy
from scipy import stats
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .core import Solver
from shinrl import utils


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
        preference = self.policy_network(
            self.all_obss).reshape(self.dS, self.dA)  # SxA
        policy = torch.softmax(preference, dim=-1)  # SxA
        policy_np = policy.detach().cpu().numpy()  # SxA
        # compute action values
        action_values = self.env.compute_action_values(
            policy_np, discount)  # SxA
        action_values = torch.tensor(
            action_values, dtype=torch.float32, device=self.device)
        # compute stationary distribution
        visitation = self.env.compute_visitation(
            policy_np, discount).sum(axis=(0, 2))  # S
        visitation = torch.tensor(
            visitation, dtype=torch.float32, device=self.device)

        if self.solve_options["coef"] == "Q":
            coef = action_values
        elif self.solve_options["coef"] == "A":
            coef = action_values - \
                (action_values*policy).sum(dim=1, keepdims=True)
        else:
            raise ValueError("{} is an invalid option.".format(
                self.solve_options["coef"]))

        # update policy
        loss = -((coef*policy).sum(dim=1) * visitation).sum(dim=0)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss.detach().cpu().item()

    def set_tb_values_policy(self):
        prefs = self.policy_network(self.all_obss).detach().cpu().numpy()
        policy = utils.softmax_policy(prefs)
        self.record_array("Policy", policy)
        values = self.env.compute_action_values(policy)
        self.record_array("Values", values)

    def record_history(self):
        if self.step % self.solve_options["record_performance_interval"] == 0:
            expected_return = \
                self.env.compute_expected_return(self.tb_policy)
            self.record_scalar("Return", expected_return, tag="Policy")
