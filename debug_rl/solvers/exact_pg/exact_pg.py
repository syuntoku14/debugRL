from copy import deepcopy
from scipy import stats
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .core import Solver


class ExactPgSolver(Solver):
    def run(self, num_steps=10000):
        for _ in tqdm(range(num_steps)):
            self.record_performance()

            # ----- update networks -----
            policy_loss = self.update_actor()
            self.record_scalar("policy loss", policy_loss)

            self.step += 1

    def update_actor(self):
        discount = self.solve_options["discount"]
        preference = self.policy_network(
            self.all_obss).reshape(self.dS, self.dA)  # SxA
        policy = torch.softmax(preference, dim=-1)  # SxA

        # compute action values
        policy_np = policy.detach().cpu().numpy()  # SxA
        action_values = self.env.compute_action_values(
            policy_np, discount)  # SxA
        action_values = torch.tensor(
            action_values, dtype=torch.float32, device=self.device)

        # compute stationary distribution
        visitation = self.env.compute_visitation(
            policy_np, discount).sum(axis=(0, 2))  # S
        visitation = torch.tensor(
            visitation, dtype=torch.float32, device=self.device)

        # update policy
        loss = -((action_values*policy).sum(dim=1) * visitation).sum(dim=0)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss.detach().cpu().item()