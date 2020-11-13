import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F
from .core import Solver
from debug_rl.solvers import (OracleViSolver, OracleCviSolver)
from debug_rl.utils import get_all_observations


class ExactFittedSolver(Solver):
    def solve(self):
        """
        Do value iteration from the end of the episode with a value approximator.
        Store the results and backup to self.values

        Returns:
            values: SxA matrix
        """
        values = np.zeros((self.dS, self.dA))  # SxA
        obss = get_all_observations(self.env)
        obss = torch.tensor(obss, dtype=torch.float32, device=self.device)
        error = np.inf
        # start training
        for _ in tqdm(range(self.solve_options["num_trains"])):
            target = self.backup(values)  # SxA
            target = torch.tensor(
                target, dtype=torch.float32, device=self.device)
            self.update_network(target, obss)
            projected = self.value_network(obss).detach().cpu().numpy()
            error = np.abs(values - projected).max()
            values = projected
            if error < 1e-7:
                break

        print("Iteration finished with maximum error: ", error)
        self.values = values
        return values

    def update_network(self, target, obss):
        values = self.value_network(obss)
        loss = self.critic_loss(target, values)
        self.value_optimizer.zero_grad()
        loss.backward()
        if self.solve_options["clip_grad"]:
            for param in self.value_network.parameters():
                param.grad.data.clamp_(-1, 1)
        self.value_optimizer.step()
        return loss.detach().cpu().item()

    def reset_weights(self):
        for layer in self.value_network:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class ExactFittedViSolver(ExactFittedSolver, OracleViSolver):
    pass


class ExactFittedCviSolver(ExactFittedSolver, OracleCviSolver):
    pass
