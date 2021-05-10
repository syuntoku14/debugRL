import itertools
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.beta import Beta

from shinrl import utils
from shinrl.solvers import BaseSolver

OPTIONS = {
    "activation": "ReLU",
    "hidden": 128,
    "depth": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "optimizer": "Adam",
    "lr": 3e-4,
    # coefficient of PG. "Q" or "A". See https://arxiv.org/abs/1506.02438 for details.
    "coef": "Q",
}


def fc_net(env, hidden, depth, act_layer):
    act_layer = getattr(nn, act_layer)
    modules = [
        nn.Linear(env.observation_space.shape[0], hidden),
    ]
    for _ in range(depth - 1):
        modules += [act_layer(), nn.Linear(hidden, hidden)]
    modules.append(act_layer())
    return nn.Sequential(*modules)


class PolicyNet(nn.Module):
    def __init__(self, env, solve_options):
        super().__init__()
        self.fc = fc_net(
            env,
            solve_options["hidden"],
            solve_options["depth"],
            solve_options["activation"],
        )
        act_dim = env.action_space.shape[0]
        self.device = solve_options["device"]
        self.high = torch.tensor(
            env.action_space.high[0], dtype=torch.float32, device=self.device
        )
        self.low = torch.tensor(
            env.action_space.low[0], dtype=torch.float32, device=self.device
        )
        self.log_alpha_layer = nn.Linear(solve_options["hidden"], act_dim)
        self.log_beta_layer = nn.Linear(solve_options["hidden"], act_dim)
        self.solve_options = solve_options

    def forward(self, obs):
        pi_dist = self.compute_distribution(obs)
        raw_action = pi_dist.rsample()
        logp_pi = self.compute_logp_pi(pi_dist, raw_action)
        return self.squash_action(raw_action), logp_pi

    def squash_action(self, action):
        return action * (self.high - self.low) + self.low

    def unsquash_action(self, action):
        action = (action - self.low) / (self.high - self.low)
        # to avoid numerical instability
        action = (
            action + (action == 0.0).float() * 1e-8 - (action == 1.0).float() * 1e-8
        )
        return action

    def compute_distribution(self, obss):
        net_out = self.fc(obss)
        alpha = self.log_alpha_layer(net_out).exp()
        beta = self.log_beta_layer(net_out).exp()
        pi_dist = Beta(alpha, beta)
        return pi_dist

    def compute_logp_pi(self, pi_dist, raw_action):
        logp_pi = pi_dist.log_prob(raw_action)
        logp_pi -= (self.high - self.low).log()
        return logp_pi


class Solver(BaseSolver):
    default_options = deepcopy(BaseSolver.default_options)
    default_options.update(OPTIONS)

    def initialize(self, options={}):
        assert isinstance(self.env.action_space, gym.spaces.Box)
        super().initialize(options)
        self.device = self.solve_options["device"]
        self.optimizer = getattr(torch.optim, self.solve_options["optimizer"])
        self.policy_network = PolicyNet(self.env, self.solve_options).to(self.device)
        self.policy_optimizer = self.optimizer(
            self.policy_network.parameters(), lr=self.solve_options["lr"]
        )

        self.all_obss = torch.tensor(
            self.env.all_observations, dtype=torch.float32, device=self.device
        )
        self.all_actions = (
            torch.tensor(self.env.all_actions, dtype=torch.float32, device=self.device)
            .repeat(self.dS, 1)
            .reshape(self.dS, self.dA)
        )  # dS x dA
        self.set_tb_values_policy()
