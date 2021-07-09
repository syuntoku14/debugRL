import itertools
import os
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

from shinrl import utils
from shinrl.solvers import BaseSolver

OPTIONS = {
    "num_samples": 1,
    "log_std_max": 2,
    "log_std_min": -20,
    "er_coef": 0.2,
    # Fitted iteration settings
    "activation": "ReLU",
    "hidden": 256,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 1e-3,
    "minibatch_size": 100,
    "critic_loss": "mse_loss",
    "optimizer": "Adam",
    "buffer_size": 1e6,
    "polyak": 0.995,
    "replay_start": 10000,
}


def fc_net(env, in_dim, hidden, depth, act_layer):
    act_layer = getattr(nn, act_layer)
    modules = [
        nn.Linear(env.observation_space.shape[0] + in_dim, hidden),
    ]
    for _ in range(depth - 1):
        modules += [act_layer(), nn.Linear(hidden, hidden)]
    modules.append(act_layer())
    return nn.Sequential(*modules)


class ValueNet(nn.Module):
    def __init__(self, env, solve_options):
        super().__init__()
        self.fc = fc_net(
            env,
            env.action_space.shape[0],
            solve_options["hidden"],
            solve_options["depth"],
            solve_options["activation"],
        )
        self.out = nn.Linear(solve_options["hidden"], 1)

    def forward(self, obss, actions=None):
        x = self.fc(torch.cat([obss, actions], -1))
        return self.out(x)


class PolicyNet(nn.Module):
    def __init__(self, env, solve_options):
        super().__init__()
        self.fc = fc_net(
            env,
            0,
            solve_options["hidden"],
            solve_options["depth"],
            solve_options["activation"],
        )
        act_dim = env.action_space.shape[0]
        act_limit = env.action_space.high[0]
        self.mu_layer = nn.Linear(solve_options["hidden"], act_dim)
        self.log_std_layer = nn.Linear(solve_options["hidden"], act_dim)
        self.act_limit = act_limit
        self.solve_options = solve_options

    def forward(self, obs):
        pi_dist = self.compute_pi_distribution(obs)
        pi_action = pi_dist.rsample()

        logp_pi = self.compute_logp_pi(pi_dist, pi_action)
        pi_action = self.squash_action(pi_action)
        return pi_action, logp_pi

    def squash_action(self, pi_action):
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action

    def compute_pi_distribution(self, obss):
        net_out = self.fc(obss)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(
            log_std,
            self.solve_options["log_std_min"],
            self.solve_options["log_std_max"],
        )
        std = torch.exp(log_std)
        # Pre-squash distribution and sample
        pi_dist = Normal(mu, std)
        return pi_dist

    def compute_logp_pi(self, pi_dist, pi_action):
        logp_pi = pi_dist.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
            axis=-1
        )
        return logp_pi


class Solver(BaseSolver):
    default_options = deepcopy(BaseSolver.default_options)
    default_options.update(OPTIONS)

    def initialize(self, options={}):
        assert isinstance(self.env.action_space, gym.spaces.Box)
        super().initialize(options)
        self.device = self.solve_options["device"]
        self.optimizer = getattr(torch.optim, self.solve_options["optimizer"])
        self.critic_loss = getattr(F, self.solve_options["critic_loss"])

        # set value network
        self.value_network = ValueNet(self.env, self.solve_options).to(self.device)
        self.value_network2 = ValueNet(self.env, self.solve_options).to(self.device)
        self.target_value_network = deepcopy(self.value_network)
        self.target_value_network2 = deepcopy(self.value_network2)

        # Freeze target networks
        for p1, p2 in zip(
            self.target_value_network.parameters(),
            self.target_value_network2.parameters(),
        ):
            p1.requires_grad = False
            p2.requires_grad = False

        self.val_params = itertools.chain(
            self.value_network.parameters(), self.value_network2.parameters()
        )
        self.value_optimizer = self.optimizer(
            self.val_params, lr=self.solve_options["lr"]
        )

        # actor network
        self.policy_network = PolicyNet(self.env, self.solve_options).to(self.device)
        self.policy_optimizer = self.optimizer(
            self.policy_network.parameters(), lr=self.solve_options["lr"]
        )

        # Collect random samples in advance
        self.buffer = utils.make_replay_buffer(
            self.env, self.solve_options["buffer_size"]
        )
        utils.collect_samples(self.env, self.get_action, num_samples=self.solve_options["replay_start"], buffer=self.buffer)