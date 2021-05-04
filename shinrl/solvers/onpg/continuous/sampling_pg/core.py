import itertools
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

from shinrl import utils
from shinrl.solvers import Solver

OPTIONS = {
    "num_samples": 20,
    # Fitted iteration settings
    "activation": "tanh",
    "hidden": 256,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "critic_loss": "mse",  # mse or huber
    "optimizer": "Adam",
    "lr": 3e-4,
    "num_minibatches": 1,
    "vf_coef": 0.5,
    # coefficient of PG. "Q" or "A" or "GAE". See https://arxiv.org/abs/1506.02438 for details.
    "coef": "GAE",
    "td_lam": 0.95,
    "clip_grad": True,
    "clip_ratio": 0.2,  # for PPO
    "ent_coef": 0.001,
}


def fc_net(env, hidden, depth, act_layer):
    if act_layer == "tanh":
        act_layer = nn.Tanh
    elif act_layer == "relu":
        act_layer = nn.ReLU
    else:
        raise ValueError("Invalid activation layer.")
    modules = [
        nn.Linear(env.observation_space.shape[0], hidden),
    ]
    for _ in range(depth - 1):
        modules += [act_layer(), nn.Linear(hidden, hidden)]
    modules.append(act_layer())
    return nn.Sequential(*modules), None


def conv_net(env, hidden, depth, act_layer):
    if act_layer == "tanh":
        act_layer = nn.Tanh
    elif act_layer == "relu":
        act_layer = nn.ReLU
    else:
        raise ValueError("Invalid activation layer.")
    # this assumes image shape == (1, 28, 28)
    n_acts = env.action_space.n
    conv_modules = [
        nn.Conv2d(1, 10, kernel_size=5, stride=2),
        nn.Conv2d(10, 20, kernel_size=5, stride=2),
        nn.Flatten(),
    ]
    fc_modules = []
    fc_modules.append(nn.Linear(320, hidden))
    for _ in range(depth - 1):
        fc_modules += [act_layer(), nn.Linear(hidden, hidden)]
    fc_modules.append(act_layer())
    conv_modules = nn.Sequential(*conv_modules)
    fc_modules = nn.Sequential(*fc_modules)
    return fc_modules, conv_modules


class ValueNet(nn.Module):
    def __init__(self, env, solve_options):
        super().__init__()
        net = fc_net if len(env.observation_space.shape) == 1 else conv_net
        self.fc, self.conv = net(
            env,
            solve_options["hidden"],
            solve_options["depth"],
            solve_options["activation"],
        )
        self.out = nn.Linear(solve_options["hidden"], 1)

    def forward(self, obss):
        x = self.conv(obss) if self.conv is not None else obss
        x = self.fc(x)
        return self.out(x)


class PolicyNet(nn.Module):
    def __init__(self, env, solve_options):
        super().__init__()
        net = fc_net if len(env.observation_space.shape) == 1 else conv_net
        self.fc, self.conv = net(
            env,
            solve_options["hidden"],
            solve_options["depth"],
            solve_options["activation"],
        )
        act_dim = env.action_space.shape[0]
        log_std = np.zeros(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_layer = nn.Linear(solve_options["hidden"], act_dim)
        self.solve_options = solve_options

    def forward(self, obs):
        pi_dist = self.compute_pi_distribution(obs)
        pi_action = pi_dist.sample()
        logp_pi = self.compute_logp_pi(pi_dist, pi_action)
        return pi_action, logp_pi

    def compute_pi_distribution(self, obss):
        if self.conv is not None:
            x = self.conv(obss)
        else:
            x = obss
        net_out = self.fc(x)
        mu = self.mu_layer(net_out)
        std = torch.exp(self.log_std).expand_as(mu)
        pi_dist = Normal(mu, std)
        return pi_dist

    def compute_logp_pi(self, pi_dist, pi_action):
        logp_pi = pi_dist.log_prob(pi_action).sum(-1)
        return logp_pi


class Solver(Solver):
    def initialize(self, options={}):
        assert isinstance(self.env.action_space, gym.spaces.Box)
        self.solve_options.update(OPTIONS)
        super().initialize(options)
        self.device = self.solve_options["device"]

        # set networks
        if self.solve_options["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam
        else:
            self.optimizer = torch.optim.RMSprop

        # set critic loss
        if self.solve_options["critic_loss"] == "mse":
            self.critic_loss = F.mse_loss
        elif self.solve_options["critic_loss"] == "huber":
            self.critic_loss = F.smooth_l1_loss
        else:
            raise ValueError("Invalid critic_loss")

        # set value network
        self.value_network = ValueNet(self.env, self.solve_options).to(self.device)
        self.policy_network = PolicyNet(self.env, self.solve_options).to(self.device)
        self.params = itertools.chain(
            self.value_network.parameters(), self.policy_network.parameters()
        )
        self.optimizer = self.optimizer(self.params, lr=self.solve_options["lr"])

        # Collect random samples in advance
        if self.is_tabular:
            self.all_obss = torch.tensor(
                self.env.all_observations, dtype=torch.float32, device=self.device
            )
            self.all_actions = (
                torch.tensor(
                    self.env.all_actions, dtype=torch.float32, device=self.device
                )
                .repeat(self.dS, 1)
                .reshape(self.dS, self.dA)
            )  # dS x dA
            self.set_tb_values_policy()

    def collect_samples(self, num_samples):
        if self.is_tabular:
            return utils.collect_samples(
                self.env, utils.get_tb_action, num_samples, policy=self.tb_policy
            )
        else:
            return utils.collect_samples(self.env, self.get_action_gym, num_samples)

    def save(self, path):
        data = {
            "vnet": self.value_network.state_dict(),
            "polnet": self.policy_network.state_dict(),
            "opt": self.optimizer.state_dict(),
        }
        super().save(path, data)

    def load(self, path):
        data = super().load(path, device=self.device)
        self.value_network.load_state_dict(data["vnet"])
        self.policy_network.load_state_dict(data["polnet"])
        self.optimizer.load_state_dict(data["opt"])
