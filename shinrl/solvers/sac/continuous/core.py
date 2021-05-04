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
from shinrl.solvers import Solver

OPTIONS = {
    "num_samples": 1,
    "log_std_max": 2,
    "log_std_min": -20,
    "er_coef": 0.2,
    # Fitted iteration settings
    "activation": "relu",
    "hidden": 256,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 1e-3,
    "minibatch_size": 100,
    "critic_loss": "mse",  # mse or huber
    "optimizer": "Adam",
    "buffer_size": 1e6,
    "polyak": 0.995,
    "num_random_samples": 10000,
}


def fc_net(env, in_dim, hidden, depth, act_layer):
    if act_layer == "tanh":
        act_layer = nn.Tanh
    elif act_layer == "relu":
        act_layer = nn.ReLU
    else:
        raise ValueError("Invalid activation layer.")
    modules = [
        nn.Linear(env.observation_space.shape[0] + in_dim, hidden),
    ]
    for _ in range(depth - 1):
        modules += [act_layer(), nn.Linear(hidden, hidden)]
    modules.append(act_layer())
    return nn.Sequential(*modules), None


def conv_net(env, in_dim, hidden, depth, act_layer):
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
    fc_modules.append(nn.Linear(320 + in_dim, hidden))
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
            env.action_space.shape[0],
            solve_options["hidden"],
            solve_options["depth"],
            solve_options["activation"],
        )
        self.out = nn.Linear(solve_options["hidden"], 1)

    def forward(self, obss, actions=None):
        x = self.conv(obss) if self.conv is not None else obss
        x = self.fc(torch.cat([x, actions], -1))
        return self.out(x)


class PolicyNet(nn.Module):
    def __init__(self, env, solve_options):
        super().__init__()
        net = fc_net if len(env.observation_space.shape) == 1 else conv_net
        self.fc, self.conv = net(
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
        if self.conv is not None:
            x = self.conv(obss)
        else:
            x = obss
        net_out = self.fc(x)
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
        self.buffer = utils.make_replay_buffer(
            self.env, self.solve_options["buffer_size"]
        )
        trajectory = self.collect_samples(self.solve_options["minibatch_size"] * 10)
        self.buffer.add(**trajectory)

    def collect_samples(self, num_samples):
        if self.is_tabular:
            return utils.collect_samples(
                self.env, utils.get_tb_action, num_samples, policy=self.tb_policy
            )
        else:
            return utils.collect_samples(self.env, self.get_action_gym, num_samples)

    def save(self, dir_name):
        data = {
            "vnet1": self.value_network.state_dict(),
            "vnet2": self.value_network2.state_dict(),
            "valopt": self.value_optimizer.state_dict(),
            "targ_vnet1": self.target_value_network.state_dict(),
            "targ_vnet2": self.target_value_network2.state_dict(),
            "polnet": self.policy_network.state_dict(),
            "polopt": self.policy_optimizer.state_dict(),
        }
        super().save(dir_name, data)
        self.buffer.save_transitions(os.path.join(dir_name, "rbuf.npz"), safe=True)

    def load(self, dir_name):
        data = super().load(dir_name, device=self.device)
        self.value_network.load_state_dict(data["vnet1"])
        self.value_network2.load_state_dict(data["vnet2"])
        self.value_optimizer.load_state_dict(data["valopt"])
        self.target_value_network.load_state_dict(data["targ_vnet1"])
        self.target_value_network2.load_state_dict(data["targ_vnet2"])
        self.policy_network.load_state_dict(data["polnet"])
        self.policy_optimizer.load_state_dict(data["polopt"])

        self.buffer = utils.make_replay_buffer(
            self.env, self.solve_options["buffer_size"]
        )
        self.buffer.load_transitions(os.path.join(dir_name, "rbuf.npz"))
