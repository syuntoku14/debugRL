import itertools
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from cpprb import ReplayBuffer, create_env_dict
from torch import nn

from shinrl import utils
from shinrl.solvers import BaseSolver

OPTIONS = {
    "num_samples": 1,
    "er_coef": 0.2,
    # Exploration
    "eps_start": 1.0,
    "eps_end": 0.1,
    "eps_decay": 0.01,
    "eps_period": 10 ** 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 2.5 * 1e-4,
    "minibatch_size": 32,
    "replay_start": 5000,
    "critic_loss": "mse_loss",
    "optimizer": "Adam",
    "buffer_size": 1e5,
    "target_update_interval": 1000,
}


class Permute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


def net(env, frames=4):
    in_channels = env.observation_space.shape[2]
    n_actions = env.action_space.n

    def size_linear_unit(size, kernel_size=3, stride=1):
        return (size - (kernel_size - 1) - 1) // stride + 1

    num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
    return nn.Sequential(
        Permute(),
        nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(num_linear_units, 128),
        nn.ReLU(),
        nn.Linear(128, env.action_space.n),
    )


class Solver(BaseSolver):
    default_options = deepcopy(BaseSolver.default_options)
    default_options.update(OPTIONS)

    def initialize(self, options={}):
        super().initialize(options)

        self.device = self.solve_options["device"]
        self.optimizer = getattr(torch.optim, self.solve_options["optimizer"])
        self.critic_loss = getattr(F, self.solve_options["critic_loss"])

        self.value_network = net(self.env).to(self.device)
        self.value_optimizer = self.optimizer(
            self.value_network.parameters(), lr=self.solve_options["lr"]
        )
        self.target_value_network = deepcopy(self.value_network)

        self.policy_network = net(self.env).to(self.device)
        self.policy_optimizer = self.optimizer(
            self.policy_network.parameters(), lr=self.solve_options["lr"]
        )

        # Collect random samples in advance
        env_dict = create_env_dict(self.env)
        del env_dict["next_obs"]
        env_dict.update(
            {
                "obs": {"dtype": np.ubyte, "shape": self.env.observation_space.shape},
                "log_prob": {"dtype": np.float32, "shape": 1},
                "timeout": {"dtype": np.bool, "shape": 1},
            }
        )
        self.buffer = ReplayBuffer(
            int(self.solve_options["buffer_size"]), env_dict, next_of="obs"
        )
        utils.collect_samples(
            self.env,
            self.explore,
            num_samples=self.solve_options["replay_start"],
            buffer=self.buffer,
        )
