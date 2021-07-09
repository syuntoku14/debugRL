from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cpprb import ReplayBuffer, create_env_dict

from shinrl import utils
from shinrl.solvers import BaseSolver

OPTIONS = {
    "num_samples": 4,
    "er_coef": 0.003,
    "kl_coef": 0.027,
    "clipping": -1,
    "max_operator": "mellow_max",
    # Q settings
    "eps_start": 1.0,
    "eps_end": 0.01,
    "eps_decay": 0.01,
    "eps_period": 2.5*(10**5),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 5e-5,
    "minibatch_size": 32,
    "init_samples": 10000,
    "critic_loss": "mse_loss",
    "optimizer": "Adam",
    "buffer_size": 1e6,
    "target_update_interval": 8000,
}


class Linear0(nn.Linear):
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x.reshape(-1, 4, 84, 84) * self.scale


def nature_dqn(env, frames=4):
    return nn.Sequential(
        Scale(1 / 255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        Linear0(512, env.action_space.n)
    )


class Solver(BaseSolver):
    default_options = deepcopy(BaseSolver.default_options)
    default_options.update(OPTIONS)

    def initialize(self, options={}):
        super().initialize(options)

        self.device = self.solve_options["device"]
        self.max_operator = getattr(utils, self.solve_options["max_operator"])
        self.optimizer = getattr(torch.optim, self.solve_options["optimizer"])
        self.critic_loss = getattr(F, self.solve_options["critic_loss"])

        self.value_network = nature_dqn(self.env).to(self.device)
        self.value_optimizer = self.optimizer(
            self.value_network.parameters(), lr=self.solve_options["lr"]
        )
        self.target_value_network = deepcopy(self.value_network)

        # Collect random samples in advance
        env_dict = create_env_dict(self.env)
        del env_dict["next_obs"]
        env_dict.update(
            {
                "obs": {"dtype": np.ubyte, "shape": (84, 84, 4)},
                "log_prob": {"dtype": np.float32, "shape": 1},
                "timeout": {"dtype": np.bool, "shape": 1},
            }
        )
        self.buffer = ReplayBuffer(int(self.solve_options["buffer_size"]), env_dict, next_of="obs", stack_compress=("obs"))
        utils.collect_samples(self.env, self.explore, num_samples=self.solve_options["minibatch_size"], buffer=self.buffer)
