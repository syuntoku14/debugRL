from copy import deepcopy

import torch
from torch import nn

from shinrl.solvers import BaseSolver

OPTIONS = {
    "activation": "ReLU",
    "hidden": 128,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cuda",
    "optimizer": "Adam",
    "lr": 3e-4,
    # coefficient of PG. "Q" or "A". See https://arxiv.org/abs/1506.02438 for details.
    "coef": "Q",
}


def fc_net(env, num_output, hidden=256, depth=1, act_layer=nn.ReLU):
    modules = []
    obs_shape = env.observation_space.shape[0]
    if depth > 0:
        modules.append(nn.Linear(obs_shape, hidden))
        for _ in range(depth - 1):
            modules += [act_layer(), nn.Linear(hidden, hidden)]
        modules += [act_layer(), nn.Linear(hidden, num_output)]
    else:
        modules.append(nn.Linear(obs_shape, num_output))
    return nn.Sequential(*modules)


def conv_net(env, num_output, hidden=32, depth=1, act_layer=nn.ReLU):
    # this assumes image shape == (1, 28, 28)
    conv_modules = [
        nn.Conv2d(1, 10, kernel_size=5, stride=2),
        nn.Conv2d(10, 20, kernel_size=5, stride=2),
        nn.Flatten(),
    ]
    fc_modules = []
    if depth > 0:
        fc_modules.append(nn.Linear(320, hidden))
        for _ in range(depth - 1):
            fc_modules += [act_layer(), nn.Linear(hidden, hidden)]
        fc_modules += [act_layer(), nn.Linear(hidden, num_output)]
    else:
        fc_modules.append(nn.Linear(320, num_output))
    return nn.Sequential(*(conv_modules + fc_modules))


class Solver(BaseSolver):
    default_options = deepcopy(BaseSolver.default_options)
    default_options.update(OPTIONS)

    def initialize(self, options={}):
        super().initialize(options)
        self.device = self.solve_options["device"]
        self.all_obss = torch.tensor(
            self.env.all_observations, dtype=torch.float32, device=self.device
        )

        self.optimizer = getattr(torch.optim, self.solve_options["optimizer"])
        act_layer = getattr(nn, self.solve_options["activation"])

        net = fc_net if len(self.env.observation_space.shape) == 1 else conv_net
        self.policy_network = net(
            self.env,
            self.env.action_space.n,
            hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            act_layer=act_layer,
        ).to(self.device)
        self.policy_optimizer = self.optimizer(
            self.policy_network.parameters(), lr=self.solve_options["lr"]
        )
