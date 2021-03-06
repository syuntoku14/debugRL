import itertools
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from shinrl import utils
from shinrl.solvers import BaseSolver

OPTIONS = {
    "num_samples": 4,
    "er_coef": 0.2,
    # Fitted iteration settings
    "activation": "ReLU",
    "hidden": 256,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 1e-3,
    "minibatch_size": 32,
    "critic_loss": "mse_loss",
    "optimizer": "Adam",
    "buffer_size": 1e6,
    "polyak": 0.995,
    "num_random_samples": 10000,
}


def fc_net(env, hidden=256, depth=1, act_layer=nn.ReLU):
    modules = []
    obs_shape = env.observation_space.shape[0]
    n_acts = env.action_space.n
    if depth > 0:
        modules.append(nn.Linear(obs_shape, hidden))
        for _ in range(depth - 1):
            modules += [act_layer(), nn.Linear(hidden, hidden)]
        modules += [act_layer(), nn.Linear(hidden, n_acts)]
    else:
        modules.append(nn.Linear(obs_shape, n_acts))
    return nn.Sequential(*modules)


def conv_net(env, hidden=32, depth=1, act_layer=nn.ReLU):
    # this assumes image shape == (1, 28, 28)
    n_acts = env.action_space.n
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
        fc_modules += [act_layer(), nn.Linear(hidden, n_acts)]
    else:
        fc_modules.append(nn.Linear(320, n_acts))
    return nn.Sequential(*(conv_modules + fc_modules))


class Solver(BaseSolver):
    default_options = deepcopy(BaseSolver.default_options)
    default_options.update(OPTIONS)

    def initialize(self, options={}):
        super().initialize(options)

        self.device = self.solve_options["device"]
        self.optimizer = getattr(torch.optim, self.solve_options["optimizer"])
        self.critic_loss = getattr(F, self.solve_options["critic_loss"])
        act_layer = getattr(nn, self.solve_options["activation"])

        net = fc_net if len(self.env.observation_space.shape) == 1 else conv_net
        self.value_network = net(
            self.env,
            hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            act_layer=act_layer,
        ).to(self.device)
        self.value_network2 = net(
            self.env,
            hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            act_layer=act_layer,
        ).to(self.device)
        self.target_value_network = deepcopy(self.value_network)
        self.target_value_network2 = deepcopy(self.value_network)

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
        self.policy_network = net(
            self.env,
            hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            act_layer=act_layer,
        ).to(self.device)
        self.policy_optimizer = self.optimizer(
            self.policy_network.parameters(), lr=self.solve_options["lr"]
        )

        # Collect random samples in advance
        if self.is_tabular:
            self.all_obss = torch.tensor(
                self.env.all_observations, dtype=torch.float32, device=self.device
            )
            self.set_tb_values_policy()
        self.buffer = utils.make_replay_buffer(
            self.env, self.solve_options["buffer_size"]
        )
        trajectory = self.collect_samples(self.solve_options["minibatch_size"] * 10)
        self.buffer.add(**trajectory)

    def collect_samples(self, num_samples):
        if self.is_tabular:
            return utils.collect_samples(
                self.env,
                utils.get_tb_action,
                self.solve_options["num_samples"],
                policy=self.tb_policy,
            )
        else:
            return utils.collect_samples(
                self.env, self.get_action_gym, self.solve_options["num_samples"]
            )
