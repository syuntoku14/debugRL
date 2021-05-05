from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from shinrl import utils
from shinrl.solvers import BaseSolver

OPTIONS = {
    "num_samples": 8,
    # Fitted iteration settings
    "activation": "ReLU",
    "hidden": 128,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cuda",
    "critic_loss": "mse_loss",
    "optimizer": "Adam",
    "lr": 3e-4,
    # coefficient of PG. "Q" or "A" or "GAE". See https://arxiv.org/abs/1506.02438 for details.
    "coef": "GAE",
    "last_val": "critic",  # last value of coef calculation. "oracle" or "critic"
    "on_off_coef": 0.2,
    "td_lam": 0.95,
    "er_coef": 0.01,
    "buffer_size": 1e6,
    "minibatch_size": 32,
    "target_update_interval": 100,
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
    def initialize(self, options={}):
        self.solve_options.update(OPTIONS)
        super().initialize(options)
        self.device = self.solve_options["device"]
        self.optimizer = getattr(torch.optim, self.solve_options["optimizer"])
        self.critic_loss = getattr(F, self.solve_options["critic_loss"])
        act_layer = getattr(nn, self.solve_options["activation"])

        net = fc_net if len(self.env.observation_space.shape) == 1 else conv_net
        self.value_network = net(
            self.env,
            self.env.action_space.n,
            hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            act_layer=act_layer,
        ).to(self.device)
        self.value_optimizer = self.optimizer(
            self.value_network.parameters(), lr=self.solve_options["lr"]
        )
        self.target_value_network = deepcopy(self.value_network)

        self.baseline_network = net(
            self.env,
            1,
            hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            act_layer=act_layer,
        ).to(self.device)
        self.baseline_optimizer = self.optimizer(
            self.baseline_network.parameters(), lr=self.solve_options["lr"]
        )

        # actor network
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

        # Collect random samples in advance
        if self.is_tabular:
            self.all_obss = torch.tensor(
                self.env.all_observations, dtype=torch.float32, device=self.device
            )
            self.set_tb_values_policy()
        self.buffer = utils.make_replay_buffer(
            self.env, self.solve_options["buffer_size"]
        )
        trajectory = self.collect_samples(self.solve_options["minibatch_size"])
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
