import torch
import torch.nn.functional as F
from torch import nn

from shinrl import utils
from shinrl.solvers import BaseSolver

OPTIONS = {
    # CVI settings
    "er_coef": 0.2,
    "kl_coef": 0.1,
    "max_operator": "mellow_max",
    "proj_iters": 10,
    # Fitted iteration settings
    "activation": "ReLU",
    "hidden": 128,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 0.001,
    "critic_loss": "mse_loss",
    "optimizer": "Adam",
    "noise_scale": 0.0,
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
    def initialize(self, options={}):
        assert self.is_tabular
        self.solve_options.update(OPTIONS)
        super().initialize(options)

        self.device = self.solve_options["device"]
        self.max_operator = getattr(utils, self.solve_options["max_operator"])
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
        self.value_optimizer = self.optimizer(
            self.value_network.parameters(), lr=self.solve_options["lr"]
        )
        self.all_obss = torch.tensor(
            self.env.all_observations, dtype=torch.float32, device=self.device
        )
        self.set_tb_values_policy()
