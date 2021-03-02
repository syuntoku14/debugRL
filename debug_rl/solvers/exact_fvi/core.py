import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from debug_rl.solvers import Solver
from debug_rl.utils import (
    boltzmann_softmax,
    mellow_max,
)


OPTIONS = {
    # CVI settings
    "er_coef": 0.2,
    "kl_coef": 0.1,
    "max_operator": "mellow_max",
    # Fitted iteration settings
    "activation": "relu",
    "hidden": 128,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cuda",
    "lr": 0.001,
    "minibatch_size": 32,
    "critic_loss": "mse",  # mse or huber
    "clip_grad": False,  # clip the gradient if True
    "optimizer": "Adam",  # Adam or RMSprop
}


def fc_net(env, hidden=256, depth=1, act_layer=nn.ReLU):
    modules = []
    obs_shape = env.observation_space.shape[0]
    n_acts = env.action_space.n
    if depth > 0:
        modules.append(nn.Linear(obs_shape, hidden))
        for _ in range(depth-1):
            modules += [act_layer(), nn.Linear(hidden, hidden)]
        modules += [act_layer(), nn.Linear(hidden, n_acts)]
    else:
        modules.append(nn.Linear(obs_shape, n_acts))
    return nn.Sequential(*modules)


def conv_net(env, hidden=32, depth=1, act_layer=nn.ReLU):
    # this assumes image shape == (1, 28, 28)
    n_acts = env.action_space.n
    conv_modules = [nn.Conv2d(1, 10, kernel_size=5, stride=2),
                    nn.Conv2d(10, 20, kernel_size=5, stride=2),
                    nn.Flatten()]
    fc_modules = []
    if depth > 0:
        fc_modules.append(nn.Linear(320, hidden))
        for _ in range(depth-1):
            fc_modules += [act_layer(), nn.Linear(hidden, hidden)]
        fc_modules += [act_layer(), nn.Linear(hidden, n_acts)]
    else:
        fc_modules.append(nn.Linear(320, n_acts))
    return nn.Sequential(*(conv_modules+fc_modules))


class Solver(Solver):
    def initialize(self, options={}):
        assert self.is_tabular
        self.solve_options.update(OPTIONS)
        super().initialize(options)
        self.device = self.solve_options["device"]
        self.all_obss = torch.tensor(self.env.all_observations,
                                     dtype=torch.float32, device=self.device)
        self.record_array("Values", np.zeros((self.dS, self.dA)))

        # set max_operator
        if self.solve_options["max_operator"] == "boltzmann_softmax":
            self.max_operator = boltzmann_softmax
        elif self.solve_options["max_operator"] == "mellow_max":
            self.max_operator = mellow_max
        else:
            raise ValueError("Invalid max_operator")

        # set networks
        if self.solve_options["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam
        elif self.solve_options["optimizer"] == "RMSprop":
            self.optimizer = torch.optim.RMSprop

        # set critic loss
        if self.solve_options["critic_loss"] == "mse":
            self.critic_loss = F.mse_loss
        elif self.solve_options["critic_loss"] == "huber":
            self.critic_loss = F.smooth_l1_loss
        else:
            raise ValueError("Invalid critic_loss")

        # set value network
        if self.solve_options["activation"] == "tanh":
            act_layer = nn.Tanh
        elif self.solve_options["activation"] == "relu":
            act_layer = nn.ReLU
        else:
            raise ValueError("Invalid activation layer.")

        net = fc_net if len(self.env.observation_space.shape) == 1 else conv_net
        self.value_network = net(
            self.env,
            hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            act_layer=act_layer).to(self.device)
        self.value_optimizer = self.optimizer(self.value_network.parameters(),
                                              lr=self.solve_options["lr"])

    def record_performance(self):
        values = self.value_network(self.all_obss).detach().cpu().numpy()
        policy = self.to_policy(values)
        self.record_array("Policy", policy)
        self.record_array("Values", values)
        if self.step % self.solve_options["record_performance_interval"] == 0:
            expected_return = self.env.compute_expected_return(policy)
            self.record_scalar("Return", expected_return, tag="Policy")
