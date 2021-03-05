from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from shinrl.solvers import Solver
from shinrl.utils import softmax_policy

OPTIONS = {
    "activation": "relu",
    "hidden": 128,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cuda",
    "optimizer": "Adam",
    "lr": 3e-4,
    # coefficient of PG. "Q" or "A". See https://arxiv.org/abs/1506.02438 for details.
    "coef": "Q"
}


def fc_net(env, num_output, hidden=256, depth=1, activation="tanh"):
    if activation == "tanh":
        act_layer = nn.Tanh
    elif activation == "relu":
        act_layer = nn.ReLU
    else:
        raise ValueError("Activation layer is not valid.")

    modules = []
    if depth == 0:
        modules.append(
            nn.Linear(env.observation_space.shape[0], num_output))
    elif depth > 0:
        modules.append(
            nn.Linear(env.observation_space.shape[0], hidden))
        for _ in range(depth-1):
            modules.append(act_layer())
            modules.append(nn.Linear(hidden, hidden))
        modules.append(act_layer())
        modules.append(nn.Linear(hidden, num_output))
    else:
        raise ValueError("Invalid depth of fc layer")
    return nn.Sequential(*modules)


def conv_net(env, num_output, hidden=32, depth=1, activation="tanh"):
    # this assumes image shape == (1, 28, 28)
    if activation == "tanh":
        act_layer = nn.Tanh
    elif activation == "relu":
        act_layer = nn.ReLU
    else:
        raise ValueError("Activation layer is not valid.")

    # conv layers
    conv_modules = []
    conv_modules.append(nn.Conv2d(1, 10, kernel_size=5, stride=2))
    conv_modules.append(nn.Conv2d(10, 20, kernel_size=5, stride=2))
    conv_modules.append(nn.Flatten())

    fc_modules = []
    # fc layers
    if depth == 0:
        fc_modules.append(
            nn.Linear(320, num_output))
    elif depth > 0:
        fc_modules.append(
            nn.Linear(320, hidden))
        for _ in range(depth-1):
            fc_modules.append(act_layer())
            fc_modules.append(nn.Linear(hidden, hidden))
        fc_modules.append(act_layer())
        fc_modules.append(nn.Linear(hidden, num_output))
    else:
        raise ValueError("Invalid depth of fc layer")
    return nn.Sequential(*(conv_modules+fc_modules))


class Solver(Solver):
    def initialize(self, options={}):
        assert self.is_tabular
        self.solve_options.update(OPTIONS)
        super().initialize(options)
        self.device = self.solve_options["device"]
        self.all_obss = torch.tensor(
            self.env.all_observations, dtype=torch.float32, device=self.device)

        # set networks
        if self.solve_options["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam
        else:
            self.optimizer = torch.optim.RMSprop

        self.device = self.solve_options["device"]
        if len(self.env.observation_space.shape) == 1:
            net = fc_net
        else:
            net = conv_net

        # actor network
        self.policy_network = net(
            self.env, self.env.action_space.n,
            hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            activation=self.solve_options["activation"]).to(self.device)
        self.policy_optimizer = self.optimizer(self.policy_network.parameters(),
                                               lr=self.solve_options["lr"])

    def record_performance(self):
        if self.step % self.solve_options["record_performance_interval"] == 0:
            preference = self.policy_network(self.all_obss).reshape(
                self.dS, self.dA).detach().cpu().numpy()
            policy = self.compute_policy(preference)
            expected_return = self.env.compute_expected_return(policy)
            self.record_scalar("Return", expected_return, tag="Policy")
            self.record_array("Policy", policy)
            values = self.env.compute_action_values(
                policy, self.solve_options["discount"])
            self.record_array("Values", values)

    def compute_policy(self, preference):
        # return softmax policy
        return softmax_policy(preference, beta=1.0)
