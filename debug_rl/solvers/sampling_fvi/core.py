import pprint
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from debug_rl.solvers import Solver
from debug_rl.utils import (
    boltzmann_softmax,
    mellow_max,
    make_replay_buffer
)


OPTIONS = {
    "num_samples": 4,
    # CVI settings
    "alpha": 0.9,
    "beta": 1.0,
    "max_operator": "mellow_max",
    # Q settings
    "eps_start": 0.9,
    "eps_end": 0.05,
    "eps_decay": 200,
    # Fitted iteration settings
    "num_trains": 10000,
    "activation": "relu",
    "hidden": 128,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cuda",
    "lr": 0.001,
    "minibatch_size": 32,
    "critic_loss": "mse",  # mse or huber
    "clip_grad": False,  # clip the gradient if True
    "optimizer": "Adam",
    "buffer_size": 1e6,
    "use_double_estimation": False,
    "use_replay_buffer": True,
    "use_target_network": True,
    "target_update_interval": 100,
}


def fc_net(env, hidden=256, depth=1, activation="tanh"):
    if activation == "tanh":
        act_layer = nn.Tanh
    elif activation == "relu":
        act_layer = nn.ReLU
    else:
        raise ValueError("Activation layer is not valid.")

    modules = []
    if depth == 0:
        modules.append(
            nn.Linear(env.observation_space.shape[0],
                      env.action_space.n))
    elif depth > 0:
        modules.append(
            nn.Linear(env.observation_space.shape[0], hidden))
        for _ in range(depth-1):
            modules.append(act_layer())
            modules.append(nn.Linear(hidden, hidden))
        modules.append(act_layer())
        modules.append(nn.Linear(hidden, env.action_space.n))
    else:
        raise ValueError("Invalid depth of fc layer")
    return nn.Sequential(*modules)


def conv_net(env, hidden=32, depth=1, activation="tanh"):
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
            nn.Linear(320, env.action_space.n))
    elif depth > 0:
        fc_modules.append(
            nn.Linear(320, hidden))
        for _ in range(depth-1):
            fc_modules.append(act_layer())
            fc_modules.append(nn.Linear(hidden, hidden))
        fc_modules.append(act_layer())
        fc_modules.append(nn.Linear(hidden, env.action_space.n))
    else:
        raise ValueError("Invalid depth of fc layer")
    return nn.Sequential(*(conv_modules+fc_modules))


class Solver(Solver):
    def set_options(self, options={}):
        self.solve_options.update(OPTIONS)
        super().set_options(options)
        self.device = self.solve_options["device"]

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
        else:
            self.optimizer = torch.optim.RMSprop

        self.device = self.solve_options["device"]
        if len(self.env.observation_space.shape) == 1:
            net = fc_net
        else:
            net = conv_net

        # set critic loss
        if self.solve_options["critic_loss"] == "mse":
            self.critic_loss = F.mse_loss
        elif self.solve_options["critic_loss"] == "huber":
            self.critic_loss = F.smooth_l1_loss
        else:
            raise ValueError("Invalid critic_loss")

        # set value network
        self.value_network = net(
            self.env,
            hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            activation=self.solve_options["activation"]).to(self.device)
        self.value_optimizer = self.optimizer(self.value_network.parameters(),
                                              lr=self.solve_options["lr"])
        self.target_value_network = deepcopy(self.value_network)

        # network for double estimation
        self.value_network2 = net(
            self.env,
            hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            activation=self.solve_options["activation"]).to(self.device)
        self.value_optimizer2 = self.optimizer(self.value_network2.parameters(),
                                               lr=self.solve_options["lr"])
        self.target_value_network2 = deepcopy(self.value_network2)

        # set replay buffer
        self.buffer = make_replay_buffer(
            self.env, self.solve_options["buffer_size"])

        print("{} solve_options:".format(type(self).__name__))
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.solve_options)

    def record_performance(self, k, eval_policy, tensor_all_obss, force=True):
        if k % self.solve_options["record_performance_interval"] == 0 or force:
            expected_return = \
                self.compute_expected_return(eval_policy)
            self.record_scalar("Return mean", expected_return, x=k)

            aval = self.compute_action_values(eval_policy)
            values = self.value_network(tensor_all_obss).reshape(
                self.dS, self.dA).detach().cpu().numpy()
            self.record_scalar("Q error", ((aval-values)**2).mean(), x=k)
            self.record_array("values", values, x=k)
            self.record_array("policy", eval_policy, x=k)
