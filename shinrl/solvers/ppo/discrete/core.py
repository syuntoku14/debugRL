from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import itertools
from shinrl.solvers import Solver

OPTIONS = {
    "num_samples": 20,
    # Fitted iteration settings
    "activation": "tanh",
    "hidden": 256,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "critic_loss": "mse",  # mse or huber
    "optimizer": "Adam",
    # PPO settings
    "lr": 3e-4,
    "num_minibatches": 1,
    "vf_coef": 0.5,
    "td_lam": 0.95,
    "clip_ratio": 0.2,
    "ent_coef": 0.001,
    "clip_grad": True
}


def fc_net(env, num_output, hidden=256, depth=1, act_layer=nn.ReLU):
    modules = []
    obs_shape = env.observation_space.shape[0]
    if depth > 0:
        modules.append(nn.Linear(obs_shape, hidden))
        for _ in range(depth-1):
            modules += [act_layer(), nn.Linear(hidden, hidden)]
        modules += [act_layer(), nn.Linear(hidden, num_output)]
    else:
        modules.append(nn.Linear(obs_shape, num_output))
    return nn.Sequential(*modules)


def conv_net(env, num_output, hidden=32, depth=1, act_layer=nn.ReLU):
    # this assumes image shape == (1, 28, 28)
    conv_modules = [nn.Conv2d(1, 10, kernel_size=5, stride=2),
                    nn.Conv2d(10, 20, kernel_size=5, stride=2),
                    nn.Flatten()]
    fc_modules = []
    if depth > 0:
        fc_modules.append(nn.Linear(320, hidden))
        for _ in range(depth-1):
            fc_modules += [act_layer(), nn.Linear(hidden, hidden)]
        fc_modules += [act_layer(), nn.Linear(hidden, num_output)]
    else:
        fc_modules.append(nn.Linear(320, num_output))
    return nn.Sequential(*(conv_modules+fc_modules))


class Solver(Solver):
    def initialize(self, options={}):
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
        if self.solve_options["activation"] == "tanh":
            act_layer = nn.Tanh
        elif self.solve_options["activation"] == "relu":
            act_layer = nn.ReLU
        else:
            raise ValueError("Invalid activation layer.")

        net = fc_net if len(
            self.env.observation_space.shape) == 1 else conv_net
        self.value_network = net(
            self.env, 1,
            hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            act_layer=act_layer).to(self.device)
        self.policy_network = net(
            self.env, self.env.action_space.n,
            hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            act_layer=act_layer).to(self.device)
        self.params = itertools.chain(
            self.value_network.parameters(),
            self.policy_network.parameters())
        self.optimizer = self.optimizer(self.params, lr=self.solve_options["lr"])

        if self.is_tabular:
            self.all_obss = torch.tensor(
                self.env.all_observations, dtype=torch.float32, device=self.device)

    def save(self, path):
        data = {
            "vnet": self.value_network.state_dict(),
            "polnet": self.policy_network.state_dict(),
            "opt": self.optimizer.state_dict(),
        }
        super().save(path, data)

    def load(self, path):
        data = super().load(path, device=self.device)
        self.value_network.load_state_dict(data["vnet"])
        self.policy_network.load_state_dict(data["polnet"])
        self.optimizer.load_state_dict(data["opt"])
