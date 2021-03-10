from copy import deepcopy
import torch
import itertools
from torch import nn
import torch.nn.functional as F
from shinrl.solvers import Solver
from shinrl import utils

OPTIONS = {
    "num_samples": 4,
    # CVI settings
    "er_coef": 0.2,
    # Fitted iteration settings
    "activation": "relu",
    "hidden": 256,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 1e-3,
    "minibatch_size": 32,
    "critic_loss": "mse",  # mse or huber
    "optimizer": "Adam",
    "buffer_size": 1e6,
    "polyak": 0.995,
    "num_random_samples": 10000
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
        self.solve_options.update(OPTIONS)
        super().initialize(options)
        self.device = self.solve_options["device"]

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

        net = fc_net if len(
            self.env.observation_space.shape) == 1 else conv_net
        self.value_network = net(
            self.env, hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            act_layer=act_layer).to(self.device)
        self.value_network2 = net(
            self.env, hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            act_layer=act_layer).to(self.device)
        self.target_value_network = deepcopy(self.value_network)
        self.target_value_network2 = deepcopy(self.value_network)

        # Freeze target networks
        for p1, p2 in zip(self.target_value_network.parameters(),
                          self.target_value_network2.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False
        self.val_params = itertools.chain(self.value_network.parameters(),
                                          self.value_network2.parameters())
        self.value_optimizer = self.optimizer(
            self.val_params, lr=self.solve_options["lr"])

        # actor network
        self.policy_network = net(
            self.env, hidden=self.solve_options["hidden"],
            depth=self.solve_options["depth"],
            act_layer=act_layer).to(self.device)
        self.policy_optimizer = self.optimizer(self.policy_network.parameters(),
                                               lr=self.solve_options["lr"])

        # Collect random samples in advance
        if self.is_tabular:
            self.all_obss = torch.tensor(self.env.all_observations,
                                         dtype=torch.float32, device=self.device)
            self.set_tb_values_policy()
        self.buffer = utils.make_replay_buffer(
            self.env, self.solve_options["buffer_size"])
        trajectory = self.collect_samples(
            self.solve_options["minibatch_size"]*10)
        self.buffer.add(**trajectory)

    def collect_samples(self, num_samples):
        if self.is_tabular:
            return utils.collect_samples(
                self.env, utils.get_tb_action, self.solve_options["num_samples"],
                policy=self.tb_policy)
        else:
            return utils.collect_samples(
                self.env, self.get_action_gym, self.solve_options["num_samples"])

    def save(self, path):
        data = {
            "vnet1": self.value_network.state_dict(),
            "vnet2": self.value_network2.state_dict(),
            "valopt": self.value_optimizer.state_dict(),
            "targ_vnet1": self.target_value_network.state_dict(),
            "targ_vnet2": self.target_value_network2.state_dict(),
            "polnet": self.policy_network.state_dict(),
            "polopt": self.policy_optimizer.state_dict(),
            "buf_data": self.buffer.get_all_transitions()
        }
        super().save(path, data)

    def load(self, path):
        data = super().load(path, device=self.device)
        self.value_network.load_state_dict(data["vnet1"])
        self.value_network2.load_state_dict(data["vnet2"])
        self.value_optimizer.load_state_dict(data["valopt"])
        self.target_value_network.load_state_dict(data["targ_vnet1"])
        self.target_value_network2.load_state_dict(data["targ_vnet2"])
        self.policy_network.load_state_dict(data["polnet"])
        self.policy_optimizer.load_state_dict(data["polopt"])
        self.buffer = utils.make_replay_buffer(
            self.env, self.solve_options["buffer_size"])
        self.buffer.add(**data["buf_data"])
