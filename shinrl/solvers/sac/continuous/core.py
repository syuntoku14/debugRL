from copy import deepcopy
import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from shinrl.solvers import Solver
from shinrl import utils


OPTIONS = {
    "num_samples": 4,
    "log_std_max": 2,
    "log_std_min": -20,
    # CVI settings
    "er_coef": 0.2,
    # Fitted iteration settings
    "activation": "relu",
    "hidden": 128,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 0.001,
    "minibatch_size": 32,
    "critic_loss": "mse",  # mse or huber
    "clip_grad": False,  # clip the gradient if True
    "optimizer": "Adam",
    "buffer_size": 1e6,
    "polyak": 0.995,
}


def fc_net(env, in_dim, hidden, depth, act_layer):
    if act_layer == "tanh":
        act_layer = nn.Tanh
    elif act_layer == "relu":
        act_layer = nn.ReLU
    else:
        raise ValueError("Invalid activation layer.")
    modules = [nn.Linear(env.observation_space.shape[0] + in_dim, hidden), ]
    for _ in range(depth-1):
        modules += [act_layer(), nn.Linear(hidden, hidden)]
    modules.append(act_layer())
    return nn.Sequential(*modules), None


def conv_net(env, in_dim, hidden, depth, act_layer):
    if act_layer == "tanh":
        act_layer = nn.Tanh
    elif act_layer == "relu":
        act_layer = nn.ReLU
    else:
        raise ValueError("Invalid activation layer.")
    # this assumes image shape == (1, 28, 28)
    n_acts = env.action_space.n
    conv_modules = [nn.Conv2d(1, 10, kernel_size=5, stride=2),
                    nn.Conv2d(10, 20, kernel_size=5, stride=2),
                    nn.Flatten()]
    fc_modules = []
    fc_modules.append(nn.Linear(320+in_dim, hidden))
    for _ in range(depth-1):
        fc_modules += [act_layer(), nn.Linear(hidden, hidden)]
    fc_modules.append(act_layer())
    conv_modules = nn.Sequential(*conv_modules)
    fc_modules = nn.Sequential(*fc_modules)
    return fc_modules, conv_modules


class ValueNet(nn.Module):
    def __init__(self, env, solve_options):
        super().__init__()
        net = fc_net if len(env.observation_space.shape) == 1 else conv_net
        self.fc, self.conv = net(
            env, env.action_space.shape[0],
            solve_options["hidden"], solve_options["depth"],
            solve_options["activation"])
        self.out = nn.Linear(solve_options["hidden"], 1)

    def forward(self, obss, actions=None):
        x = self.conv(obss) if self.conv is not None else obss
        x = self.fc(torch.cat([x, actions], -1))
        return self.out(x)


class PolicyNet(nn.Module):
    def __init__(self, env, solve_options):
        super().__init__()
        net = fc_net if len(env.observation_space.shape) == 1 else conv_net
        self.fc, self.conv = net(
            env, 0, solve_options["hidden"], solve_options["depth"],
            solve_options["activation"])
        act_dim = env.action_space.shape[0]
        act_limit = env.action_space.high[0]
        self.mu_layer = nn.Linear(solve_options["hidden"], act_dim)
        self.log_std_layer = nn.Linear(solve_options["hidden"], act_dim)
        self.act_limit = act_limit
        self.solve_options = solve_options

    def forward(self, obs, return_raw=False):
        pi_dist = self.compute_pi_distribution(obs)
        pi_action = pi_dist.rsample()

        logp_pi = self.compute_logp_pi(pi_dist, pi_action)

        if return_raw:
            return pi_action, logp_pi
        else:
            pi_action = self.squash_action(pi_action)
            return pi_action, logp_pi

    def squash_action(self, pi_action):
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action

    def unsquash_action(self, pi_action):
        pi_action = pi_action / self.act_limit
        # to avoid numerical instability
        pi_action = pi_action - torch.sign(pi_action) * 1e-5
        pi_action = torch.atanh(pi_action)
        return pi_action

    def compute_pi_distribution(self, obss):
        if self.conv is not None:
            x = self.conv(obss)
        else:
            x = obss
        net_out = self.fc(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(
            log_std, self.solve_options["log_std_min"], self.solve_options["log_std_max"])
        std = torch.exp(log_std)
        # Pre-squash distribution and sample
        pi_dist = Normal(mu, std)
        return pi_dist

    def compute_logp_pi(self, pi_dist, pi_action):
        logp_pi = pi_dist.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - pi_action -
                       F.softplus(-2*pi_action))).sum(axis=-1)
        return logp_pi


class Solver(Solver):
    def initialize(self, options={}):
        assert isinstance(self.env.action_space, gym.spaces.Box)
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
        self.value_network = ValueNet(
            self.env, self.solve_options).to(self.device)
        self.value_optimizer = self.optimizer(self.value_network.parameters(),
                                              lr=self.solve_options["lr"])
        self.target_value_network = deepcopy(self.value_network)

        # network for double estimation
        self.value_network2 = ValueNet(
            self.env, self.solve_options).to(self.device)
        self.value_optimizer2 = self.optimizer(self.value_network2.parameters(),
                                               lr=self.solve_options["lr"])
        self.target_value_network2 = deepcopy(self.value_network2)

        # actor network
        self.policy_network = PolicyNet(
            self.env, self.solve_options).to(self.device)
        self.policy_optimizer = self.optimizer(self.policy_network.parameters(),
                                               lr=self.solve_options["lr"])

        # Collect random samples in advance
        if self.is_tabular:
            self.all_obss = torch.tensor(
                self.env.all_observations, dtype=torch.float32, device=self.device)
            self.all_actions = torch.tensor(
                self.env.all_actions, dtype=torch.float32,
                device=self.device).repeat(self.dS, 1).reshape(self.dS, self.dA)  # dS x dA
            self.set_tb_values_policy()
        self.buffer = utils.make_replay_buffer(
            self.env, self.solve_options["buffer_size"])
        trajectory = self.collect_samples(self.solve_options["minibatch_size"])
        self.buffer.add(**trajectory)

    def collect_samples(self, num_samples):
        if self.is_tabular:
            return utils.collect_samples(
                self.env, utils.get_tb_action, self.solve_options["num_samples"],
                policy=self.tb_policy)
        else:
            return utils.collect_samples(
                self.env, self.get_action_gym, self.solve_options["num_samples"])
