from copy import deepcopy
import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from debug_rl.solvers import Solver
from debug_rl.utils import (
    boltzmann_softmax,
    mellow_max,
    make_replay_buffer,
    collect_samples
)


OPTIONS = {
    "num_samples": 4,
    "log_std_max": 2,
    "log_std_min": -20,
    # CVI settings
    "sigma": 0.2,
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
    "optimizer": "Adam",
    "buffer_size": 1e6,
    "polyak": 0.995,
}


def fc_net(env, in_dim, solve_options):
    hidden = solve_options["hidden"]
    depth = solve_options["depth"]
    if solve_options["activation"] == "tanh":
        act_layer = nn.Tanh
    elif solve_options["activation"] == "relu":
        act_layer = nn.ReLU
    modules = []
    modules.append(
        nn.Linear(env.observation_space.shape[0] + in_dim, hidden))
    for _ in range(depth-1):
        modules.append(act_layer())
        modules.append(nn.Linear(hidden, hidden))
    modules.append(act_layer())
    return nn.Sequential(*modules), None


def conv_net(env, in_dim, solve_options):
    hidden = solve_options["hidden"]
    depth = solve_options["depth"]
    if solve_options["activation"] == "tanh":
        act_layer = nn.Tanh
    elif solve_options["activation"] == "relu":
        act_layer = nn.ReLU
    # this assumes image shape == (1, 28, 28)
    conv_modules = []
    conv_modules.append(nn.Conv2d(1, 10, kernel_size=5, stride=2))
    conv_modules.append(nn.Conv2d(10, 20, kernel_size=5, stride=2))
    conv_modules.append(nn.Flatten())

    fc_modules = []
    fc_modules.append(nn.Linear(320+in_dim, hidden))
    for _ in range(depth-1):
        fc_modules.append(act_layer())
        fc_modules.append(nn.Linear(hidden, hidden))
    fc_modules.append(act_layer())
    conv_modules = nn.Sequential(*conv_modules)
    fc_modules = nn.Sequential(*fc_modules)
    return fc_modules, conv_modules


class ValueNet(nn.Module):
    def __init__(self, env, solve_options):
        super().__init__()
        if len(env.observation_space.shape) == 1:
            net = fc_net
        else:
            net = conv_net
        self.fc, self.conv = net(
            env, env.action_space.shape[0], solve_options)
        self.out = nn.Linear(solve_options["hidden"], 1)

    def forward(self, obss, actions=None):
        if self.conv is not None:
            x = self.conv(obss)
        else:
            x = obss
        x = torch.cat([x, actions], -1)
        x = self.fc(x)
        return self.out(x)


class PolicyNet(nn.Module):
    def __init__(self, env, solve_options):
        super().__init__()
        if len(env.observation_space.shape) == 1:
            net = fc_net
        else:
            net = conv_net
        self.fc, self.conv = net(env, 0, solve_options)
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

        options = deepcopy(options)
        self.solve_options.update(OPTIONS)
        super().initialize(options)
        self.device = self.solve_options["device"]
        self.all_obss = torch.tensor(
            self.env.all_observations, dtype=torch.float32, device=self.device)
        self.all_actions = torch.tensor(
            self.env.all_actions, dtype=torch.float32,
            device=self.device).repeat(self.dS, 1).reshape(self.dS, self.dA)  # dS x dA

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
        # set replay buffer
        self.buffer = make_replay_buffer(
            self.env, self.solve_options["buffer_size"])
        # Collect random samples in advance
        self.env.reset()

        policy_probs = self.make_policy_probs()
        policy = self.compute_policy(policy_probs)
        trajectory = collect_samples(
            self.env, policy, self.solve_options["minibatch_size"])
        self.buffer.add(**trajectory)

    def record_performance(self):
        if self.step % self.solve_options["record_performance_interval"] == 0:
            nets = {self.policy_network: "Policy"}
            for net, name in nets.items():
                dist = net.compute_pi_distribution(self.all_obss)
                log_policy = dist.log_prob(self.all_actions) \
                    - (2*(np.log(2) - self.all_actions -
                          F.softplus(-2*self.all_actions)))
                policy_probs = torch.softmax(log_policy, dim=-1).reshape(
                    self.dS, self.dA).detach().cpu().numpy()
                policy = self.compute_policy(policy_probs)
                expected_return = self.env.compute_expected_return(
                    policy)
                self.record_scalar(
                    " Return mean", expected_return, tag=name)
                self.record_array("policy", policy)

            tensor_all_obss = torch.repeat_interleave(self.all_obss, self.dA, 0)  # (SxA) x obss
            tensor_all_actions = self.all_actions.reshape(-1, 1)
            q1 = self.value_network(tensor_all_obss, tensor_all_actions)
            q2 = self.value_network2(tensor_all_obss, tensor_all_actions)
            values = torch.min(q1, q2).reshape(self.dS, self.dA)
            self.record_array("values", values)

    def compute_policy(self, policy_probs):
        # return softmax policy
        return policy_probs

    def make_policy_probs(self):
        dist = self.policy_network.compute_pi_distribution(self.all_obss)
        log_policy = dist.log_prob(self.all_actions) \
            - (2*(np.log(2) - self.all_actions - F.softplus(-2*self.all_actions)))
        policy_probs = torch.softmax(log_policy, dim=-1).reshape(
            self.dS, self.dA).detach().cpu().numpy() 
        return policy_probs
