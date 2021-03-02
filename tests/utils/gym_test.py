import pytest
import numpy as np
import torch
import gym
from torch import nn
from debug_rl.utils import *
from debug_rl.utils.gym import *


def fc_net(env):
    modules = []
    obs_shape = env.observation_space.shape[0]
    n_acts = env.action_space.n
    modules.append(nn.Linear(obs_shape, 256))
    modules += [nn.ReLU(), nn.Linear(256, n_acts)]
    return nn.Sequential(*modules)


@pytest.fixture
def setUp():
    env = gym.make("CartPole-v0")
    net = fc_net(env)
    yield env, net


def test_collect_samples(setUp):
    env, net = setUp

    def get_action(env):
        if not hasattr(env, 'obs'):
            env.obs = env.reset()
        obs = torch.as_tensor(env.obs, dtype=torch.float32).unsqueeze(0)
        probs = net(obs).detach().cpu().numpy()  # 1 x dA
        probs = eps_greedy_policy(probs, eps_greedy=0.1).reshape(-1)
        log_probs = np.log(probs)
        action = np.random.choice(np.arange(0, env.action_space.n), p=probs)
        log_prob = log_probs[action]
        return action, log_prob

    buf = make_replay_buffer(env, 100)
    traj = collect_samples(env, get_action, 10)
    buf.add(**traj)
    assert len(traj["obs"]) == 10
    assert (traj["act"]).dtype == np.long
