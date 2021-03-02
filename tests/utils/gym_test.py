import pytest
import numpy as np
import torch
import gym
from scipy import special
from torch import nn
from debug_rl.utils import *


def fc_net(env):
    modules = []
    obs_shape = env.observation_space.shape[0]
    n_acts = env.action_space.n
    modules.append(nn.Linear(obs_shape, 256))
    modules += [nn.ReLU(), nn.Linear(256, n_acts)]
    return nn.Sequential(*modules)


def mock_policy(preference):
    policy = special.softmax(preference, axis=-1).astype(np.float64)
    policy /= policy.sum(axis=-1, keepdims=True)
    return policy.astype(np.float32)


@pytest.fixture
def setUp():
    env = gym.make("CartPole-v0")
    env.obs = env.reset()
    net = fc_net(env)
    yield env, net


def get_action(env, net):
    if not hasattr(env, 'obs'):
        env.obs = env.reset()
    obs = torch.as_tensor(env.obs, dtype=torch.float32).unsqueeze(0)
    probs = net(obs).detach().cpu().numpy()  # 1 x dA
    probs = mock_policy(probs).reshape(-1)
    log_probs = np.log(probs)
    action = np.random.choice(np.arange(0, env.action_space.n), p=probs)
    log_prob = log_probs[action]
    return action, log_prob


def test_collect_samples(setUp):
    env, net = setUp
    buf = make_replay_buffer(env, 100)
    traj = collect_samples(env, get_action, 10, net=net)
    buf.add(**traj)
    assert len(traj["obs"]) == 10
    assert (traj["act"]).dtype == np.long


def test_collect_samples_episodic(setUp):
    env, net = setUp
    buf = make_replay_buffer(env, 100)
    traj = collect_samples(env, get_action, 10, num_episodes=5, net=net)
    buf.add(**traj)
    assert np.sum(traj["done"]) == 5
    assert (traj["act"]).dtype == np.long
