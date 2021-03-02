import pytest
import numpy as np
import torch
from torch import nn
from debug_rl.envs.pendulum import Pendulum
from debug_rl.utils import *


@pytest.fixture
def setUp():
    env = Pendulum(state_disc=5, dA=3)
    env.obs = env.reset()
    policy = np.random.rand(env.dS, env.dA)
    policy = policy / policy.sum(1, keepdims=True)

    cont_env = Pendulum(state_disc=5, dA=3, action_mode="continuous")
    cont_env.obs = cont_env.reset()
    cont_policy = np.random.rand(cont_env.dS, cont_env.dA)
    cont_policy = cont_policy / cont_policy.sum(1, keepdims=True)
    yield env, policy, cont_env, cont_policy


def test_collect_samples(setUp):
    env, policy, cont_env, cont_policy = setUp

    # discrete
    buf = make_replay_buffer(env, 100)
    traj = collect_samples(env, get_tb_action, 10, policy=policy)
    buf.add(**traj)
    assert len(traj["obs"]) == 10
    assert (traj["act"]).dtype == np.long

    # continuous
    buf = make_replay_buffer(cont_env, 100)
    traj = collect_samples(cont_env, get_tb_action, 10, policy=cont_policy)
    buf.add(**traj)
    assert (traj["obs"]).dtype == np.float32
