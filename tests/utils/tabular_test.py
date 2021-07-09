import numpy as np
import pytest
import torch
from torch import nn

from shinrl import utils
from shinrl.envs import Pendulum


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
    buf = utils.make_replay_buffer(env, 100)
    traj = utils.collect_samples(env, utils.get_tb_action, 10, policy=policy, buffer=buf)
    assert len(traj["obs"]) == 10
    assert (traj["act"]).dtype == np.long
    assert buf.get_stored_size() == 10

    # continuous
    buf = utils.make_replay_buffer(cont_env, 100)
    traj = utils.collect_samples(cont_env, utils.get_tb_action, 10, policy=cont_policy, buffer=buf)
    assert (traj["obs"]).dtype == np.float32
    assert buf.get_stored_size() == 10


def test_collect_samples_episodic(setUp):
    env, policy, cont_env, cont_policy = setUp

    # discrete
    traj = utils.collect_samples(
        env, utils.get_tb_action, 10, num_episodes=5, policy=policy
    )
    assert np.sum(traj["done"]) == 5
    assert (traj["act"]).dtype == np.long

    # continuous
    traj = utils.collect_samples(
        cont_env, utils.get_tb_action, 10, num_episodes=5, policy=cont_policy
    )
    assert np.sum(traj["done"]) == 5
    assert (traj["obs"]).dtype == np.float32
