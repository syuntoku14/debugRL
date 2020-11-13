import pytest
import numpy as np
from debug_rl.envs.mountaincar import MountainCar


def test_mountaincar_env():
    env = MountainCar(posdisc=5, veldisc=5)
    env.reset()
    for i in range(env.horizon-1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert done

    trans_matrix = env.transition_matrix()
    rew_matrix = env.reward_matrix()

    assert trans_matrix.shape == (
        env.num_states*env.num_actions, env.num_states)
    assert rew_matrix.shape == (
        env.num_states*env.num_actions, 1)


def test_mountaincar_continuous_env():
    env = MountainCar(posdisc=5, veldisc=5, action_mode="continuous")
    env.reset()
    for i in range(env.horizon-1):
        a = env.action_space.sample()
        a = env.discretize_action(a)
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert done

    trans_matrix = env.transition_matrix()
    rew_matrix = env.reward_matrix()

    assert trans_matrix.shape == (
        env.num_states*env.num_actions, env.num_states)
    assert rew_matrix.shape == (
        env.num_states*env.num_actions, 1)


def test_image_obs():
    env = MountainCar(posdisc=5, veldisc=5, obs_mode="image")
    env.reset()
    for i in range(env.horizon-1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert obs.shape == (1, 28, 28)
