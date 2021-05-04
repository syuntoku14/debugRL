import numpy as np
import pytest

from shinrl.envs import MountainCar


def test_mountaincar_env():
    env = MountainCar(state_disc=5)
    env.reset()
    for i in range(env.horizon - 1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert done

    trans_matrix = env.transition_matrix
    reward_matrix = env.reward_matrix

    assert trans_matrix.shape == (env.dS * env.dA, env.dS)
    assert reward_matrix.shape == (env.dS, env.dA)


def test_mountaincar_continuous_env():
    env = MountainCar(state_disc=5, action_mode="continuous")
    env.reset()
    for i in range(env.horizon - 1):
        a = env.action_space.sample()
        a = env.discretize_action(a)
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert done

    trans_matrix = env.transition_matrix
    reward_matrix = env.reward_matrix

    assert trans_matrix.shape == (env.dS * env.dA, env.dS)
    assert reward_matrix.shape == (env.dS, env.dA)


def test_image_obs():
    env = MountainCar(state_disc=5, obs_mode="image")
    env.reset()
    for i in range(env.horizon - 1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert obs.shape == (1, 28, 28)
