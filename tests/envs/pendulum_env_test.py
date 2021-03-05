import pytest
import numpy as np
from shinrl.envs.pendulum import Pendulum


def test_pendulum_env():
    env = Pendulum(state_disc=5, dA=3)
    env.reset()
    for i in range(env.horizon-1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert done

    trans_matrix = env.transition_matrix
    reward_matrix = env.reward_matrix

    assert trans_matrix.shape == (
        env.dS*env.dA, env.dS)
    assert reward_matrix.shape == (
        env.dS, env.dA)


def test_pendulum_continuous_env():
    env = Pendulum(state_disc=5, dA=3, action_mode="continuous")
    env.reset()
    for i in range(env.horizon-1):
        a = env.action_space.sample()
        a = env.discretize_action(a)
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert done

    trans_matrix = env.transition_matrix
    reward_matrix = env.reward_matrix

    assert trans_matrix.shape == (
        env.dS*env.dA, env.dS)
    assert reward_matrix.shape == (
        env.dS, env.dA)


def test_image_obs():
    env = Pendulum(state_disc=5, dA=3, obs_mode="image")
    env.reset()
    for i in range(env.horizon-1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert obs.shape == (1, 28, 28)
