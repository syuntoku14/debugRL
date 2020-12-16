import pytest
import numpy as np
from debug_rl.envs.cartpole import CartPole


def test_cartpole_env():
    env = CartPole(
        x_disc=5, x_dot_disc=5, th_disc=8, th_dot_disc=16, horizon=10)
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
        env.dS*env.dA, 1)


def test_cartpole_continuous_env():
    env = CartPole(
        x_disc=5, x_dot_disc=5, th_disc=8, th_dot_disc=16,
        horizon=10, action_mode="continuous")
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
        env.dS*env.dA, 1)
