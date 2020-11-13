import pytest
import numpy as np
from debug_rl.envs.pendulum import Pendulum


def test_pendulum_env():
    env = Pendulum(state_disc=5, num_actions=3)
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


def test_pendulum_continuous_env():
    env = Pendulum(state_disc=5, num_actions=3, action_mode="continuous")
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
    env = Pendulum(state_disc=5, num_actions=3, obs_mode="image")
    env.reset()
    for i in range(env.horizon-1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert obs.shape == (1, 28, 28)
