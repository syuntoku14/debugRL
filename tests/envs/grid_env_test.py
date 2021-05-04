import numpy as np
import pytest

from shinrl.envs.gridcraft import GridEnv, grid_spec_from_string


def test_grid_env():
    maze = grid_spec_from_string("SOOO\\" + "OLLL\\" + "OOOO\\" + "OLRO\\")
    # maze = MAZE_LAVA
    env = GridEnv(maze, trans_eps=0.0)
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


def test_onehot():
    maze = grid_spec_from_string("SOOO\\" + "OLLL\\" + "OOOO\\" + "OLRO\\")
    # maze = MAZE_LAVA
    env = GridEnv(maze, trans_eps=0.0, obs_mode="onehot")
    env.reset()
    for i in range(10):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
    assert obs.shape == (8,)


def test_random():
    maze = grid_spec_from_string("SOOO\\" + "OLLL\\" + "OOOO\\" + "OLRO\\")
    # maze = MAZE_LAVA
    obs_dim = 5
    env = GridEnv(maze, trans_eps=0.0, obs_dim=obs_dim, obs_mode="random")
    env.reset()
    for i in range(10):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
    assert obs.shape == (obs_dim,)
