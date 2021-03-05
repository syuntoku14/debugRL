import pytest
import numpy as np
from shinrl.envs.gridcraft import GridEnv, spec_from_string
from shinrl.envs.gridcraft.wrappers import OneHotObsWrapper, RandomObsWrapper


def test_grid_env():
    maze = spec_from_string("SOOO\\" +
                            "OLLL\\" +
                            "OOOO\\" +
                            "OLRO\\")
    # maze = MAZE_LAVA
    env = GridEnv(maze, trans_eps=0.0)
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


def test_coord_wrapper():
    maze = spec_from_string("SOOO\\" +
                            "OLLL\\" +
                            "OOOO\\" +
                            "OLRO\\")
    # maze = MAZE_LAVA
    env = GridEnv(maze, trans_eps=0.0)
    env = OneHotObsWrapper(env)
    env.reset()
    for i in range(10):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
    assert obs.shape == (8, )


def test_random_wrapper():
    maze = spec_from_string("SOOO\\" +
                            "OLLL\\" +
                            "OOOO\\" +
                            "OLRO\\")
    # maze = MAZE_LAVA
    env = GridEnv(maze, trans_eps=0.0)
    obs_dim = 5
    env = RandomObsWrapper(env, obs_dim=obs_dim)
    env.reset()
    for i in range(10):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
    assert obs.shape == (obs_dim, )
