import pytest
import numpy as np
import gym
from shinrl.envs.pendulum import Pendulum
from shinrl.solvers.sac.discrete import SacSolver
from ..misc import run_solver_tb, run_solver_gym
import torch


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    yield pend_env


def test_tb(setUp):
    pend_env = setUp
    solver = SacSolver(pend_env)
    run_solver_tb(solver, pend_env)


def test_image_obs(setUp):
    pend_env = Pendulum(state_disc=5, dA=3,
                        horizon=5, obs_mode="image")
    pend_env.reset()
    solver = SacSolver(pend_env)
    run_solver_tb(solver, pend_env)


def test_gym(setUp):
    gym_env = gym.make("CartPole-v0")
    gym_env.reset()
    solver = SacSolver(gym_env)
    run_solver_gym(solver, gym_env)
