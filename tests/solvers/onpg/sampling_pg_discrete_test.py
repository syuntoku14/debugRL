import gym
import numpy as np
import pytest
from shinrl.envs import Pendulum
from shinrl.solvers.onpg.discrete import SamplingPgSolver

from ..misc import run_solver_gym, run_solver_tb


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    yield pend_env


def test_tb(setUp):
    pend_env = setUp
    solver = SamplingPgSolver(pend_env)
    run_solver_tb(solver, pend_env)


def test_gym(setUp):
    gym_env = gym.make("CartPole-v0")
    gym_env.reset()
    solver = SamplingPgSolver(gym_env)
    run_solver_gym(solver, gym_env)
