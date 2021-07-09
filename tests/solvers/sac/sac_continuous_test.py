import gym
import numpy as np
import pytest
import torch

from shinrl.envs import Pendulum
from shinrl.solvers.sac.continuous import SacSolver

from ..misc import run_solver_gym, run_solver_tb


@pytest.fixture
def setUp():
    tb_env = Pendulum(state_disc=5, dA=4, horizon=5, action_mode="continuous")
    tb_env.reset()

    gym_env = gym.make("Pendulum-v0")
    gym_env.reset()
    yield tb_env, gym_env


def test_tb(setUp, tmpdir):
    tb_env, _ = setUp
    solver = SacSolver(tb_env)
    run_solver_tb(solver, tb_env)


def test_gym(setUp, tmpdir):
    _, gym_env = setUp
    solver = SacSolver(gym_env)
    run_solver_gym(solver, gym_env)
