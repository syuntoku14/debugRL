import pytest
import gym
import numpy as np
import pybullet_envs
from shinrl.envs.pendulum import Pendulum
from shinrl.solvers.sac.continuous import SacSolver
from ..misc import run_solver_tb, run_solver_gym
import torch


@pytest.fixture
def setUp():
    tb_env = Pendulum(state_disc=5, dA=4, horizon=5, action_mode="continuous")
    tb_env.reset()

    gym_env = gym.make("AntBulletEnv-v0")
    gym_env.reset()
    yield tb_env, gym_env


def test_tb(setUp):
    tb_env, _ = setUp
    solver = SacSolver(tb_env)
    solver.initialize()
    run_solver_tb(solver, tb_env)


def test_gym(setUp):
    _, gym_env = setUp
    solver = SacSolver(gym_env)
    solver.initialize()
    run_solver_gym(solver, gym_env)
