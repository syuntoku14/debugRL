import gym
import numpy as np
import pybullet_envs
import pytest
import torch

from shinrl.envs import Pendulum
from shinrl.solvers.onpg.continuous import SamplingPgSolver

from ..misc import run_solver_gym, run_solver_tb


@pytest.fixture
def setUp():
    tb_env = Pendulum(state_disc=5, dA=4, horizon=5, action_mode="continuous")
    tb_env.reset()

    gym_env = gym.make("AntBulletEnv-v0")
    gym_env.reset()
    yield tb_env, gym_env


def test_tb(setUp):
    tb_env, _ = setUp
    solver = SamplingPgSolver(
        tb_env, solve_options={"num_samples": 10, "num_minibatches": 1}
    )
    run_solver_tb(solver, tb_env)


def test_gym(setUp):
    _, gym_env = setUp
    solver = SamplingPgSolver(
        gym_env, solve_options={"num_samples": 10, "num_minibatches": 1}
    )
    run_solver_gym(solver, gym_env)
