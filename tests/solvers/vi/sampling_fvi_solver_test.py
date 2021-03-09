import gym
import pytest
import numpy as np
from shinrl.envs import Pendulum
from shinrl.solvers.vi.discrete import (
    SamplingFittedViSolver,
    SamplingFittedCviSolver
)
from ..misc import run_solver_tb, run_solver_gym
import torch


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()

    gym_env = gym.make("CartPole-v0")
    gym_env.reset()
    yield pend_env, gym_env


def test_vi_tb(setUp):
    pend_env, gym_env = setUp
    solver = SamplingFittedViSolver(pend_env)
    solver.initialize()
    run_solver_tb(solver, pend_env)


def test_vi_gym(setUp):
    pend_env, gym_env = setUp
    solver = SamplingFittedViSolver(gym_env)
    solver.initialize()
    run_solver_gym(solver, gym_env)


def test_cvi_tb(setUp):
    pend_env, gym_env = setUp
    solver = SamplingFittedCviSolver(pend_env)
    solver.initialize()
    run_solver_tb(solver, pend_env)


def test_cvi_gym(setUp):
    pend_env, gym_env = setUp
    solver = SamplingFittedCviSolver(gym_env)
    solver.initialize()
    run_solver_gym(solver, gym_env)


def test_image_obs(setUp):
    pend_env = Pendulum(state_disc=5, dA=3,
                        horizon=5, obs_mode="image")
    pend_env.reset()
    solver = SamplingFittedViSolver(pend_env)
    solver.initialize()
    run_solver_tb(solver, pend_env)
