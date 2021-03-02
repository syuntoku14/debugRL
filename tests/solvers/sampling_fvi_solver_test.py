import gym
import pytest
import numpy as np
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import (
    SamplingFittedViSolver,
    SamplingFittedCviSolver
)
from .oracle_vi_solver_test import run_solver
import torch


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()

    gym_env = gym.make("CartPole-v0")
    gym_env.reset()
    yield pend_env, gym_env


def test_value_iteration(setUp):
    pend_env, gym_env = setUp
    # solver = SamplingFittedViSolver(pend_env)
    # solver.initialize()
    # run_solver(solver, pend_env)

    solver = SamplingFittedViSolver(gym_env)
    solver.initialize()
    run_solver(solver, pend_env)


def test_cvi(setUp):
    pend_env, gym_env = setUp
    solver = SamplingFittedCviSolver(pend_env)
    solver.initialize()
    run_solver(solver, pend_env)


def test_image_obs(setUp):
    pend_env = Pendulum(state_disc=5, dA=3,
                        horizon=5, obs_mode="image")
    pend_env.reset()
    solver = SamplingFittedViSolver(pend_env)
    solver.initialize()
    run_solver(solver, pend_env)
