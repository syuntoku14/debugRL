import numpy as np
import pytest
import torch

from shinrl.envs import Pendulum
from shinrl.solvers.tabular.discrete.vi import (
    SamplingFittedCviSolver,
    SamplingFittedMviSolver,
    SamplingFittedViSolver,
)

from ..misc import run_solver_tb


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    yield pend_env


def test_vi_tb(setUp):
    pend_env = setUp
    solver = SamplingFittedViSolver(pend_env)
    solver.initialize()
    run_solver_tb(solver, pend_env)


def test_cvi_tb(setUp):
    pend_env = setUp
    solver = SamplingFittedCviSolver(pend_env)
    solver.initialize()
    run_solver_tb(solver, pend_env)


def test_mvi_tb(setUp):
    pend_env = setUp
    solver = SamplingFittedMviSolver(pend_env)
    solver.initialize()
    run_solver_tb(solver, pend_env)


def test_image_obs(setUp):
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5, obs_mode="image")
    pend_env.reset()
    solver = SamplingFittedViSolver(pend_env)
    solver.initialize()
    run_solver_tb(solver, pend_env)
