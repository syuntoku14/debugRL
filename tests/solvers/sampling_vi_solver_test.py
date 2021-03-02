import pytest
import numpy as np
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import (
    SamplingViSolver,
    SamplingCviSolver)
from .run_solver import run_solver_tb


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    yield pend_env


def test_value_iteration(setUp):
    pend_env = setUp
    solver = SamplingViSolver(pend_env)
    solver.initialize()
    run_solver_tb(solver, pend_env)


def test_cvi(setUp):
    pend_env = setUp
    solver = SamplingCviSolver(pend_env)
    solver.initialize()
    run_solver_tb(solver, pend_env)
