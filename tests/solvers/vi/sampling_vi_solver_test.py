import numpy as np
import pytest
from shinrl.envs import Pendulum
from shinrl.solvers.vi.discrete import (
    SamplingCviSolver,
    SamplingMviSolver,
    SamplingViSolver,
)

from ..misc import run_solver_tb


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    yield pend_env


def test_vi(setUp):
    pend_env = setUp
    solver = SamplingViSolver(pend_env)
    solver.initialize()
    run_solver_tb(solver, pend_env)


def test_cvi(setUp):
    pend_env = setUp
    solver = SamplingCviSolver(pend_env)
    solver.initialize()
    run_solver_tb(solver, pend_env)


def test_mvi(setUp):
    pend_env = setUp
    solver = SamplingMviSolver(pend_env)
    solver.initialize()
    run_solver_tb(solver, pend_env)
