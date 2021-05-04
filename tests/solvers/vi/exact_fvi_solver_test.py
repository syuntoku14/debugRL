import pytest
import numpy as np
from shinrl.envs import Pendulum
from shinrl.solvers.vi.discrete import (
    ExactFittedViSolver,
    ExactFittedCviSolver,
    ExactFittedMviSolver)
from ..misc import run_solver_tb


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    yield pend_env


def test_vi(setUp):
    pend_env = setUp
    solver = ExactFittedViSolver(pend_env)
    run_solver_tb(solver, pend_env)


def test_cvi(setUp):
    pend_env = setUp
    solver = ExactFittedCviSolver(pend_env)
    run_solver_tb(solver, pend_env)


def test_mvi(setUp):
    pend_env = setUp
    solver = ExactFittedMviSolver(pend_env)
    run_solver_tb(solver, pend_env)
