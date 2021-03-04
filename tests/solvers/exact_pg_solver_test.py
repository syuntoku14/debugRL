import pytest
import numpy as np
from rlowan.envs.pendulum import Pendulum
from rlowan.solvers import ExactPgSolver
from .oracle_vi_solver_test import run_solver


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    yield pend_env


def test_pg(setUp):
    pend_env = setUp
    solver = ExactPgSolver(pend_env)
    solver.initialize(options={"coef": "Q"})
    run_solver(solver, pend_env)

    solver.initialize(options={"coef": "A"})
    run_solver(solver, pend_env)
