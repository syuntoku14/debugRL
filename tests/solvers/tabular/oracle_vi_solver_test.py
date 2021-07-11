import numpy as np
import pytest

from shinrl.envs import Pendulum
from shinrl.solvers.tabular.discrete.vi import (
    OracleCviSolver,
    OracleMviSolver,
    OracleViSolver,
)

from ..misc import run_solver_tb


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    yield pend_env


def test_visitation_matrix(setUp):
    pend_env = setUp
    solver = OracleViSolver(pend_env, solve_options={"use_oracle_visitation": False})
    run_solver_tb(solver, pend_env)


def test_approx_matrix(setUp):
    pend_env = setUp
    solver = OracleViSolver(pend_env, solve_options={"no_approx": False})
    run_solver_tb(solver, pend_env)


def test_vi(setUp):
    pend_env = setUp
    solver = OracleViSolver(pend_env)
    run_solver_tb(solver, pend_env)


def test_cvi(setUp):
    pend_env = setUp
    solver = OracleCviSolver(pend_env)
    run_solver_tb(solver, pend_env)


def test_mvi(setUp):
    pend_env = setUp
    solver = OracleMviSolver(pend_env)
    run_solver_tb(solver, pend_env)
