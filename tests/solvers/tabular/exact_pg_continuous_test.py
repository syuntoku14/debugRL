import numpy as np
import pytest

from shinrl.envs import Pendulum
from shinrl.solvers.tabular.continuous.pg import ExactPgSolver

from ..misc import run_solver_tb


@pytest.fixture
def setUp():
    tb_env = Pendulum(state_disc=5, dA=4, horizon=5, action_mode="continuous")
    tb_env.reset()
    yield tb_env


def test_pg(setUp):
    pend_env = setUp
    solver = ExactPgSolver(pend_env)
    solver.initialize(options={"coef": "Q"})
    run_solver_tb(solver, pend_env)

    solver.initialize(options={"coef": "A"})
    run_solver_tb(solver, pend_env)
