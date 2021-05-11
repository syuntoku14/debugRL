import pytest
import numpy as np
from shinrl.envs import Pendulum
from shinrl.solvers.pi.discrete import OracleCpiSolver
from ..misc import run_solver_tb


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    yield pend_env


def test_oracle_cpi(setUp):
    pend_env = setUp
    solver = OracleCpiSolver(pend_env)
    run_solver_tb(solver, pend_env)
