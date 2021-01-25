import pytest
import numpy as np
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import SamplingPgSolver
from .oracle_vi_solver_test import run_solver


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    yield pend_env


def test_sampling_pg(setUp):
    pend_env = setUp
    solver = SamplingPgSolver(pend_env)
    solver.initialize()
    run_solver(solver, pend_env)