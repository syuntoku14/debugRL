import pytest
import numpy as np
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import (
    SamplingViSolver,
    SamplingCviSolver)
from .oracle_vi_solver_test import run_solver


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()

    solve_options = {
        "num_trains": 3,
    }
    yield pend_env, solve_options


def test_value_iteration(setUp):
    pend_env, solve_options = setUp
    solver = SamplingViSolver(pend_env)
    solver.set_options(solve_options)
    run_solver(solver, pend_env)


def test_cvi(setUp):
    pend_env, solve_options = setUp
    solver = SamplingCviSolver(pend_env)
    solver.set_options(solve_options)
    run_solver(solver, pend_env)
