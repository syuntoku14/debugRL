import pytest
import numpy as np
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import (
    ExactFittedViSolver,
    ExactFittedCviSolver)
from .oracle_vi_solver_test import run_solver

@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, num_actions=3, horizon=5)
    pend_env.reset()

    solve_option = {
        "num_trains": 10,
    }
    yield pend_env, solve_option


def test_value_iteration(setUp):
    pend_env, solve_option = setUp
    solver = ExactFittedViSolver(pend_env)
    solver.set_options(solve_option)
    run_solver(solver, pend_env)


def test_cvi(setUp):
    pend_env, solve_option = setUp
    solver = ExactFittedCviSolver(pend_env)
    solver.set_options(solve_option)
    run_solver(solver, pend_env)
