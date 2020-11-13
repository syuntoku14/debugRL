import pytest
import numpy as np
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import SacContinuousSolver
from .oracle_vi_solver_test import run_solver
import torch


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, num_actions=4, horizon=5, action_mode="continuous")
    pend_env.reset()

    solve_options = {
        "num_trains": 5,
    }
    yield pend_env, solve_options


def test_sac_continuous(setUp):
    pend_env, solve_options = setUp
    solver = SacContinuousSolver(pend_env)
    solver.set_options(solve_options)
    run_solver(solver, pend_env)
