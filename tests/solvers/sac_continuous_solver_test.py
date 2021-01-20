import pytest
import numpy as np
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import SacContinuousSolver
from .oracle_vi_solver_test import run_solver
import torch


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=4, horizon=5, action_mode="continuous")
    pend_env.reset()
    yield pend_env


def test_sac_continuous(setUp):
    pend_env = setUp
    solver = SacContinuousSolver(pend_env)
    solver.initialize()
    run_solver(solver, pend_env)
