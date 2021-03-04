import pytest
import numpy as np
from rlowan.envs.pendulum import Pendulum
from rlowan.solvers import *
from .oracle_vi_solver_test import run_solver
import torch


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    yield pend_env


def test_ppo(setUp):
    pend_env = setUp
    solver = PpoSolver(pend_env)
    run_solver(solver, pend_env)


def test_image_obs(setUp):
    pend_env = Pendulum(state_disc=5, dA=3,
                        horizon=5, obs_mode="image")
    pend_env.reset()
    solver = PpoSolver(pend_env)
    run_solver(solver, pend_env)
