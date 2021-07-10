import os

os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA

import gym
import numpy as np
import pytest
import torch

from shinrl.envs import Pendulum
from shinrl.solvers.minatar import DqnSolver, MDqnSolver, SacSolver, make_minatar

from ..misc import run_solver_gym


@pytest.fixture
def setUp():
    env = make_minatar("breakout")
    env.reset()
    yield env


def test_dqn(setUp):
    env = setUp
    solver = DqnSolver(env)
    solver.initialize()
    run_solver_gym(solver, env)


def test_mdqn(setUp):
    env = setUp
    solver = MDqnSolver(env)
    solver.initialize()
    run_solver_gym(solver, env)


def test_sac(setUp):
    env = setUp
    solver = SacSolver(env)
    solver.initialize()
    run_solver_gym(solver, env)
