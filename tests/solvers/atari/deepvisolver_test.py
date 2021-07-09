import os

os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA

import gym
import numpy as np
import pytest
import torch

from shinrl.envs import Pendulum
from shinrl.solvers.atari import DQNSolver, MDQNSolver, make_atari

from ..misc import run_solver_gym


@pytest.fixture
def setUp():
    env = make_atari("PongNoFrameskip-v4")
    env.reset()
    yield env 


def test_dqn(setUp):
    env = setUp
    solver = DQNSolver(env)
    solver.initialize()
    run_solver_gym(solver, env)


def test_mdqn(setUp):
    env = setUp
    solver = MDQNSolver(env)
    solver.initialize()
    run_solver_gym(solver, env)
