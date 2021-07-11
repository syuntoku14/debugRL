import os

os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA

import gym
import numpy as np

try:
    import pybullet_envs

    skip = False
except ImportError:
    skip = True

import pytest
import torch

from shinrl.envs import Pendulum
from shinrl.solvers.mujoco import SacSolver

from ..misc import run_solver_gym

import_skip = pytest.mark.skipif(skip, reason="pybullet is not installed")


@pytest.fixture
def setUp():
    env = gym.make("AntBulletEnv-v0")
    env.reset()
    yield env


@import_skip
def test_sac(setUp):
    env = setUp
    solver = SacSolver(env)
    solver.initialize()
    run_solver_gym(solver, env)
