import pytest
import gym
import numpy as np
import pybullet_envs
from shinrl.envs import Pendulum
from shinrl.solvers.ppo.continuous import PpoSolver
from ..misc import run_solver_tb, run_solver_gym
import torch


@pytest.fixture
def setUp():
    tb_env = Pendulum(state_disc=5, dA=4, horizon=5, action_mode="continuous")
    tb_env.reset()

    gym_env = gym.make("AntBulletEnv-v0")
    gym_env.reset()
    yield tb_env, gym_env


def test_tb(setUp):
    tb_env, _ = setUp
    solver = PpoSolver(
        tb_env, solve_options={"num_samples": 10, "train_net_iters": 1})
    run_solver_tb(solver, tb_env)


def test_gym(setUp):
    _, gym_env = setUp
    solver = PpoSolver(
        gym_env, solve_options={"num_samples": 10, "train_net_iters": 1})
    run_solver_gym(solver, gym_env)
