import pytest
import numpy as np
import gym
from shinrl.envs.pendulum import Pendulum
from shinrl.solvers.ipg.discrete import IpgSolver
from ..misc import run_solver_tb, run_solver_gym


def test_tb():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    solver = IpgSolver(pend_env)
    run_solver_tb(solver, pend_env)


def test_gym():
    gym_env = gym.make("CartPole-v0")
    gym_env.reset()
    solver = IpgSolver(gym_env)
    run_solver_gym(solver, gym_env)
