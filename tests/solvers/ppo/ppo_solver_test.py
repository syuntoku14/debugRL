import pytest
import numpy as np
import gym
from shinrl.envs import Pendulum
from shinrl.solvers.ppo.discrete import PpoSolver
from ..misc import run_solver_tb, run_solver_gym


def test_tb():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    solver = PpoSolver(pend_env)
    run_solver_tb(solver, pend_env)


def test_gym():
    gym_env = gym.make("CartPole-v0")
    gym_env.reset()
    solver = PpoSolver(gym_env)
    run_solver_gym(solver, gym_env)
