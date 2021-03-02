import pytest
import numpy as np
import gym
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import Solver


class MockSolver(Solver):
    def run(self, num_steps=10):
        for _ in range(num_steps):
            pass


def test_make_solver():
    env = Pendulum(state_disc=32, dA=5, horizon=200)
    solver = MockSolver(env)
    solver.run(num_steps=10)

    env = gym.make("Pendulum-v0")
    solver = MockSolver(env)
    solver.run(num_steps=10)
