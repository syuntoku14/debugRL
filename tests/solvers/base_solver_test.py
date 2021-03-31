import pytest
import numpy as np
import gym
import os
from shinrl.envs import Pendulum
from shinrl.solvers import Solver


class MockSolver(Solver):
    def run(self, num_steps=10):
        for i in range(num_steps):
            self.record_scalar("test", i)
            self.record_array("test_array", np.array([1, 2, 3]))
            self.step += 1


def test_make_solver():
    env = Pendulum(state_disc=32, dA=5, horizon=200)
    solver = MockSolver(env)
    solver.run(num_steps=10)

    env = gym.make("Pendulum-v0")
    solver = MockSolver(env)
    solver.run(num_steps=10)


def test_save_solver(tmpdir):
    path = tmpdir.mkdir("tmp")

    env = gym.make("Pendulum-v0")
    solver = MockSolver(env)
    solver.run(num_steps=10)
    data = solver.save(path)

    def assert_list(d):
        for k, v in d.items():
            if isinstance(v, dict):
                assert_list(v)
            assert not isinstance(v, list)

    assert_list(data)
    data = solver.load(path)

    def assert_array(d):
        for k, v in d.items():
            if isinstance(v, dict):
                assert_array(v)
            if not k == "array":
                assert not isinstance(v, np.ndarray)
    assert_array(data)
    assert data["step"] == 10
