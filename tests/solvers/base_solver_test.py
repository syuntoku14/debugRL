import pytest
import numpy as np
import gym
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import (
    OracleViSolver,
    SamplingViSolver,
    SamplingFittedViSolver
)


def test_gym_env_raise():
    with pytest.raises(AssertionError):
        env = gym.make("Pendulum-v0")
        solver = OracleViSolver(env)
        solver.run(num_steps=10)


def test_seed():
    env = Pendulum(state_disc=32, dA=5, horizon=200)
    env.reset()
    solver = SamplingViSolver(env)
    fsolver = SamplingFittedViSolver(env)
    solver.initialize({"seed": 0, "record_performance_interval": 1})
    fsolver.initialize({"seed": 0, "record_performance_interval": 1})
    solver.run(num_steps=10)
    fsolver.run(num_steps=10)
    val1 = solver.values
    fval1 = fsolver.values

    solver.initialize({"seed": 0, "record_performance_interval": 1})
    fsolver.initialize({"seed": 0, "record_performance_interval": 1})
    solver.run(num_steps=10)
    fsolver.run(num_steps=10)
    val2 = solver.values
    fval2 = fsolver.values

    assert np.all(val1 == val2)
    assert np.all(fval1 == fval2)

    solver.initialize({"seed": 1, "record_performance_interval": 1})
    fsolver.initialize({"seed": 1, "record_performance_interval": 1})
    solver.run(num_steps=10)
    fsolver.run(num_steps=10)
    val3 = solver.values
    fval3 = fsolver.values

    assert not np.all(val1 == val3)
    assert not np.all(fval1 == fval3)
