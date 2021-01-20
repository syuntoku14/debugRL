import pytest
import numpy as np
import gym
from debug_rl.envs.gridcraft import GridEnv, spec_from_string
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import (
    OracleViSolver,
    OracleCviSolver,
    SamplingViSolver,
    SamplingFittedViSolver)


@pytest.fixture
def setUp():
    env = Pendulum(state_disc=32, dA=5, horizon=200)
    env.reset()
    yield env


def compute_visitation(solver, policy):
    return solver.env.compute_visitation(policy)


def compute_expected_return(solver, policy):
    return solver.env.compute_expected_return(policy)


def test_compute_visitation(setUp, benchmark):
    env = setUp
    solver = OracleViSolver(env)
    solver.solve(num_steps=10)
    policy = solver.compute_policy(solver.values)
    benchmark.pedantic(compute_visitation,
                       kwargs={"solver": solver, "policy": policy})


def test_compute_expected_return(setUp, benchmark):
    env = setUp
    solver = OracleViSolver(env)
    solver.solve(num_steps=10)
    policy = solver.compute_policy(solver.values)
    benchmark.pedantic(compute_expected_return,
                       kwargs={"solver": solver, "policy": policy})


def test_gym_env_raise():
    with pytest.raises(AssertionError):
        env = gym.make("Pendulum-v0")
        solver = OracleViSolver(env)
        solver.solve(num_steps=10)


def test_seed(setUp):
    env = setUp
    solver = SamplingViSolver(env)
    fsolver = SamplingFittedViSolver(env)
    solver.initialize({"seed": 0, "record_performance_interval": 1})
    fsolver.initialize({"seed": 0, "record_performance_interval": 1})
    solver.solve(num_steps=10)
    fsolver.solve(num_steps=10)
    val1 = solver.values
    fval1 = fsolver.values

    solver.initialize({"seed": 0, "record_performance_interval": 1})
    fsolver.initialize({"seed": 0, "record_performance_interval": 1})
    solver.solve(num_steps=10)
    fsolver.solve(num_steps=10)
    val2 = solver.values
    fval2 = fsolver.values

    assert np.all(val1 == val2)
    assert np.all(fval1 == fval2)

    solver.initialize({"seed": 1, "record_performance_interval": 1})
    fsolver.initialize({"seed": 1, "record_performance_interval": 1})
    solver.solve(num_steps=10)
    fsolver.solve(num_steps=10)
    val3 = solver.values
    fval3 = fsolver.values

    assert not np.all(val1 == val3)
    assert not np.all(fval1 == fval3)
