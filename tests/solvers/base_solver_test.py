import pytest
import numpy as np
from debug_rl.envs.gridcraft import GridEnv, spec_from_string
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import (
    OracleViSolver,
    OracleCviSolver,
    SamplingViSolver,
    SamplingFittedViSolver)


@pytest.fixture
def setUp():
    env = Pendulum(state_disc=32, num_actions=5, horizon=200)
    env.reset()
    yield env


def compute_visitation(solver, policy):
    return solver.compute_visitation(policy)


def compute_expected_return(solver, policy):
    return solver.compute_expected_return(policy)


def test_compute_visitation(setUp, benchmark):
    env = setUp
    solver = OracleViSolver(env)
    solver.solve()
    policy = solver.compute_policy(solver.values)
    benchmark.pedantic(compute_visitation,
                       kwargs={"solver": solver, "policy": policy})


def test_compute_expected_return(setUp, benchmark):
    env = setUp
    solver = OracleViSolver(env)
    solver.solve()
    policy = solver.compute_policy(solver.values)
    benchmark.pedantic(compute_expected_return,
                       kwargs={"solver": solver, "policy": policy})


def test_seed(setUp):
    env = setUp
    solver = SamplingViSolver(env)
    fsolver = SamplingFittedViSolver(env)
    solver.set_options({"seed": 0, "num_trains": 10})
    fsolver.set_options({"seed": 0, "num_trains": 10})
    solver.solve()
    fsolver.solve()
    val1 = solver.values
    fval1 = fsolver.values

    solver.set_options({"seed": 0, "num_trains": 10})
    fsolver.set_options({"seed": 0, "num_trains": 10})
    solver.solve()
    fsolver.solve()
    val2 = solver.values
    fval2 = fsolver.values

    assert np.all(val1 == val2)
    assert np.all(fval1 == fval2)

    solver.set_options({"seed": 1, "num_trains": 10})
    fsolver.set_options({"seed": 1, "num_trains": 10})
    solver.solve()
    fsolver.solve()
    val3 = solver.values
    fval3 = fsolver.values

    assert not np.all(val1 == val3)
    assert not np.all(fval1 == fval3)
