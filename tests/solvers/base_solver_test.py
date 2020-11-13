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


def compute_visitation(solver):
    return solver.compute_visitation(solver.policy)


def compute_expected_return(solver):
    return solver.compute_expected_return(solver.policy)


def test_compute_visitation(setUp, benchmark):
    env = setUp
    solver = OracleViSolver(env)
    solver.solve()
    solver.compute_policy(solver.values)
    benchmark.pedantic(compute_visitation,
                       kwargs={"solver": solver})


def test_compute_expected_return(setUp, benchmark):
    env = setUp
    solver = OracleViSolver(env)
    solver.solve()
    solver.compute_policy(solver.values)
    benchmark.pedantic(compute_expected_return,
                       kwargs={"solver": solver})


def test_seed(setUp):
    env = setUp
    solver = SamplingViSolver(env)
    fsolver = SamplingFittedViSolver(env)
    solver.set_options({"seed": 0, "num_trains": 10})
    fsolver.set_options({"seed": 0, "num_trains": 10})
    val1 = solver.solve()
    fval1 = fsolver.solve()

    solver.set_options({"seed": 0, "num_trains": 10})
    fsolver.set_options({"seed": 0, "num_trains": 10})
    val2 = solver.solve()
    fval2 = fsolver.solve()

    assert np.all(val1 == val2)
    assert np.all(fval1 == fval2)

    solver.set_options({"seed": 1, "num_trains": 10})
    fsolver.set_options({"seed": 1, "num_trains": 10})
    val3 = solver.solve()
    fval3 = fsolver.solve()

    assert not np.all(val1 == val3)
    assert not np.all(fval1 == fval3)