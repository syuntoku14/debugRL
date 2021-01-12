import pytest
import numpy as np
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import (
    OracleViSolver,
    OracleCviSolver)


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    yield pend_env


def test_value_iteration(setUp):
    pend_env = setUp
    solver = OracleViSolver(pend_env)
    run_solver(solver, pend_env)


def test_cvi(setUp):
    pend_env = setUp
    solver = OracleCviSolver(pend_env)
    run_solver(solver, pend_env)


def run_solver(solver, env):
    solver.solve()
    policy = solver.compute_policy(solver.values)
    action_values = solver.env.compute_action_values(solver.policy)
    visitation = env.compute_visitation(solver.policy)
    assert solver.values.shape \
        == solver.policy.shape \
        == action_values.shape \
        == (env.dS, env.dA)
    assert visitation.shape \
        == (env.horizon, env.dS, env.dA)
