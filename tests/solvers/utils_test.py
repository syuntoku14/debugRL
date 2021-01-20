import pytest
import numpy as np
import torch
from torch import nn
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import (
    OracleViSolver,
    OracleCviSolver)
from debug_rl.utils import *


@pytest.fixture
def setUp():
    env = Pendulum(state_disc=5, dA=3)
    env.reset()
    solver = OracleViSolver(env)

    cont_env = Pendulum(state_disc=5, dA=3, action_mode="continuous")
    cont_env.reset()
    cont_solver = OracleViSolver(cont_env)
    yield env, solver, cont_env, cont_solver


def test_max_operator():
    # check shape
    beta = 0.9
    sample = np.random.rand(2, 3, 4)
    assert boltzmann_softmax(sample, beta=beta).shape == sample.shape[:-1]
    assert mellow_max(sample, beta=beta).shape == sample.shape[:-1]
    sample = torch.randn((2, 3, 4))
    assert boltzmann_softmax(sample, beta=beta).shape == sample.shape[:-1]
    assert mellow_max(sample, beta=beta).shape == sample.shape[:-1]

    # value check
    sample = np.array([2, 3])
    res = boltzmann_softmax(sample, beta=beta)
    np.testing.assert_almost_equal(
        res, np.sum(np.exp(beta * sample) * sample, axis=-1)
        / np.sum(np.exp(beta * sample), axis=-1))

    res = mellow_max(sample, beta=beta)
    np.testing.assert_almost_equal(
        res, np.log(np.mean(np.exp(beta*sample), axis=-1)) / beta)


def test_policy():
    value = np.random.rand(3, 4)
    assert eps_greedy_policy(value).shape == value.shape
    assert softmax_policy(value).shape == value.shape


def test_collect_samples(setUp):
    env, solver, cont_env, cont_solver = setUp
    solver.run()
    policy = solver.compute_policy(solver.values)
    traj = collect_samples(env, policy, 10)
    assert len(traj["obs"]) == 10
    assert (traj["act"]).dtype == np.long

    cont_solver.run()
    policy = cont_solver.compute_policy(cont_solver.values)
    traj = collect_samples(cont_env, policy, 10)
    assert (traj["obs"]).dtype == np.float32
