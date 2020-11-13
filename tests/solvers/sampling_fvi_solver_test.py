import pytest
import numpy as np
from debug_rl.envs.pendulum import Pendulum
from debug_rl.solvers import (
    SamplingFittedViSolver,
    SamplingFittedCviSolver
)
from .oracle_vi_solver_test import run_solver
import torch


@pytest.fixture
def setUp():
    pend_env = Pendulum(state_disc=5, num_actions=3, horizon=5)
    pend_env.reset()

    solve_options = {
        "num_trains": 5,
    }
    yield pend_env, solve_options


def test_value_iteration(setUp):
    pend_env, solve_options = setUp
    buffer_options = {"use_replay_buffer": False}
    buffer_options.update(solve_options)
    solver = SamplingFittedViSolver(pend_env)
    solver.set_options(buffer_options)
    run_solver(solver, pend_env)


def test_cvi(setUp):
    pend_env, solve_options = setUp
    buffer_options = {"use_replay_buffer": False}
    buffer_options.update(solve_options)
    solver = SamplingFittedCviSolver(pend_env)
    solver.set_options(buffer_options)
    run_solver(solver, pend_env)


def test_image_obs(setUp):
    _, solve_options = setUp
    pend_env = Pendulum(state_disc=5, num_actions=3,
                        horizon=5, obs_mode="image")
    pend_env.reset()
    solver = SamplingFittedViSolver(pend_env)
    buffer_options = {"use_replay_buffer": True}
    buffer_options.update(solve_options)
    solver.set_options(buffer_options)
    run_solver(solver, pend_env)


def test_target_network(setUp):
    pend_env, solve_options = setUp
    solver = SamplingFittedViSolver(pend_env)
    buffer_options = {"use_target_network": True, "target_update_interval": 1}
    buffer_options.update(solve_options)
    solver.set_options(buffer_options)
    run_solver(solver, pend_env)
    net_tensors = solver.value_network.state_dict().values()
    tnet_tensors = solver.target_value_network.state_dict().values()
    for net, tnet in zip(net_tensors, tnet_tensors):
        assert torch.allclose(net, tnet)
