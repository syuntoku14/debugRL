import gym
import numpy as np
import pytest
import torch

from shinrl.envs import Pendulum
from shinrl.solvers.tabular.continuous.pg import PpoSolver, SamplingPgSolver

from ..misc import run_solver_gym, run_solver_tb


@pytest.fixture
def setUp():
    tb_env = Pendulum(state_disc=5, dA=4, horizon=5, action_mode="continuous")
    tb_env.reset()
    yield tb_env


def test_tb(setUp):
    tb_env = setUp
    solver = SamplingPgSolver(
        tb_env, solve_options={"num_samples": 10, "num_minibatches": 1}
    )
    run_solver_tb(solver, tb_env)

    solver = PpoSolver(tb_env, solve_options={"num_samples": 10, "num_minibatches": 1})
    run_solver_tb(solver, tb_env)
