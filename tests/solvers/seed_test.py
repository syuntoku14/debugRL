import pytest
import numpy as np
from shinrl.envs import Pendulum
from shinrl.solvers.vi.discrete import SamplingViSolver, SamplingFittedViSolver
from .misc import run_solver_tb


def test_seed_tabular():
    env = Pendulum(state_disc=32, dA=5, horizon=200)
    env.reset()
    solver = SamplingViSolver(env)
    fsolver = SamplingFittedViSolver(env)
    solver.initialize({"seed": 0, "record_performance_interval": 1})
    fsolver.initialize({"seed": 0, "record_performance_interval": 1})
    solver.run(num_steps=10)
    fsolver.run(num_steps=10)
    val1 = solver.tb_values
    fval1 = fsolver.tb_values

    solver.initialize({"seed": 0, "record_performance_interval": 1})
    fsolver.initialize({"seed": 0, "record_performance_interval": 1})
    solver.run(num_steps=10)
    fsolver.run(num_steps=10)
    val2 = solver.tb_values
    fval2 = fsolver.tb_values

    assert np.all(val1 == val2)
    assert np.all(fval1 == fval2)

    solver.initialize({"seed": 1, "record_performance_interval": 1})
    fsolver.initialize({"seed": 1, "record_performance_interval": 1})
    solver.run(num_steps=10)
    fsolver.run(num_steps=10)
    val3 = solver.tb_values
    fval3 = fsolver.tb_values

    assert not np.all(val1 == val3)
    assert not np.all(fval1 == fval3)
