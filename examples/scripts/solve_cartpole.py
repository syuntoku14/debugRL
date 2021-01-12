import os
os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import numpy as np
from debug_rl.envs.cartpole import CartPole
from debug_rl.solvers import *
from debug_rl.utils import boltzmann_softmax, mellow_max
from debug_rl.solvers.base import DEFAULT_OPTIONS
import matplotlib.pyplot as plt
import matplotlib
from clearml import Task
matplotlib.use('Agg')

SOLVERS = {
    "VI": SamplingViSolver,
    "CVI": SamplingCviSolver,
    "FVI": SamplingFittedViSolver,
    "FCVI": SamplingFittedCviSolver,
    "SAC": SacSolver,
}


options = DEFAULT_OPTIONS
options.update({
    "solver": "SAC",
    "num_trains": 10000,
    "polyak": 0.995,
    })


def main():
    task_name = options["solver"]
    project_name = "cartpole-tuple"
    task = Task.init(project_name=project_name, task_name=task_name)
    logger = task.get_logger()
    task_params = task.connect(options)

    # Construct the environment
    env = CartPole()

    # solve tabular MDP
    solver_cls = SOLVERS[options["solver"]]
    del options["solver"]
    solver = solver_cls(env, logger=logger, solve_options=options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")
    solver.solve()


if __name__ == "__main__":
    main()
