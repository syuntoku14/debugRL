import os
os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import numpy as np
from debug_rl.envs.gridcraft import (
    GridEnv, OneHotObsWrapper, spec_from_string, plot_grid_values)
from debug_rl.solvers import *
from debug_rl.utils import boltzmann_softmax, mellow_max
import matplotlib.pyplot as plt
import matplotlib
from clearml import Task
matplotlib.use('Agg')

SOLVERS = {
    "TVI": SamplingViSolver,
    "TCVI": SamplingCviSolver,
    "FVI": SamplingFittedViSolver,
    "FCVI": SamplingFittedCviSolver,
}


options = {
    "solver": "TVI",
}


def main():
    task_name = options["solver"]
    task = Task.init(project_name='gridcraft', task_name=task_name)
    logger = task.get_logger()
    task_params = task.connect(options)

    # Construct the environment
    maze = spec_from_string("SOOO\\" +
                            "OLLL\\" +
                            "OOOO\\" +
                            "OLRO\\")
    env = GridEnv(maze, trans_eps=0.1, horizon=20)
    env = OneHotObsWrapper(env)

    # solve tabular MDP
    solver_cls = SOLVERS[options["solver"]]
    del options["solver"]
    solver = solver_cls(env, logger=logger, solve_options=options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")
    solver.solve(num_steps=1000)
    solver.compute_policy(solver.values)

    # draw results
    plot_grid_values(env.gs, solver.values, title="Values")
    plt.show()
    plot_grid_values(
        env.gs,
        env.compute_action_values(solver.policy),
        title="Q values")
    plt.show()
    plot_grid_values(env.gs, solver.policy, title="Policy")
    plt.show()


if __name__ == "__main__":
    main()
