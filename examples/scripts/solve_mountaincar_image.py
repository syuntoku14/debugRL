import os
os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import numpy as np
from debug_rl.envs.mountaincar import MountainCar
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
    "SAC-Continuous": SacContinuousSolver
}


options = DEFAULT_OPTIONS
options.update({
    "solver": "SAC",
    "device": "cuda:0",
    })


def main():
    task_name = options["solver"]
    project_name = "mountaincar-image"
    task = Task.init(project_name=project_name, task_name=task_name)
    logger = task.get_logger()

    # Construct the environment
    if "Continuous" in options["solver"]:
        action_mode = "continuous"
        dA = 50
    else:
        action_mode = "discrete"
        dA = 5
    # Construct the environment
    env = MountainCar(horizon=200, state_disc=32, dA=dA,
                      action_mode=action_mode, obs_mode="image")

    # solve tabular MDP
    solver_cls = SOLVERS[options["solver"]]
    del options["solver"]
    solver = solver_cls(env, logger=logger, solve_options=options)
    task_params = task.connect(solver.solve_options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")
    solver.run(num_steps=200000)
    solver.compute_policy(solver.values)

    # dir_name = os.path.join("results", solver_name)
    # if not os.path.exists(dir_name):
    #     os.makedirs(dir_name)
    # solver.save(os.path.join(dir_name, task.id))


if __name__ == "__main__":
    main()
