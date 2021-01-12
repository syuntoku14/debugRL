import os
os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
from clearml import Task
import matplotlib
import matplotlib.pyplot as plt
from debug_rl.solvers.base import DEFAULT_OPTIONS
from debug_rl.utils import boltzmann_softmax, mellow_max
from debug_rl.solvers import *
from debug_rl.envs.pendulum import (
    Pendulum, plot_pendulum_values, reshape_values)
import numpy as np

matplotlib.use('Agg')

SOLVERS = {
    "VI": SamplingViSolver,
    "CVI": SamplingCviSolver,
    "FVI": SamplingFittedViSolver,
    "FCVI": SamplingFittedCviSolver,
    "SAC": SacSolver,
    "SAC-Continuous": SacContinuousSolver
}

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

options = DEFAULT_OPTIONS
options.update({
    "solver": "SAC",
    "num_trains": 30000,
    "device": "cuda:0",
    })


def main():
    task_name = options["solver"]
    project_name = "pendulum-image"
    task = Task.init(project_name=project_name, task_name=task_name)
    logger = task.get_logger()

    # Construct the environment
    if "Continuous" in options["solver"]:
        action_mode = "continuous"
        dA = 50
    else:
        action_mode = "discrete"
        dA = 5
    env = Pendulum(horizon=200, state_disc=32,
                   dA=dA, obs_mode="image", action_mode=action_mode)

    # solve tabular MDP
    solver_cls = SOLVERS[options["solver"]]
    del options["solver"]
    solver = solver_cls(env, logger=logger, solve_options=options)
    task_params = task.connect(solver.solve_options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")
    solver.solve()
    env.compute_policy(solver.values)

    # draw results
    q_values = env.compute_action_values(solver.policy)
    v_values = np.sum(solver.policy*q_values, axis=-1)
    vmin = v_values.min()
    vmax = v_values.max()
    v_values = reshape_values(env, v_values)
    plot_pendulum_values(env, v_values, vmin=vmin,
                         vmax=vmax, title="State values")
    plt.show()

    # dir_name = os.path.join("results", solver_name)
    # if not os.path.exists(dir_name):
    #     os.makedirs(dir_name)
    # solver.save(os.path.join(dir_name, task.id))


if __name__ == "__main__":
    main()
