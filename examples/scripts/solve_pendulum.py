import os
os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import numpy as np
import argparse
from debug_rl.envs.pendulum import (
    Pendulum, plot_pendulum_values, reshape_values)
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
    "EPG": ExactPgSolver,
    "SAC": SacSolver,
    "SAC-Continuous": SacContinuousSolver,
    "PPO": PpoSolver
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, default="SAC")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--exp_name', type=str, default="Pendulum-tuple")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    options = DEFAULT_OPTIONS
    options.update({
        "solver": args.solver,
        "device": args.device,
        "seed": args.seed
    })

    task_name = options["solver"]
    task = Task.init(project_name=args.exp_name, task_name=task_name)
    logger = task.get_logger()

    # Construct the environment
    if "Continuous" in options["solver"]:
        action_mode = "continuous"
        dA = 50
    else:
        action_mode = "discrete"
        dA = 5
    env = Pendulum(horizon=200, state_disc=32,
                   dA=dA, obs_mode="tuple", action_mode=action_mode)

    # solve tabular MDP
    solver_cls = SOLVERS[options["solver"]]
    del options["solver"]
    solver = solver_cls(env, logger=logger, solve_options=options)
    task_params = task.connect(solver.solve_options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")

    for _ in range(10):
        solver.run(num_steps=500)

        # draw results
        q_values = env.compute_action_values(solver.policy)
        v_values = np.sum(solver.policy*q_values, axis=-1)
        vmin = v_values.min()
        vmax = v_values.max()
        v_values = reshape_values(env, v_values)
        plot_pendulum_values(env, v_values, vmin=vmin,
                             vmax=vmax, title="State values")
        plt.show()

    env_name = env.__class__.__name__
    dir_name = os.path.join("results", env_name, solver_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    solver.save(os.path.join(dir_name, str(solver.solve_options["seed"])))


if __name__ == "__main__":
    main()
