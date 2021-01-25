import os
os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import numpy as np
import argparse
from debug_rl.envs.gridcraft import (
    GridEnv, OneHotObsWrapper, spec_from_string, plot_grid_values)
from debug_rl.solvers import *
import matplotlib.pyplot as plt
import matplotlib
from celluloid import Camera
from clearml import Task
matplotlib.use('Agg')


SOLVERS = {
    "VI": SamplingViSolver,
    "CVI": SamplingCviSolver,
    "FVI": SamplingFittedViSolver,
    "FCVI": SamplingFittedCviSolver,
    "EPG": ExactPgSolver,
    "SAC": SacSolver,
    "PPO": PpoSolver
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, default="SAC")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--exp_name', type=str, default="GridCraft")
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    options = {
        "device": args.device,
        "seed": args.seed
    }

    task_name = args.solver if args.task_name is None else args.task_name
    project_name = args.exp_name
    task = Task.init(project_name=project_name,
                     task_name=task_name, reuse_last_task_id=False)
    logger = task.get_logger()

    # Construct the environment
    maze = spec_from_string("SOOO\\" +
                            "OLLL\\" +
                            "OOOO\\" +
                            "OLRO\\")
    env = GridEnv(maze, trans_eps=0.1, horizon=20)
    env = OneHotObsWrapper(env)

    # solve tabular MDP
    solver_cls = SOLVERS[args.solver]
    solver = solver_cls(env, logger=logger, solve_options=options)
    task_params = task.connect(solver.solve_options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    camera = Camera(fig)

    for _ in range(args.epochs):
        # draw results
        solver.run(num_steps=500)
        plot_grid_values(env.gs, solver.policy, ax=axes[0], title="Policy")
        plot_grid_values(env.gs, solver.values,
                         ax=axes[1], title="Learned Q Values")
        plot_grid_values(env.gs, env.compute_action_values(
            solver.policy), ax=axes[2], title="Oracle Q Values")
        camera.snap()

    dir_name = os.path.join("results", project_name, solver_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    solver.save(os.path.join(dir_name, str(solver.solve_options["seed"])))

    animation = camera.animate()
    animation.save(os.path.join(
        dir_name, "values-{}.mp4".format(solver.solve_options["seed"])))


if __name__ == "__main__":
    main()
