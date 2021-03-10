import os
os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import numpy as np
import argparse
from shinrl.envs.gridcraft import (GridEnv, grid_spec_from_string)
from shinrl.solvers import *
import matplotlib.pyplot as plt
import matplotlib
from celluloid import Camera
from clearml import Task
from misc import DISCRETE_SOLVERS, to_numeric
matplotlib.use('Agg')


def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, default="SAC",
                        choices=list(DISCRETE_SOLVERS.keys()))
    parser.add_argument('--exp_name', type=str, default="GridCraft")
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5)

    # for arbitrary options, e.g. --device cpu --seed 0
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split('=')[0])
    args = parser.parse_args()

    options = {}
    for key, val in vars(args).items():
        if key not in ["solver", "exp_name", "task_name", "epochs"]:
            options[key] = to_numeric(val)

    project_name = args.exp_name
    task_name = args.solver if args.task_name is None else args.task_name
    task = Task.init(project_name=project_name,
                     task_name=task_name, reuse_last_task_id=False)
    logger = task.get_logger()

    # Construct the environment
    maze = grid_spec_from_string("SOOO\\" +
                                 "OLLL\\" +
                                 "OOOO\\" +
                                 "OLRO\\")
    env = GridEnv(maze, trans_eps=0.1, horizon=20, obs_mode="onehot")

    # solve tabular MDP
    solver_cls = DISCRETE_SOLVERS[args.solver]
    solver = solver_cls(env, logger=logger, solve_options=options)
    task_params = task.connect(solver.solve_options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    camera = Camera(fig)

    for _ in range(args.epochs):
        # draw results
        solver.run(num_steps=500)
        env.plot_values(solver.tb_policy, ax=axes[0], title="Policy")
        env.plot_values(solver.tb_values, ax=axes[1], title="Learned Q Values")
        env.plot_values(env.compute_action_values(
            solver.tb_policy), ax=axes[2], title="Oracle Q Values")
        camera.snap()

    dir_name = os.path.join("results", project_name, task_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    solver.save(os.path.join(dir_name, str(solver.solve_options["seed"])))

    animation = camera.animate()
    animation.save(os.path.join(
        dir_name, "values-{}.mp4".format(solver.solve_options["seed"])))


if __name__ == "__main__":
    main()
