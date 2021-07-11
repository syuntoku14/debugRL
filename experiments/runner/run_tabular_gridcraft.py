import os

os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml import Task
from misc import TABULAR_DISCRETE_SOLVERS, make_valid_options, prepare, to_numeric

from shinrl.envs.gridcraft import GridEnv, create_maze, grid_spec_from_string

matplotlib.use("Agg")


def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        type=str,
        default="SFVI",
        choices=list(TABULAR_DISCRETE_SOLVERS.keys()),
    )
    parser.add_argument("--env", type=str, default="GridCraft-v0")  # dummy
    parser.add_argument("--obs_mode", type=str, default="one-hot")
    defaults = {
        "--epochs": 10,
        "--evaluation_interval": 1,
        "--steps_per_epoch": 10,
        "--log_interval": 1,
    }
    args, project_name, task_name, task, logger = prepare(parser, defaults)

    spec = grid_spec_from_string("SOOO\\" + "O###\\" + "OOOO\\" + "O#RO\\")
    if args.obs_mode == "one-hot":
        env = GridEnv(spec, trans_eps=0.1, horizon=25)
    else:
        env = GridEnv(spec, trans_eps=0.1, horizon=25, obs_mode="random", obs_dim=100)

    # solve tabular MDP
    SOLVER = TABULAR_DISCRETE_SOLVERS[args.solver]
    options = make_valid_options(args, SOLVER)
    solver = SOLVER(env, logger=logger, solve_options=options)

    if args.clearml:
        task_params = task.connect(solver.solve_options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

    for _ in range(args.epochs):
        # draw results
        solver.run(num_steps=args.steps_per_epoch)

    dir_name = os.path.join(
        "results", project_name, task_name, str(solver.solve_options["seed"])
    )
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    solver.save(os.path.join(dir_name))


if __name__ == "__main__":
    main()
