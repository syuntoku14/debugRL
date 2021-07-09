import os

os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import argparse

import gym
import numpy as np
from misc import CONTINUOUS_SOLVERS, make_valid_options, prepare

from shinrl import solvers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", type=str, default="SAC", choices=list(CONTINUOUS_SOLVERS.keys())
    )
    parser.add_argument("--env", type=str, default="HalfCheetah-v2")
    defaults = {
        "--epochs": 30,
        "--evaluation_interval": 5000,
        "--steps_per_epoch": 100000,
    }
    args, project_name, task_name, task, logger = prepare(parser, defaults)

    # Construct solver
    env = gym.make(args.env)
    SOLVER = CONTINUOUS_SOLVERS[args.solver]
    options = make_valid_options(args, SOLVER)
    solver = SOLVER(env, logger=logger, solve_options=options)
    if args.clearml:
        task_params = task.connect(solver.solve_options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")

    # Run solver
    dir_name = os.path.join(
        "results", project_name, task_name, str(solver.solve_options["seed"])
    )
    if args.save and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for epoch in range(args.epochs):
        solver.run(num_steps=args.steps_per_epoch)
        if args.save:
            # save in "results/[project_name]/[task_name]/[seed]/"
            solver.save(dir_name)


if __name__ == "__main__":
    main()
