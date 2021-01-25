import os
os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import numpy as np
import argparse
from debug_rl.envs.cartpole import CartPole
from debug_rl.solvers import *
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
    "PPO": PpoSolver
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, default="SAC")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--exp_name', type=str, default="CartPole")
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
    env = CartPole()

    # solve tabular MDP
    solver_cls = SOLVERS[args.solver]
    solver = solver_cls(env, logger=logger, solve_options=options)
    task_params = task.connect(solver.solve_options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")

    for _ in range(args.epochs):
        solver.run(1000)

    dir_name = os.path.join("results", project_name, solver_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    solver.save(os.path.join(dir_name, str(solver.solve_options["seed"])))


if __name__ == "__main__":
    main()
