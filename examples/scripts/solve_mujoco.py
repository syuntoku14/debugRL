import os
os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import gym
import numpy as np
import argparse
from shinrl import solvers
from clearml import Task
from misc import CONTINUOUS_SOLVERS, to_numeric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, default="SAC",
                        choices=list(CONTINUOUS_SOLVERS.keys()))
    parser.add_argument('--env', type=str, default="HalfCheetah-v2")
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5)

    # For arbitrary options, e.g. --device cpu --seed 0
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split('=')[0])
    args = parser.parse_args()

    options = {"record_performance_interval": 5000}
    for key, val in vars(args).items():
        if key not in ["solver", "env", "exp_name", "task_name", "epochs"]:
            options[key] = to_numeric(val)

    project_name = args.env if args.exp_name is None else args.exp_name
    task_name = args.solver if args.task_name is None else args.task_name
    task = Task.init(project_name=project_name,
                     task_name=task_name, reuse_last_task_id=False)
    logger = task.get_logger()

    # Construct solver
    env = gym.make(args.env)
    solver = CONTINUOUS_SOLVERS[args.solver](env, logger=logger, solve_options=options)
    task_params = task.connect(solver.solve_options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")

    # Run solver
    for epoch in range(args.epochs):
        solver.run(num_steps=10000)

    dir_name = os.path.join("results", project_name, task_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    solver.save(os.path.join(dir_name, str(solver.solve_options["seed"])))


if __name__ == "__main__":
    main()
