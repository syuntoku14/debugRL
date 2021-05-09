import os

os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import argparse

import gym
import numpy as np
from clearml import Task
from misc import CONTINUOUS_SOLVERS, DISCRETE_SOLVERS

from shinrl import solvers, utils


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", type=str, default="SAC", choices=list(DISCRETE_SOLVERS.keys())
    )
    parser.add_argument("--env", type=str, default="HalfCheetah-v2")
    parser.add_argument("--load_path", type=str, default=None)
    args = parser.parse_args()

    # Construct solver
    env = gym.make(args.env)
    SOLVERS = (
        CONTINUOUS_SOLVERS
        if isinstance(env.action_space, gym.spaces.Box)
        else DISCRETE_SOLVERS
    )
    solver = SOLVERS[args.solver](env)
    solver.load(args.load_path)
    print("Loaded solver from {}".format(args.load_path))

    # Run solver
    env.obs = env.reset()
    utils.collect_samples(env, solver.get_action_gym, num_episodes=10, render=True)


if __name__ == "__main__":
    main()
