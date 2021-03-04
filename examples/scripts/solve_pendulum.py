import os
os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import numpy as np
import argparse
from copy import deepcopy
from rlowan.envs.pendulum import (
    Pendulum, plot_pendulum_values, reshape_values)
from rlowan.solvers import *
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
    "SPG": SamplingPgSolver,
    "IPG": IpgSolver,
    "SAC": SacSolver,
    "SAC-Continuous": SacContinuousSolver,
    "PPO": PpoSolver
}


def to_numeric(arg):
    try:
        float(arg)
    except ValueError:
        return arg
    else:
        return int(float(arg)) if float(arg).is_integer() else float(arg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, default="SAC")
    parser.add_argument('--exp_name', type=str, default="Pendulum")
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--obs_mode', type=str,
                        default="tuple", choices=["tuple", "image"])

    # for arbitrary options, e.g. --device cpu --seed 0
    parsed, unknown = parser.parse_known_args() 
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split('=')[0])
    args = parser.parse_args()

    options = {}
    for key, val in vars(args).items():
        if key not in ["solver", "exp_name", "task_name", "epochs", "obs_mode"]:
            options[key] = to_numeric(val)

    task_name = args.solver if args.task_name is None else args.task_name
    project_name = args.exp_name + "-" + args.obs_mode
    task = Task.init(project_name=project_name,
                     task_name=task_name, reuse_last_task_id=False)
    logger = task.get_logger()

    # Construct the environment
    if "Continuous" in args.solver:
        action_mode, dA = "continuous", 50
    else:
        action_mode, dA = "discrete", 5
    env = Pendulum(horizon=200, state_disc=32,
                   dA=dA, obs_mode=args.obs_mode, action_mode=action_mode)

    # solve tabular MDP
    solver_cls = SOLVERS[args.solver]
    solver = solver_cls(env, logger=logger, solve_options=options)
    task_params = task.connect(solver.solve_options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")

    grid_kws = {"width_ratios": (.49, .49, .02)}
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(20, 6), gridspec_kw=grid_kws)
    camera = Camera(fig)

    for epoch in range(args.epochs):
        solver.run(num_steps=500)
        # learned value
        q_values = solver.values
        v_values = np.sum(solver.policy*q_values, axis=-1)

        # oracle value
        oracle_q_values = env.compute_action_values(solver.policy)
        oracle_v_values = np.sum(solver.policy*oracle_q_values, axis=-1)
        vmin = min(v_values.min(), oracle_v_values.min())
        vmax = max(v_values.max(), oracle_v_values.max())

        v_values = reshape_values(env, v_values)
        oracle_v_values = reshape_values(env, oracle_v_values)

        plot_pendulum_values(env, v_values, ax=axes[0], cbar_ax=axes[2],
                             vmin=vmin, vmax=vmax,
                             title="Learned State Values".format(epoch))
        plot_pendulum_values(env, oracle_v_values, ax=axes[1], cbar_ax=axes[2],
                             vmin=vmin, vmax=vmax,
                             title="Oracle State Values".format(epoch))
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
