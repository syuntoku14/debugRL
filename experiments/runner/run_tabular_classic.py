import os

os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import argparse

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from clearml import Task
from misc import TABULAR_CONTINUOUS_SOLVERS, TABULAR_DISCRETE_SOLVERS, make_valid_options, prepare

from shinrl import solvers

matplotlib.use("Agg")


def main():
    parser = argparse.ArgumentParser()
    KEYS = set(list(TABULAR_DISCRETE_SOLVERS.keys()) + list(TABULAR_CONTINUOUS_SOLVERS.keys()))
    parser.add_argument("--solver", type=str, default="SFVI", choices=KEYS)
    parser.add_argument(
        "--env",
        type=str,
        default="TabularPendulum-v0",
        choices=[
            "TabularPendulum-v0",
            "TabularPendulumContinuous-v0",
            "TabularPendulumImage-v0",
            "TabularPendulumImageContinuous-v0",
            "TabularMountainCar-v0",
            "TabularMountainCarContinuous-v0",
            "TabularMountainCarImage-v0",
            "TabularMountainCarImageContinuous-v0",
            "TabularCartPole-v0",
        ],
    )
    defaults = {"--epochs": 5, "--evaluation_interval": 100, "--steps_per_epoch": 500}
    args, project_name, task_name, task, logger = prepare(parser, defaults)

    # Construct solver
    env = gym.make(args.env)
    SOLVER = (
        TABULAR_CONTINUOUS_SOLVERS
        if isinstance(env.action_space, gym.spaces.Box)
        else TABULAR_DISCRETE_SOLVERS
    )[args.solver]
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

        if hasattr(env, "plot_values") and logger is not None:
            # learned value
            q_values = solver.tb_values
            v_values = np.sum(solver.tb_policy * q_values, axis=-1)

            # oracle value
            oracle_q_values = env.compute_action_values(solver.tb_policy)
            oracle_v_values = np.sum(solver.tb_policy * oracle_q_values, axis=-1)
            vmin = min(v_values.min(), oracle_v_values.min())
            vmax = max(v_values.max(), oracle_v_values.max())

            if hasattr(env, "plot_values"):
                grid_kws = {"width_ratios": (0.49, 0.49, 0.02)}
                fig, axes = plt.subplots(
                    nrows=1, ncols=3, figsize=(20, 6), gridspec_kw=grid_kws
                )
                env.plot_values(
                    v_values,
                    ax=axes[0],
                    cbar_ax=axes[2],
                    vmin=vmin,
                    vmax=vmax,
                    title="Learned State Values".format(epoch),
                )
                env.plot_values(
                    oracle_v_values,
                    ax=axes[1],
                    cbar_ax=axes[2],
                    vmin=vmin,
                    vmax=vmax,
                    title="Oracle State Values".format(epoch),
                )
                solver.logger.report_matplotlib_figure(
                    "Q matrix", "ignore", solver.step, fig
                )
                plt.clf()
                plt.close()


if __name__ == "__main__":
    main()
