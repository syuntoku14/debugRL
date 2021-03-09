import os
os.environ["MKL_NUM_THREADS"] = "1"  # NOQA
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NOQA
os.environ["OMP_NUM_THREADS"] = "1"  # NOQA
import gym
import numpy as np
import argparse
from shinrl import solvers
import matplotlib.pyplot as plt
import matplotlib
from celluloid import Camera
from clearml import Task
from misc import DISCRETE_SOLVERS, CONTINUOUS_SOLVERS, to_numeric
matplotlib.use('Agg')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, default="SAC",
                        choices=list(DISCRETE_SOLVERS.keys()))
    parser.add_argument('--env', type=str, default="TabularPendulum-v0",
                        choices=[
                            'TabularPendulum-v0',
                            'TabularPendulumContinuous-v0',
                            'TabularPendulumImage-v0',
                            'TabularPendulumImageContinuous-v0',
                            'TabularMountainCar-v0',
                            'TabularMountainCarContinuous-v0',
                            'TabularMountainCarImage-v0',
                            'TabularMountainCarImageContinuous-v0',
                            'TabularCartPole-v0',
                        ])
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5)

    # For arbitrary options, e.g. --device cpu --seed 0
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split('=')[0])
    args = parser.parse_args()

    options = {}
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
    SOLVERS = CONTINUOUS_SOLVERS if isinstance(
        env.action_space, gym.spaces.Box) else DISCRETE_SOLVERS
    solver = SOLVERS[args.solver](env, logger=logger, solve_options=options)
    task_params = task.connect(solver.solve_options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")

    # Make directory to save
    dir_name = os.path.join("results", project_name,
                            task_name, str(solver.solve_options["seed"]))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if hasattr(env, "plot_values"):
        grid_kws = {"width_ratios": (.49, .49, .02)}
        fig, axes = plt.subplots(
            nrows=1, ncols=3, figsize=(20, 6), gridspec_kw=grid_kws)
        camera = Camera(fig)

    # Run solver
    for epoch in range(args.epochs):
        solver.run(num_steps=500)
        solver.save(os.path.join(dir_name, "data.pt"))

        if hasattr(env, "plot_values"):
            # learned value
            q_values = solver.tb_values
            v_values = np.sum(solver.tb_policy*q_values, axis=-1)

            # oracle value
            oracle_q_values = env.compute_action_values(solver.tb_policy)
            oracle_v_values = np.sum(solver.tb_policy*oracle_q_values, axis=-1)
            vmin = min(v_values.min(), oracle_v_values.min())
            vmax = max(v_values.max(), oracle_v_values.max())
            env.plot_values(v_values, ax=axes[0], cbar_ax=axes[2],
                            vmin=vmin, vmax=vmax,
                            title="Learned State Values".format(epoch))
            env.plot_values(oracle_v_values, ax=axes[1], cbar_ax=axes[2],
                            vmin=vmin, vmax=vmax,
                            title="Oracle State Values".format(epoch))
            camera.snap()

    if hasattr(env, "plot_values"):
        animation = camera.animate()
        animation.save(os.path.join(
            dir_name, "values-{}.mp4".format(solver.solve_options["seed"])))


if __name__ == "__main__":
    main()
