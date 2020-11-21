import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
from debug_rl.envs.gridcraft import (
    GridEnv, OneHotObsWrapper, spec_from_string, plot_grid_values)
from debug_rl.solvers import *
from debug_rl.utils import boltzmann_softmax, mellow_max
import matplotlib.pyplot as plt
import matplotlib
from trains import Task
matplotlib.use('Agg')

SOLVERS = {
    "TVI": SamplingViSolver,
    "TCVI": SamplingCviSolver,
    "FVI": SamplingFittedViSolver,
    "FCVI": SamplingFittedCviSolver,
}


options = {
    "solver": "TVI",
    # general settings
    "seed": 0,
    "discount": 0.99,
    "num_samples": 200,  # number of samples to collect
    "buffer_size": 1e6,
    # CVI settings
    "alpha": 0.9,
    "beta": 10,
    "max_operator": "boltzmann_softmax",
    # Safe CVI settings
    "use_oracle_distribution": True,  # use compute_visitation to compute d_pi if True
    "use_oracle_advantage": True,  # use compute_action_values to compute advantage if True
    # FVI settings
    "activation": "relu",
    "hidden": 128,  # size of hidden layer
    "depth": 2,  # depth of the network
    "device": "cpu",
    "lr": 0.001,  # learning rate of a nn
    "num_trains": 2000,  # for sampling methods
    "proj_iters": 1,  # num of updates per a minibatch
    "use_target_network": True,
    "target_update_interval": 50,
    "use_replay_buffer": True,
    "minibatch_size": 128,  # size of minibatch when use_replay_buffer is True
    # Tabular settings
    "lr": 0.1,  # learning rate of tabular methods
    "num_trains": 1000,  # for sampling methods
}


def main():
    # task_name = '{}-α:{}-β:{}'.format(options["solver"], options["alpha"], options["beta"])
    task_name = "q-learning"
    task = Task.init(project_name='gridcraft', task_name=task_name)
    logger = task.get_logger()
    task_params = task.connect(options)

    # Construct the environment
    maze = spec_from_string("SOOO\\" +
                            "OLLL\\" +
                            "OOOO\\" +
                            "OLRO\\")
    env = GridEnv(maze, trans_eps=0.1, horizon=20)
    env = OneHotObsWrapper(env)

    # solve tabular MDP
    solver_cls = SOLVERS[options["solver"]]
    del options["solver"]
    solver = solver_cls(env, logger=logger, solve_options=options)
    solver_name = solver.__class__.__name__
    print(solver_name, "starts...")
    solver.solve()
    solver.compute_policy(solver.values)
    solver.compute_visitation(solver.policy)

    # draw results
    plot_grid_values(solver.env.gs, solver.values, title="Values")
    plot_grid_values(
        solver.env.gs,
        solver.compute_action_values(solver.policy),
        title="Q values")
    plot_grid_values(solver.env.gs, solver.policy, title="Policy")
    plot_grid_values(solver.env.gs, np.sum(
        solver.visitation, axis=0), title="Visitation")

    dir_name = os.path.join("results", solver_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    solver.save(os.path.join(dir_name, task.id))


if __name__ == "__main__":
    main()
