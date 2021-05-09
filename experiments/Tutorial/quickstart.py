import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from celluloid import Camera

from shinrl.solvers.sac.discrete import SacSolver

device = "cuda" if torch.cuda.is_available() else "cpu"
env = gym.make("TabularPendulum-v0")
solver = SacSolver(env, solve_options={"device": device})

# for animation
grid_kws = {"width_ratios": (0.49, 0.49, 0.02)}
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), gridspec_kw=grid_kws)
camera = Camera(fig)

for _ in range(10):
    solver.run(num_steps=500)
    # learned value
    q_values = solver.tb_values
    v_values = np.sum(solver.tb_policy * q_values, axis=-1)
    # oracle value
    oracle_q_values = env.compute_action_values(
        solver.tb_policy, er_coef=solver.solve_options["er_coef"]
    )
    oracle_v_values = np.sum(solver.tb_policy * oracle_q_values, axis=-1)
    # plot values
    vmin = min(v_values.min(), oracle_v_values.min())
    vmax = max(v_values.max(), oracle_v_values.max())
    env.plot_values(
        v_values,
        ax=axes[0],
        cbar_ax=axes[2],
        vmin=vmin,
        vmax=vmax,
        title="Learned State Values",
    )
    env.plot_values(
        oracle_v_values,
        ax=axes[1],
        cbar_ax=axes[2],
        vmin=vmin,
        vmax=vmax,
        title="Oracle State Values",
    )
    camera.snap()

# create gif animation
animation = camera.animate()
animation.save("../assets/values.gif")
