import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from debug_rl.utils import make_image


def reshape_values(env, values):
    # reshape the values(one-dimensional array) into state_disc x state_disc
    reshaped_values = np.empty((env.state_disc, env.state_disc))
    reshaped_values[:] = np.nan
    for s in range(len(values)):
        pos, vel = env.pos_vel_from_state_id(s)
        pos, vel = env.disc_pos_vel(pos, vel)
        reshaped_values[pos, vel] = values[s]
    return reshaped_values


def plot_mountaincar_values(
        env, reshaped_values, title=None, ax=None, cbar_ax=None,
        return_image=False, vmin=None, vmax=None):
    # reshaped_values: state_disc x state_disc
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1,
                               figsize=(8, 6))
    vmin = reshaped_values.min() if vmin is None else vmin
    vmax = reshaped_values.max() if vmax is None else vmax

    pos_ticks, vel_ticks = [], []
    for i in range(env.state_disc):
        pos, vel = env.undisc_pos_vel(i, i)
        pos_ticks.append(round(pos, 3))
        vel_ticks.append(round(vel, 3))

    data = pd.DataFrame(
        reshaped_values, index=pos_ticks,
        columns=vel_ticks)
    data = data.ffill(downcast='infer', axis=0)
    data = data.ffill(downcast='infer', axis=1)
    sns.heatmap(data, ax=ax, cbar=True, cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_ylabel(r"$x$")
    ax.set_xlabel(r"$\dot{x}$")

    if return_image:
        return make_image()
