import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from debug_rl.utils import gen_image


def reshape_values(env, values):
    # reshape the values(one-dimensional array) into state_disc x state_disc
    reshaped_values = np.empty((env.state_disc, env.state_disc))
    reshaped_values[:] = np.nan
    for s in range(len(values)):
        th, thv = env.th_thv_from_state_id(s)
        th, thv = env.disc_th_thv(th, thv)
        reshaped_values[th, thv] = values[s]
    return reshaped_values


def plot_pendulum_values(
        env, reshaped_values, title=None,
        return_image=False, vmin=None, vmax=None):
    # reshaped_values: state_disc x state_disc
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           figsize=(8, 6))
    vmin = reshaped_values.min() if vmin is None else vmin
    vmax = reshaped_values.max() if vmax is None else vmax

    th_ticks, thv_ticks = [], []
    for i in range(env.state_disc):
        th, thv = env.undisc_th_thv(i, i)
        th_ticks.append(round(th, 3))
        thv_ticks.append(round(thv, 3))

    data = pd.DataFrame(
        reshaped_values, index=th_ticks,
        columns=thv_ticks)
    data = data.ffill(downcast='infer', axis=0)
    data = data.ffill(downcast='infer', axis=1)
    sns.heatmap(data, ax=ax, cbar=True, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_ylabel(r"$\theta$")
    ax.set_xlabel(r"$\dot{\theta}$")

    if return_image:
        return gen_image()
    else:
        plt.show()
