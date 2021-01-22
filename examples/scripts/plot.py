import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
import os.path as osp
import numpy as np
from collections import defaultdict

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

xlimits = defaultdict(lambda: 3e6)
xlimits.update({
    "Ant-v2": 8e6,
    "HalfCheetah-v2": 2.5e6
})

colors = sns.color_palette()
agents = ['PPO', 'TRPO', 'TD3', 'SAC', 'ERAC', 'ERCAC']
color_idx = [0, 5, 4, 2, 3, 1]


def plot_data(data, ax, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x

    data = pd.concat(data, ignore_index=True)
    for i, agent in enumerate(agents):
        color = colors[color_idx[i]]
        datum = data[data["Condition1"]==agent]
        if agent == "ERCAC":
            agent = "ERCAC (Ours)"
        elif agent == "ERAC":
            agent = r"ERCAC ($\zeta=1$)"
        sns.lineplot(data=datum, x=xaxis, y=value, ci='sd', ax=ax, label=agent,
                     legend=None, color=color, **kwargs)
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))


def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
            except:
                print('Could not read from %s' %
                      os.path.join(root, 'progress.txt'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns),
                            'Performance', exp_data[performance])
            datasets.append(exp_data)
    return datasets


def make_plot(logdir, ax, xaxis=None, value=None,
              font_scale=1.5, smooth=1, estimator='mean'):
    data = get_datasets(logdir)
    condition = 'Condition1'
    # choose what to show on main curve: mean? max? min?
    estimator = getattr(np, estimator)
    plot_data(data, ax, xaxis=xaxis, value=value, condition=condition,
                smooth=smooth, estimator=estimator)
    env_name = os.path.basename(logdir)
    ax.set_title(env_name)
    ax.set_xlim(right=xlimits[env_name])
    ax.set_ylabel("Average Return")
    ax.set_xlabel("Steps")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    value = "Performance"

    def get_immediate_subdirectories(a_dir):
        return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]

    sns.set(style="darkgrid", font_scale=2.5)
    env_dirs = get_immediate_subdirectories(args.logdir)
    env_dirs = sorted(env_dirs)
    num_envs = len(env_dirs)
    num_rows = math.ceil(num_envs / 3)
    fig, axes = plt.subplots(num_rows, 3, figsize=(num_envs*6, 6*2))
    if num_envs == 1:
        axes = [[axes]]

    for i, env_dir in enumerate(env_dirs):
        col = i % 3
        row = math.floor(i / 3)
        make_plot(env_dir, axes[row][col], args.xaxis, value,
                  smooth=args.smooth, estimator=args.est)

    handles = [None] * len(agents)
    agents[-1] = "ERCAC (Ours)"
    agents[-2] = r"ERCAC ($\zeta=1$)"
    for ax in axes.flatten():
        handle, label = ax.get_legend_handles_labels()
        for h, agent in zip(handle, label):
            handles[agents.index(agent)] = h

    fig.delaxes(axes[-1][-1])
    # lgd = fig.legend(handles, agents, loc="upper center",
    #                  bbox_to_anchor=(0.5, 1.1), ncol=len(agents))
    lgd = fig.legend(handles, agents, loc="lower right",
                     bbox_to_anchor=(0.88, 0.15), ncol=1)
    for line in lgd.get_lines():
        line.set_linewidth(4.0)

    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(args.logdir, value+".png"),
                bbox_extra_artists=(lgd, ), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
