import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
import os.path as osp
import numpy as np
import pandas as pd
import pickle
import torch
import pathlib


def load_data(exp_path):
    data = {}
    for solver in exp_path.glob("*"):
        if not solver.is_dir():
            continue
        solver_name = solver.name
        data[solver_name] = []
        for seed in solver.glob("*"):
            datum = torch.load(str(seed))
            num_samples = datum["options"]["num_samples"]
            datum = pd.DataFrame(datum["history"]["ReturnPolicy"])
            datum["x"] = datum["x"] * num_samples
            datum = datum.rename(columns={"y": "Average return", "x": "Samples"})
            data[solver_name].append(datum)
        data[solver_name] = pd.concat(data[solver_name])
    return data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir')
    parser.add_argument('-x', default='Samples')
    parser.add_argument('-y', default="Performance")
    args = parser.parse_args()

    exp_path = pathlib.Path(args.expdir)
    data = load_data(exp_path)

    sns.set(style="darkgrid", font_scale=1.0)
    ax = plt.gca()
    for name, datum in data.items():
        sns.lineplot(data=datum, x="Samples", y="Average return", ci='sd', ax=ax, label=name)
        xscale = np.max(np.asarray(datum["Samples"])) > 5e3
        if xscale:
            # Just some formatting niceness: x-axis scale in scientific notation if max x is large
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.savefig(os.path.join(args.expdir, args.y+".png"))
    plt.show()


if __name__ == "__main__":
    main()
