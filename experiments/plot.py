import math
import os
import os.path as osp
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def load_data(exp_path, x, y):
    data = {}
    for solver in exp_path.glob("*"):
        if not solver.is_dir():
            continue
        solver_name = solver.name
        data[solver_name] = []
        for seed in solver.glob("*"):
            datum = torch.load(str(seed) + "/data.tar")
            datum = pd.DataFrame(datum["history"][y])
            if x == "Samples":
                num_samples = datum["options"]["num_samples"]
                datum["x"] = datum["x"] * num_samples
            datum = datum.rename(columns={"y": y, "x": x})
            data[solver_name].append(datum)
        data[solver_name] = pd.concat(data[solver_name])
    return data


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("expdir")
    parser.add_argument("-x", default="Samples", choices=["Samples", "Steps"])
    parser.add_argument("-y", default="ReturnPolicy")
    parser.add_argument("--log_scale", action="store_true")
    args = parser.parse_args()

    exp_path = pathlib.Path(args.expdir)
    data = load_data(exp_path, args.x, args.y)

    sns.set(style="darkgrid", font_scale=1.0)
    ax = plt.gca()
    for name, datum in data.items():
        sns.lineplot(data=datum, x=args.x, y=args.y, ci="sd", ax=ax, label=name)
        xscale = np.max(np.asarray(datum[args.x])) > 5e3
        if xscale:
            # Just some formatting niceness: x-axis scale in scientific notation if max x is large
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    if args.log_scale:
        plt.yscale("log")
    plt.savefig(os.path.join(args.expdir, args.y + ".png"))


if __name__ == "__main__":
    main()
