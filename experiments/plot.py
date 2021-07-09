import math
import os
import os.path as osp
import pathlib
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def load_data(exp_paths, x, y):
    data = defaultdict(lambda: [])
    labels = set()
    for exp_path in exp_paths:
        for datum in exp_path.rglob("data.tar"):
            solver_name = datum.parents[1].name
            try:
                datum = torch.load(str(datum))
                num_samples = datum["options"]["num_samples"]
                for key in datum["history"].keys():
                    labels.add(key)
                datum = pd.DataFrame(datum["history"][y])
                if x == "Samples":
                    datum["x"] = datum["x"] * num_samples
                datum = datum.rename(columns={"y": y, "x": x})
                data[solver_name].append(datum)
            except:
                print("Failed to load data from {}".format(solver_name))

    for key, value in data.items():
        data[key] = pd.concat(value)
    print("Available labels: {}".format(labels))
    return data


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("expdirs", type=str, nargs="*")
    parser.add_argument("-x", default="Steps", choices=["Samples", "Steps"])
    parser.add_argument("-y", default="ReturnPolicy")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--xlim", type=float, default=None)
    parser.add_argument("--ylim", nargs="*", default=None)
    parser.add_argument("--log_scale", action="store_true")
    parser.add_argument("--label_order", type=str, nargs="*", default=None)
    args = parser.parse_args()

    exp_path = [pathlib.Path(expdir) for expdir in args.expdirs]
    data = load_data(exp_path, args.x, args.y)

    sns.set(style="darkgrid", font_scale=1.0)
    ax = plt.gca()

    if args.label_order is not None:
        names = list(args.label_order)
    else:
        names = list(data.keys())
    for name in names:
        datum = data[name]
        sns.lineplot(data=datum, x=args.x, y=args.y, ci="sd", ax=ax, label=name)
        xscale = np.max(np.asarray(datum[args.x])) > 5e3
        if xscale:
            # Just some formatting niceness: x-axis scale in scientific notation if max x is large
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    if args.log_scale:
        plt.yscale("log")
    if args.xlim is not None:
        plt.xlim(right=float(args.xlim))
    if args.ylim is not None:
        plt.ylim(bottom=float(args.ylim[0]), top=float(args.ylim[1]))
    if args.title is not None:
        plt.title(args.title)

    path = os.path.join(args.expdirs[0], args.y + ".png")
    plt.savefig(path, bbox_inches="tight") 
    print("Figure saved at: {}".format(path))


if __name__ == "__main__":
    main()

