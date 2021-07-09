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
    labels = set()
    for solver in exp_path.glob("*"):
        if not solver.is_dir():
            continue
        solver_name = solver.name
        try:
            datums = []
            for seed in solver.glob("*"):
                datum = torch.load(str(seed) + "/data.tar")
                for key in datum["history"].keys():
                    labels.add(key)
                num_samples = datum["options"]["num_samples"]
                datum = pd.DataFrame(datum["history"][y])
                if x == "Samples":
                    datum["x"] = datum["x"] * num_samples
                datum = datum.rename(columns={"y": y, "x": x})
                datums.append(datum)
            data[solver_name] = pd.concat(datums)
        except:
            print("Failed to load data from {}".format(solver_name))
    print("Available labels: {}".format(labels))
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
    path = os.path.join(args.expdir, args.y + ".png")
    plt.savefig(path, bbox_inches="tight") 
    print("Figure saved at: {}".format(path))


if __name__ == "__main__":
    main()
