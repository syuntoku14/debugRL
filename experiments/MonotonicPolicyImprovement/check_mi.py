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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("expdir")
    args = parser.parse_args()

    exp_path = pathlib.Path(args.expdir)
    x, y = "Steps", "ReturnPolicy"

    for solver in exp_path.glob("*"):
        if not solver.is_dir():
            continue
        solver_name = solver.name
        try:
            datums = []
            for seed in solver.glob("*"):
                datum = torch.load(str(seed) + "/data.tar")
                datum = pd.DataFrame(datum["history"][y])
                datum = datum.rename(columns={"y": y, "x": x})
                datums.append(datum)
            data = pd.concat(datums)
            returns = data[y].values
            improvements = returns[1:] - returns[:-1]
            print("Minimum improvements of {}: {}".format(solver_name, improvements.min()))
        except:
            print("Failed to load data from {}".format(solver_name))

if __name__ == "__main__":
    main()
