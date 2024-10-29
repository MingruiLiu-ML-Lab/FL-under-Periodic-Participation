import sys
import os
import glob
import csv
import subprocess
from itertools import product

import numpy as np
import matplotlib.pyplot as plt


algs = ["fedavg", "fedprox", "scaffold", "amplified_fedavg", "amplified_scaffold"]
metrics = ["train_loss", "test_acc"]
params = ["similarity", "workers"]
defaults = {
    "similarity": 0.05,
    "workers": 10,
}
ranges = {
    "similarity": [0.025, 0.1, 0.33, 1.0],
    "workers": [5, 15, 20, 25],
}
param_combos = {}
combos = []
for i, param in enumerate(params):
    param_vals = []
    for j, other_param in enumerate(params):
        if j == i:
            param_vals.append(ranges[other_param])
        else:
            param_vals.append([defaults[other_param]])
    param_combos[param] = list(product(*param_vals))
    combos += param_combos[param]

DISPLAY_NAMES = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "scaffold": "SCAFFOLD",
    "amplified_fedavg": "Amplified FedAvg",
    "amplified_scaffold": "Amplified SCAFFOLD",
    "train_loss": "Train Loss",
    "test_acc": "Test Accuracy",
    "similarity": "Data Similarity",
    "workers": "Participating Clients"
}

WINDOW = 0.1
SELECT = 0.9

BAR_WIDTH = 0.15
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "sans-serif"

log_dir = sys.argv[1]


def criteria(losses, reverse=False):
    start = min(round(len(losses) * (1 - WINDOW)), len(losses) - 1)
    window_losses = losses[start:]
    pivot = min(round(len(window_losses) * SELECT), len(window_losses) - 1)
    sorted_losses = sorted(window_losses, reverse=reverse)
    return sorted_losses[pivot]

    
# Read results.
results = {}
for (similarity, workers) in combos:
    results[(similarity, workers)] = {}
    for alg in algs:
        log_subdir = f"similarity_{similarity}_S_{workers}"
        result_prefix = os.path.join(log_dir, log_subdir, f"{alg}*.csv")
        result_names = glob.glob(result_prefix)
        assert len(result_names) == 1
        result_name = result_names[0]

        train_loss = []
        test_acc = []
        with open(result_name, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for i, x in enumerate(reader):
                if i == 0:
                    continue
                train_loss.append(float(x[2]))
                test_acc.append(float(x[4]))

        final_loss = criteria(train_loss, reverse=False)
        final_acc = criteria(test_acc, reverse=True)
        results[(similarity, workers)][alg] = {
            "train_loss": final_loss,
            "test_acc": final_acc,
        }

# Plot results.
fig, axs = plt.subplots(nrows=len(params), ncols=len(metrics))
fig.set_size_inches(7, 5.25)
for i, param in enumerate(params):
    for j, metric in enumerate(metrics):
        ax = axs[i, j]

        total_ys = []
        for k, alg in enumerate(algs):
            xs = np.arange(len(param_combos[param])) + BAR_WIDTH * (k-(len(algs)//2))
            ys = [results[combo][alg][metric] for combo in param_combos[param]]
            kwargs = {"label": DISPLAY_NAMES[alg]} if i == 0 and j == 0 else {}
            rects = ax.bar(xs, ys, BAR_WIDTH, **kwargs)
            total_ys += ys

        y_min = np.min(total_ys)
        y_max = np.max(total_ys)
        y_bot = y_min - (y_max - y_min) * 0.2
        y_top = y_max + (y_max - y_min) * 0.2
        ax.set_ylim([y_bot, y_top])

        ax.set_xlabel(DISPLAY_NAMES[param])
        ax.set_ylabel(DISPLAY_NAMES[metric])
        ax.set_xticks(np.arange(len(param_combos[param])), ranges[param])

plt.figlegend(loc="lower center", ncol=len(results), bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "ablation.eps"), bbox_inches="tight")
