import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import METRICS, read_results


ALGS = ["fedavg", "scaffold", "amplified_fedavg", "amplified_scaffold", "fedprox"]
LINE_WIDTH = 2.5
YLIM_WINDOW_START = 0.25

plt.rcParams["font.size"] = 18
plt.rcParams["font.family"] = "sans-serif"


def get_alg_from_name(run_name):
    under_pos = []
    start = 0
    while run_name.find("_", start) != -1:
        under_pos.append(run_name.find("_", start))
        start = under_pos[-1] + 1
    start = under_pos[0]
    end = under_pos[-6]
    alg = run_name[start+1:end]
    assert alg in ALGS
    return alg


def plot(output_path, log_paths, display_names=None):

    # Read results from each run.
    results = {}
    rounds = None
    for i, log_path in enumerate(log_paths):
        run_name = os.path.basename(log_path)
        current_rounds, results[run_name] = read_results(log_path)
        if rounds is None:
            rounds = list(current_rounds)
        assert rounds == current_rounds

    # Plot results.
    fig, axs = plt.subplots(1, len(METRICS))
    fig.set_size_inches(6 * len(METRICS), 5)
    for j, metric in enumerate(METRICS):
        ax = axs[j]

        y_min = None
        y_max = None
        for run_name in results:
            if display_names is None:
                label = run_name
            else:
                label = display_names[get_alg_from_name(run_name)]
            if j != 0:
                label = None
            ax.plot(rounds, results[run_name][metric], label=label, linewidth=LINE_WIDTH)

            start = round(len(rounds) * YLIM_WINDOW_START)
            current_min = float(np.min(results[run_name][metric][start:]))
            current_max = float(np.max(results[run_name][metric][start:]))
            if y_min is None:
                y_min = current_min if current_min < float('inf') else y_min
                y_max = current_max if current_max < float('inf') else y_max
            else:
                y_min = min(y_min, current_min) if current_min < float('inf') else y_min
                y_max = max(y_max, current_max) if current_max < float('inf') else y_max

        ax.set_xlabel("Rounds")
        ylabel = metric if display_names is None else display_names[metric]
        ax.set_ylabel(ylabel)
        y_min -= (y_max - y_min) * 0.05
        y_max += (y_max - y_min) * 0.05
        ax.set_ylim([y_min, y_max])

    plt.figlegend(loc="lower center", ncol=len(results), bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)
    plot(**config)
