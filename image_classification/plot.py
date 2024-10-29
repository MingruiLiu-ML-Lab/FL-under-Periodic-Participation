""" Plot training results. """

import os
import glob
import json
import argparse
import csv
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


ALGS = ["fedavg", "fedprox", "scaffold", "amplified_fedavg", "amplified_scaffold"]
LINE_WIDTH = 2.5
METRICS = ["training_loss", "test_accuracy"]
DISPLAY_NAMES = {
    "fedavg": "FedAvg",
    "scaffold": "SCAFFOLD",
    "amplified_fedavg": "Amplified FedAvg",
    "amplified_scaffold": "Amplified SCAFFOLD",
    "fedprox": "FedProx",
    "training_loss": "Train Loss",
    "test_accuracy": "Test Accuracy",
}

YLIM_WINDOW_START = 0.25
SUBSAMPLE = 1.0
SMOOTH_PROP = 0.01
AVG_WINDOW_START = 0.9


plt.rcParams["font.size"] = 18
plt.rcParams["font.family"] = "sans-serif"


def moving_average(x, window):
    assert window % 2 == 1
    x_avg = np.convolve(
        x,
        np.ones(window) / window,
        'valid',
    )
    cutoff = (window - 1) // 2
    beginning = np.zeros((cutoff))
    end = np.zeros((cutoff))
    for i in range(cutoff):
        beginning[i] = np.mean(x[:i + cutoff])
        end[-i] = np.mean(x[-(i + cutoff):])
    total_avg = np.concatenate([beginning, x_avg, end])

    assert total_avg.shape == x.shape
    return total_avg


def get_alg_from_name(run_name):
    if run_name in ALGS:
        return run_name

    under_pos = []
    start = 0
    while run_name.find("_", start) != -1:
        under_pos.append(run_name.find("_", start))
        start = under_pos[-1] + 1
    end = under_pos[-2]
    alg = run_name[:end]
    assert alg in ALGS
    return alg


def plot(plot_name, log_files, local_steps=5):

    # Read results of each run (averaged over workers).
    results = {}
    n_iterations = None
    for log_file in log_files:

        # Create ID for run.
        run_name = os.path.basename(log_file)
        dot_pos = run_name.rfind(".")
        run_name = run_name[:dot_pos]
        results[run_name] = {}
        for j, metric in enumerate(METRICS):
            results[run_name][metric] = {}

        # Get raw CSV data, confirm metric names and values.
        metric_pos = {}
        with open(log_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = []
            for i, row in enumerate(reader):
                if i == 0:
                    metric_names = list(row[2:])
                    for metric_name in METRICS:
                        assert metric_name in METRICS
                    for j, metric_name in enumerate(metric_names):
                        if metric_name in METRICS:
                            metric_pos[metric_name] = j + 2
                else:
                    rows.append(list(row))

        # Get seeds.
        seeds = []
        for i, row in enumerate(rows):
            seed = int(row[0])
            if seed not in seeds:
                seeds.append(seed)
        n_seeds = len(seeds)

        # Make sure iteration numbers from each seed align.
        iterations = [int(row[1]) for row in rows if int(row[0]) == seeds[0]]
        for seed in seeds[1:]:
            seed_iterations = [int(row[1]) for row in rows if int(row[0]) == seed]
            assert iterations == seed_iterations
        if n_iterations is None:
            n_iterations = len(iterations)
        else:
            assert n_iterations == len(iterations)

        # Aggregate results by iteration/seed.
        for j, metric in enumerate(METRICS):
            results[run_name][metric] = np.zeros((n_iterations, n_seeds))
        for row in rows:
            seed = int(row[0])
            iteration = int(row[1])
            seed_idx = seeds.index(seed)
            iteration_idx = iterations.index(iteration)
            for metric in METRICS:
                pos = metric_pos[metric]
                results[run_name][metric][iteration_idx, seed_idx] = float(row[pos])

    # Smooth results over time.
    smooth_window = 2 * round((n_iterations * SMOOTH_PROP - 1) / 2) + 1
    for run_name in results:
        for metric in METRICS:
            for i in range(n_seeds):
                results[run_name][metric][:, i] = moving_average(
                    results[run_name][metric][:, i],
                    smooth_window,
                )

    # Subsample points for plotting.
    sampled = [random.random() < SUBSAMPLE for _ in range(n_iterations)]
    iterations = [x for i, x in enumerate(iterations) if sampled[i]]
    for run_name in results:
        for metric in METRICS:
            results[run_name][metric] = [x for i, x in enumerate(results[run_name][metric]) if sampled[i]]
    if SUBSAMPLE < 1.0:
        print(f"Total points: {n_iterations}")
        print(f"Subsampled points: {len(iterations)}")
    n_iterations = len(iterations)
    rounds = [round(i / local_steps) for i in iterations]

    # Compute mean and std over seeds.
    mean_results = {}
    std_results = {}
    for run_name in results:
        mean_results[run_name] = {}
        std_results[run_name] = {}
        for metric in METRICS:
            mean_results[run_name][metric] = np.mean(results[run_name][metric], axis=1)
            std_results[run_name][metric] = np.std(results[run_name][metric], axis=1)

    # Plot results.
    fig, axs = plt.subplots(1, len(METRICS))
    fig.set_size_inches(6 * len(METRICS), 5)
    for j, metric in enumerate(METRICS):
        ax = axs[j]

        y_min = None
        y_max = None
        for run_name in results:

            label = get_alg_from_name(run_name)
            if label in DISPLAY_NAMES:
                label = DISPLAY_NAMES[label]
            if j != 0:
                label = None
            ys = mean_results[run_name][metric]
            ub = ys + std_results[run_name][metric]
            lb = ys - std_results[run_name][metric]
            ax.fill_between(rounds, y1=ub, y2=lb, alpha=0.25)
            ax.plot(rounds, ys, label=label, linewidth=LINE_WIDTH)

            start = round(n_iterations * YLIM_WINDOW_START)
            current_min = float(np.min(mean_results[run_name][metric][start:]))
            current_max = float(np.max(mean_results[run_name][metric][start:]))
            if y_min is None:
                y_min = current_min if current_min < float('inf') else y_min
                y_max = current_max if current_max < float('inf') else y_max
            else:
                y_min = min(y_min, current_min) if current_min < float('inf') else y_min
                y_max = max(y_max, current_max) if current_max < float('inf') else y_max

        ax.set_xlabel("Rounds")
        ylabel = DISPLAY_NAMES[metric] if metric in DISPLAY_NAMES else metric
        ax.set_ylabel(ylabel)
        y_min -= (y_max - y_min) * 0.05
        y_max += (y_max - y_min) * 0.05
        ax.set_ylim([y_min, y_max])

    plt.figlegend(loc="lower center", ncol=len(results), bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.savefig(f"{plot_name}.eps", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name", type=str, default="plot", help="Name of outgoing plot file"
    )
    parser.add_argument(
        "log_files", nargs="+", help="CSV files with results for individual runs to compare",
    )
    parser.add_argument(
        "--local_steps", type=int, default=5
    )
    args = parser.parse_args()
    plot(args.name, args.log_files, args.local_steps)
