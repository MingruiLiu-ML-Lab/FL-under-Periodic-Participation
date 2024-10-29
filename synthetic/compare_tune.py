import argparse
import json

from tune import tune
from plot import plot


ALGS = ["fedavg", "scaffold", "amplified_fedavg", "amplified_scaffold", "fedprox"]


def compare(default_settings):

    output_path = default_settings["default_train_config"]["output_path"]
    dot_pos = output_path.rfind(".")
    base_name = output_path[:dot_pos]

    # Run FedAvg, SCAFFOLD, Amplified FedAvg, Amplified SCAFFOLD.
    best_results = {}
    best_params = {}
    best_log_paths = {}
    for alg in ALGS:
        settings = dict(default_settings)
        settings["alg"] = alg
        settings["default_train_config"]["output_path"] = f"{base_name}_{alg}.csv"
        if alg in ["fedavg", "scaffold", "fedprox"]:
            settings["gamma_search"] = [1]
        if alg not in ["fedprox"]:
            settings["mu_search"] = [None]

        alg_results, alg_best_param, alg_best_path = tune(**settings)
        best_results[alg] = alg_results[alg_best_param]
        best_params[alg] = alg_best_param
        best_log_paths[alg] = alg_best_path

    return best_results, best_params, best_log_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        default_settings = json.load(f)
    output_path = default_settings["default_train_config"]["output_path"]
    dot_pos = output_path.rfind(".")
    base_name = output_path[:dot_pos]
    results, params, log_paths = compare(default_settings)

    # Plot results.
    plot_log_paths = [log_paths[alg] for alg in ALGS]
    plot(f"{base_name}_compare_tune.png", plot_log_paths)

    # Print results.
    for alg, result in results.items():
        alg_param = params[alg]
        print(alg)
        print(alg_param)
        print(result)
        print("")
