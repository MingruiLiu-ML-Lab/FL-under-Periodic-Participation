import argparse
import json
from itertools import product

from train import train
from plot import plot
from utils import read_results

WINDOW = 0.1
SELECT = 0.9


def evaluate_results(results_path):
    _, results = read_results(results_path)
    losses = results["obj_val"]
    start = round(len(losses) * (1-WINDOW))
    losses = sorted(losses[start:])
    pivot = round(len(losses) * SELECT)
    return losses[pivot]


def tune(
    default_train_config,
    alg="amplified_fedavg",
    lr_search=[1e-1, 1e-2, 1e-3, 1e-4],
    gamma_search=[1.25, 1.5, 2, 3],
    mu_search=[None],
):

    output_path = default_train_config["output_path"]
    start = output_path.rfind(".")
    base_name = output_path[:start]
    ext = output_path[start:]

    # Run training with each parameter combination.
    results = {}
    paths = {}
    for parameter in product(lr_search, gamma_search, mu_search):
        lr, gamma, mu = parameter
        train_config = dict(default_train_config)
        train_config["lr"] = lr
        train_config["gamma"] = gamma
        train_config["fedprox_mu"] = mu
        train_config["output_path"] = f"{base_name}_lr_{lr}_gamma_{gamma}_mu_{mu}{ext}"

        if alg == "fedavg":
            train_config.update({
                "amplify": False,
                "correction": None,
            })
        elif alg == "scaffold":
            train_config.update({
                "amplify": False,
                "correction": "round",
            })
        elif alg == "amplified_fedavg":
            train_config.update({
                "amplify": True,
                "correction": None,
            })
        elif alg == "amplified_scaffold":
            train_config.update({
                "amplify": True,
                "correction": "period",
            })
        elif alg == "fedprox":
            train_config.update({
                "amplify": False,
                "correction": None,
            })
        else:
            raise NotImplementedError

        results_path = train(**train_config)
        results[parameter] = evaluate_results(results_path)
        paths[parameter] = results_path

    # Find best settings.
    best_parameter = None
    best_result = None
    for parameter, result in results.items():
        if best_result is None or result < best_result:
            best_parameter = parameter
            best_result = result
    best_path = paths[best_parameter]

    return results, best_parameter, best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        tune_config = json.load(f)
    results, best_parameter, best_path = tune(**tune_config)

    # Plot results for best setting.
    output_path = tune_config["default_train_config"]["output_path"]
    dot_pos = output_path.rfind(".")
    base_name = output_path[:dot_pos]
    plot(f"{base_name}_tune.png", [best_path])

    # Print results.
    for parameter, result in results.items():
        print(parameter)
        print(result)
        print("")
