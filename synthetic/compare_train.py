import argparse
import json

from train import train
from plot import plot


ALG_SETTINGS = {
    "fedavg": {
        "gamma": 1,
        "amplify": False,
        "correction": None,
        "fedprox_mu": None,
    },
    "scaffold": {
        "gamma": 1,
        "amplify": False,
        "correction": "round",
        "fedprox_mu": None,
    }
    "amplified_fedavg": {
        "amplify": True,
        "correction": None,
        "fedprox_mu": None,
    }
    "amplified_scaffold": {
        "amplify": True,
        "correction": "period",
        "fedprox_mu": None,
    }
    "fedprox": {
        "gamma": 1,
        "amplify": False,
        "correction": None,
    }
}


def compare(default_settings):

    output_path = default_settings["output_path"]
    dot_pos = output_path.rfind(".")
    base_name = output_path[:dot_pos]

    # Run training for each algorithm.
    log_paths = []
    for alg, alg_settings in ALG_SETTINGS.items():
        settings = dict(default_settings)
        settings.update(alg_settings)
        settings["output_path"] = f"{base_name}_{alg}.csv"
        train(**settings)
        log_paths.append(settings["output_path"])

    return log_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        default_settings = json.load(f)
    log_paths = compare(default_settings)

    # Plot results.
    output_path = default_settings["output_path"]
    dot_pos = output_path.rfind(".")
    base_name = output_path[:dot_pos]
    plot(f"{base_name}_compare_train.png", log_paths)
