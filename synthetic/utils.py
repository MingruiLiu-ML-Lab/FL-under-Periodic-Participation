import json
import csv


METRICS = ["obj_val", "sol_dist"]


def read_results(results_path):

    results = {metric: [] for metric in METRICS}
    rounds = []
    with open(results_path, "r") as results_file:
        reader = csv.reader(results_file)
        for j, row in enumerate(reader):
            if j == 0:
                continue
            r = row[0]
            obj_val = row[1]
            sol_dist = row[2]

            rounds.append(int(r))
            results["obj_val"].append(float(obj_val))
            results["sol_dist"].append(float(sol_dist))

    return rounds, results
