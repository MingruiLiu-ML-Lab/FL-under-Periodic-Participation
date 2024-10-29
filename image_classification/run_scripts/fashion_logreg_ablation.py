import os
import subprocess
from itertools import product


algs = ["fedavg", "fedprox", "scaffold", "amplified_fedavg", "amplified_scaffold"]
P = 20
I = 30
iters = 60000
num_trials = 3

default_similarity = 0.05
default_workers = 10
similarities = [0.025, 0.1, 0.33, 1.0]
sampled_workers = [5, 15, 20, 25]
combos = (
    [(default_similarity, workers) for workers in sampled_workers] +
    [(similarity, default_workers) for similarity in similarities]
)

alg_settings = {
    "fedavg": {
        "eta": 0.0001,
        "gamma": 1,
        "extras": "",
    },
    "scaffold": {
        "eta": 0.01,
        "gamma": 1,
        "extras": "-correction round ",
    },
    "amplified_fedavg": {
        "eta": 5e-5,
        "gamma": 2,
        "extras": "",
    },
    "amplified_scaffold": {
        "eta": 0.006666666666,
        "gamma": 1.5,
        "extras": "-correction period ",
    },
    "fedprox": {
        "eta": 0.0001,
        "gamma": 1,
        "extras": "-fedprox 10.0 ",
    }
}
NUM_GPUS = 8
LOG_DIR = "logs/fashion_logreg_ablation"

disconnect = 4 * (P // 5)
seeds = ""
for i in range(1, num_trials+1):
    seeds += str(i) + ","
seeds = seeds[:-1]

jobs = 0
processes = []
for similarity, workers in combos:
    current_log = os.path.join(LOG_DIR, f"similarity_{similarity}_S_{workers}")
    if not os.path.isdir(current_log):
        os.makedirs(current_log)
    for i, alg in enumerate(algs):
        eta = alg_settings[alg]["eta"]
        gamma = alg_settings[alg]["gamma"]
        cmd = (
            "python3 main.py -data fashion -availability periodic "
            + f"-lr-warmup 0.01 -iters-warmup 0 -iters-total {iters} -model linear "
            + f"-seeds {seeds} -iters-checkpoint {iters} "
            + f"-disconnect {disconnect} -iters-per-round {I} "
            + f"-lr {eta} -lr-global {gamma} "
            + alg_settings[alg]["extras"]
            + f"-similarity {similarity} -sampled-workers {workers} "
            + f"-out {current_log}/{alg}.csv -cuda-device {jobs}"
        )
        processes.append(subprocess.Popen(cmd.split(" "), stdout=subprocess.DEVNULL))
        print(cmd + "\n")

        jobs += 1
        if jobs == NUM_GPUS:
            for p in processes:
                p.wait()
            jobs = 0
