import subprocess
from itertools import product


algs = ["fedavg", "fedprox", "scaffold", "amplified_fedavg", "amplified_scaffold"]
P = 20
I = 30
iters = 60000
num_trials = 5

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
LOG_DIR = "logs/fashion_logreg"

disconnect = 4 * (P // 5)
seeds = ""
for i in range(1, num_trials+1):
    seeds += str(i) + ","
seeds = seeds[:-1]

processes = []
for i, alg in enumerate(algs):
    eta = alg_settings[alg]["eta"]
    gamma = alg_settings[alg]["gamma"]
    name = f"{alg}_{eta}_{gamma}"
    cmd = (
        "python3 main.py -data fashion -availability periodic "
        + f"-lr-warmup 0.01 -iters-warmup 0 -iters-total {iters} -model linear "
        + f"-seeds {seeds} -iters-checkpoint {iters} "
        + f"-disconnect {disconnect} -iters-per-round {I} "
        + f"-lr {eta} -lr-global {gamma} "
        + alg_settings[alg]["extras"]
        + f"-out {LOG_DIR}/{name}.csv -cuda-device {i}"
    )
    processes.append(subprocess.Popen(cmd.split(" "), stdout=subprocess.DEVNULL))

for p in processes:
    p.wait()
