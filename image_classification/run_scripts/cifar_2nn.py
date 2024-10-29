import subprocess
from itertools import product


algs = ["fedavg", "fedprox", "scaffold", "amplified_fedavg", "amplified_scaffold"]
P = 50
I = 5
iters = 60000
channels = 64
kernel_size = 5
stride = 2
architecture = f"{channels},{kernel_size},{stride}"
num_trials = 5

alg_settings = {
    "fedavg": {
        "eta": 1e-5,
        "gamma": 1,
        "extras": "",
    },
    "scaffold": {
        "eta": 0.0001,
        "gamma": 1,
        "extras": "-correction round ",
    },
    "amplified_fedavg": {
        "eta": 3.333333e-5,
        "gamma": 3,
        "extras": "",
    },
    "amplified_scaffold": {
        "eta": 0.000666666,
        "gamma": 1.5,
        "extras": "-correction period ",
    }
    "fedprox": {
        "eta": 1e-5,
        "gamma": 1,
        "extras": "-fedprox 0.01 ",
    }
}
LOG_DIR = f"logs/cifar_2nn"

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
        "python3 main.py -data cifar -availability periodic "
        + f"-lr-warmup 0.01 -iters-warmup 0 -iters-total {iters} -model 2layer "
        + f"-seeds {seeds} -iters-checkpoint {iters} "
        + f"-disconnect {disconnect} -iters-per-round {I} "
        + f"-lr {eta} -lr-global {gamma} -architecture {architecture} "
        + alg_settings[alg]["extras"]
        + f"-out {LOG_DIR}/{name}.csv -cuda-device {i}"
    )
    processes.append(subprocess.Popen(cmd.split(" "), stdout=subprocess.DEVNULL))

for p in processes:
    p.wait()
