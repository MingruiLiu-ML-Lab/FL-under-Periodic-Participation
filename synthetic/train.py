import os
import argparse
import json
from math import sqrt, floor
from typing import Tuple, Optional

import numpy as np


DIM = 4


def log(output_path, msg):
    with open(output_path, "a+") as f:
        f.write(msg + "\n")


def obj(x: np.ndarray, mu, H, L, lamb, c, best):
    x1, x2, x3, x4 = x
    pos_x3 = max(0, x3)

    F = 0
    F += 0.5 * mu * (x1 - c) ** 2
    F += 0.5 * H * (x2 - best[1]) ** 2
    F += 0.125 * H * (x3 ** 2 + pos_x3 ** 2)
    F += 0.25 * (L + lamb) * x4 ** 2
    return F


def stochastic_grad(x: np.ndarray, client: int, mu, H, L, lamb, c, sigma, zeta, best):
    x1, x2, x3, x4 = x
    pos_x3 = max(0, x3)

    noise = np.random.normal(scale=sigma)
    if client % 2 == 0:
        fourth_coeff = L
        hetero_sign = 1
    else:
        fourth_coeff = lamb
        hetero_sign = -1
    G = np.array([
        mu * (x1 - c),
        H * (x2 - best[1]),
        0.25 * H * (x3 + pos_x3) + noise,
        0.5 * fourth_coeff * x4 + hetero_sign * zeta,
    ])
    return G


def grad(x: np.ndarray, mu, H, L, lamb, c, best):
    x1, x2, x3, x4 = x
    pos_x3 = max(0, x3)

    G = np.array([
        mu * (x1 - c),
        H * (x2 - best[1]),
        0.25 * H * (x3 + pos_x3),
        0.5 * (L + lamb) * x4,
    ])
    return G


def train(
    M=10,
    H=16,
    lamb=1,
    zeta=10,
    sigma=1,
    c=1,
    mu=1,
    L=2,
    rounds=5000,
    S=2,
    lr=0.0001,
    I=5,
    sampling="uniform",
    P=10,
    client_groups=2,
    amplify=True,
    gamma=1.5,
    correction="period",
    fedprox_mu=None,
    seed=0,
    print_freq=100,
    output_path="results.csv",
):

    assert lamb <= mu <= H / 16
    assert correction in [None, "round", "period"]
    assert sampling in ["uniform", "cyclic"]
    assert P % client_groups == 0
    assert M % client_groups == 0
    assert S <= (M // client_groups)

    if os.path.isfile(output_path):
        os.remove(output_path)

    np.random.seed(seed)
    eta = lr / gamma
    best = np.array([c, mu ** (1/2) * c / (H ** (1/2)), 0, 0])

    global_x = np.zeros(DIM)
    amplify_x = global_x.copy()
    local_corrections = [np.zeros_like(global_x) for _ in range(M)]
    global_correction = np.zeros_like(global_x)
    avg_local_grads = [np.zeros_like(global_x) for _ in range(M)]
    local_sample_sizes = [0 for _ in range(M)]

    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    log(output_path, "round,obj_val,sol_dist,x1,x2,x3,x4")
    for r in range(rounds):

        if sampling == "uniform":
            sampled_clients = np.random.choice(M, size=S, replace=False)
        elif sampling == "cyclic":
            rounds_within_period = r - P * floor(r / P)
            switch = P / client_groups
            group = rounds_within_period // switch
            available_clients = [i for i in range(M) if i % client_groups == group]
            sampled_clients = np.random.choice(available_clients, size=S, replace=False)

        trained_models = []
        for client in sampled_clients:

            local_x = global_x.copy()
            for i in range(I):
                g = stochastic_grad(local_x, client, mu, H, L, lamb, c, sigma, zeta, best)
                if correction is not None:
                    avg_local_grads[client] += g
                    local_sample_sizes[client] += 1
                    g += - local_corrections[client] + global_correction
                if fedprox_mu is not None:
                    g += fedprox_mu * (local_x - global_x)
                local_x = local_x - eta * g

            trained_models.append(local_x)

        trained_models = np.array(trained_models)
        global_x = np.mean(trained_models, axis=0)

        if correction == "round":
            for client in sampled_clients:
                global_correction -= local_corrections[client] / M
                local_corrections[client] = avg_local_grads[client] / local_sample_sizes[client]
                global_correction += local_corrections[client] / M
                avg_local_grads[client] = np.zeros_like(global_x)
                local_sample_sizes[client] = 0

        if amplify and (r+1) % P == 0:
            global_x = amplify_x + gamma * (global_x - amplify_x)
            amplify_x = global_x.copy()

            if correction == "period":
                for client in range(M):
                    if local_sample_sizes[client] > 0:
                        global_correction -= local_corrections[client] / M
                        local_corrections[client] = avg_local_grads[client] / local_sample_sizes[client]
                        global_correction += local_corrections[client] / M
                        avg_local_grads[client] = np.zeros_like(global_x)
                        local_sample_sizes[client] = 0

        if (r+1) % print_freq == 0:
            sol_dist = sqrt(float(np.sum((global_x - best) ** 2)))
            obj_val = obj(global_x, mu, H, L, lamb, c, best)
            x1, x2, x3, x4 = global_x
            log(output_path, f"{r},{obj_val},{sol_dist},{x1},{x2},{x3},{x4}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)
    train(**config)
