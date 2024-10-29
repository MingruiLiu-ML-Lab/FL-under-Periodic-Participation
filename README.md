### FL-under-Periodic-Participation

Code for the paper: [Federated Learning under Periodic Client Participation and Heterogeneous Data: A New Communication-Efficient Algorithm and Analysis](https://neurips.cc/virtual/2024/poster/94818), by Michael Crawshaw and Mingrui Liu. Presented at Neurips 2024.

Code for synthetic experiment and image classification experiments can be found in subdirectories `synthetic/` and `image_classifcation/`. See the README in each subdirectory for instructions on running the experiments.

# Abstract
In federated learning, it is common to assume that clients are always available to participate in training, which may not be feasible with user devices in practice. Recent works analyze federated learning under more realistic participation patterns, such as cyclic client availability or arbitrary participation. However, all such works either require strong assumptions (e.g., all clients participate almost surely within a bounded window), do not achieve linear speedup and reduced communication rounds, or are not applicable in the general non-convex setting. In this work, we focus on nonconvex optimization and consider participation patterns in which the chance of participation over a fixed window of rounds is equal among all clients, which includes cyclic client availability as a special case. Under this setting, we propose a new stochastic first-order algorithm, named Amplified SCAFFOLD, and prove that it achieves linear speedup, reduced communication, and resilience to data heterogeneity simultaneously. In particular, for cyclic participation, our algorithm is proved to enjoy $\mathcal{O}(\epsilon^{-2})$ communication rounds to find an $\epsilon$-stationary point in the non-convex stochastic setting. In contrast, the prior work under the same setting requires $\mathcal{O}(\kappa^2 \epsilon^{-4})$ communication rounds, where $\kappa$ denotes the data heterogeneity. Therefore, our algorithm significantly reduces communication rounds due to better dependency in terms of $\epsilon$ and $\kappa$. Our analysis relies on a fine-grained treatment of the nested dependence between client participation and errors in the control variates, which results in tighter guarantees than previous work. We also provide experimental results with (1) synthetic data and (2) real-world data with a large number of clients $(N = 250)$, demonstrating the effectiveness of our algorithm under periodic client participation.

# Citation
```
@inproceedings{
crawshaw2024federated,
title={Federated Learning under Periodic Client Participation and Heterogeneous Data: A New Communication-Efficient Algorithm and Analysis},
author={Michael Crawshaw, Mingrui Liu},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=WftaVkL6G2}
}
```
