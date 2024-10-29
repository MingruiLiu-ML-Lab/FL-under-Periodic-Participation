## Image Classification

The image classification experiments from the paper can be reproduced with the scripts in `run_scripts/`.

FashionMNIST:
```
python3 run_scripts/fashion_logreg.py
python3 plot.py fashion logs/fashion_logreg/*.csv --local_steps 30
```

CIFAR-10:
```
python3 run_scripts/cifar_2nn.py
python3 plot.py cifar logs/cifar_2nn/*.csv --local_steps 5
```

FashionMNIST ablation study:
```
python3 run_scripts/fashion_logreg_ablation.py
python3 plot_ablation.py logs/fashion_logreg_ablation
```

Note that the scripts `fashion_logreg.py` and `cifar_2nn.py` are expecting a node with 5 available GPUs. This can be modified by changing the `-cuda-device` argument passed to `main.py` on Line 58 of `fashion_logreg.py` and/or Line 62 of `cifar_2nn.py`. Similarly, the script `fashion_logreg_ablation.py` can be adapted by changing the `NUM_GPUS` variable on Line 48.

The code was run successfully in the following environment: Python 3.9.13, PyTorch 1.12.1, Torchvision 0.13.1

This code is based on the following public repository: https://github.com/IBM/fl-arbitrary-participation
