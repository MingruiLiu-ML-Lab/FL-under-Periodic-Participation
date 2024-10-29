import torch
from torch.optim import _functional as F
from torch.optim.optimizer import Optimizer, required


class CorrectedSGD(Optimizer):
    r"""Implements stochastic gradient descent with corrections

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """

    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(CorrectedSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CorrectedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, correction, node, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if len(self.param_groups) > 1:
            raise NotImplementedError

        for group in self.param_groups:
            lr = group['lr']
            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    c_i = correction.local_correction[node][i]
                    c = correction.global_correction[i]
                    d_p = p.grad - c_i + c
                    p.add_(d_p, alpha=-lr)

        return loss
