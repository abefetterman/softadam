import math
import torch
from torch.optim.optimizer import Optimizer
import numpy as np

class SoftAdam(Optimizer):

    def __init__(self, params, lr=1e-1, betas=(0.9, 0.999), eps=1e-8,
                 eta=1.0, weight_decay=0, nesterov=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, eta=eta,
                        weight_decay=weight_decay,
                        nesterov=nesterov)
        super(SoftAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SoftAdam does not support sparse gradients')
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                beta2_hat = min(beta2, 1.0 - 1.0/(state['step']))
                r_beta = (1-beta2) / (1-beta2_hat)
                eta_hat2 = group['eta']*group['eta'] * r_beta

                # Decay the first and second moment with the running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2_hat).addcmul_(1 - beta2_hat, grad, grad)

                # Create temporary tensor for the denominator
                state['adam_lr_factor'] = torch.mean(exp_avg_sq)
                denom = exp_avg_sq.mul(eta_hat2/(torch.mean(exp_avg_sq) + group['eps']*group['eps']))
                denom.sqrt_().add_(1 + group['eta'] - np.sqrt(eta_hat2))

                wd = group['weight_decay']*group['lr']

                p.data.add_(-wd, p.data)

                lr_eff = group['lr']*(1 + group['eta'])

                if nesterov:
                    p.data.addcdiv_(-lr_eff*beta1, exp_avg, denom)
                    p.data.addcdiv_(-lr_eff*(1-beta1), grad, denom)
                else:
                    p.data.addcdiv_(-lr_eff, exp_avg, denom)

        return loss
