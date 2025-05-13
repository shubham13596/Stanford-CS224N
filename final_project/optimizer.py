from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            t = 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['mt'] = torch.zeros_like(p.data)
                    state['vt'] = torch.zeros_like(p.data)

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group['eps']
                wt_decay = group["weight_decay"]

                #get state values
                mt, vt = state['mt'], state['vt']
                state['step'] = state['step'] + 1
                t = state['step']
                
                # Update biased first and second moment estimates
                mt = beta1 * mt + (1-beta1) * grad
                vt = beta2 * vt + (1-beta2) * (grad**2)

                # save the state
                state['mt'] = mt
                state['vt'] = vt

                # bias corrected estimates
                mt_hat = mt/(1 - (beta1**t))        
                vt_hat = vt/(1 - (beta2**t))

                #alpha = alpha * (1 - (beta2**t))**0.5/(1 - beta1**t)

                p.data = p.data - alpha * (mt_hat) /(vt_hat**0.5 + eps) - alpha*(wt_decay)*p.data

                state = self.state[p]
                
                ###       Complete the implementation of AdamW here, reading and saving
                ###       your state in the `state` dictionary above.
                ###       The hyperparameters can be read from the `group` dictionary
                ###       (they are lr, betas, eps, weight_decay, as saved in the constructor).
                ###
                ###       To complete this implementation:
                ###       1. Update the first and second moments of the gradients.
                ###       2. Apply bias correction
                ###          (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                ###          also given in the pseudo-c ode in the project description).
                ###       3. Update parameters (p.data).
                ###       4. Apply weight decay after the main gradient-based updates.
            
        return loss
