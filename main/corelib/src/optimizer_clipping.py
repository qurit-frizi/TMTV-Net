import torch.nn as nn
import torch
from torch import optim


class ClippingGradientNorm(optim.Optimizer):
    """
    Clips the gradient norm during optimization
    """
    def __init__(self, optimizer_base: optim.Optimizer, max_norm: float = 1.0, norm_type: float = 2.0) -> None:
        """
        Params:
            optimizer_base: the base optimizer
            max_norm: the maximum norm of the concatenated gradients of the optimizer. Note: the gradient is modulated
                by the learning rate
            norm_type: type of the used p-norm. Can be ``'inf'`` for infinity norm

        See:
            :func:`torch.nn.utils.clip_grad_norm_`
        """
        if max_norm < 0.0:
            raise ValueError("Invalid max_norm value: {}".format(max_norm))

        defaults = dict(max_norm=max_norm, norm_type=norm_type)
        defaults = {**defaults, **optimizer_base.defaults}
        self.optimizer_base = optimizer_base
        self.max_norm = max_norm
        self.norm_type = norm_type

        params = []
        for group in optimizer_base.param_groups:
            params += group['params']

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        params = []
        for group in self.optimizer_base.param_groups:
            params += group['params']

        nn.utils.clip_grad_norm_(
            params, 
            max_norm=self.max_norm, 
            norm_type=self.norm_type
        )
        return self.optimizer_base.step(closure)
        