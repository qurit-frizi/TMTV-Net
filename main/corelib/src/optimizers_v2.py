from functools import partial
import torch
from torch import nn
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import warnings
from basic_typing import Datasets
from utils.requires import torch_requires
from optimizer_clipping import ClippingGradientNorm


SchedulerType = Any
StepSchedulerType = Any


class CosineAnnealingWarmRestartsDecayed(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    """
    Scheduler based on :class:`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`. In addition,
    every time the learning rate is restarted, the base learning rate is decayed by `decay_factor`
    """
    def __init__(self, optimizer: torch.optim.Optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0, last_epoch: int = -1, decay_factor: float = 0.7) -> None:
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        self.decay_factor = decay_factor

    def step(self, epoch=None):
        # decay the base learning rate at the last epoch of the cycle
        if self.T_i == self.T_cur + 1:
            for i in range(len(self.base_lrs)):
                self.base_lrs[i] *= self.decay_factor

        # resume the processing 
        super().step(epoch=epoch)


class Optimizer:
    def __init__(
            self, 
            optimizer_fn: Callable[[Iterator[nn.parameter.Parameter]], torch.optim.Optimizer],
            scheduler_fn: Optional[Callable[[torch.optim.Optimizer], SchedulerType]] = None,
            step_scheduler_fn: Optional[Callable[[torch.optim.Optimizer], StepSchedulerType]] = None):
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.per_step_scheduler_fn = step_scheduler_fn
        self.clipping_fn = None

    def set_scheduler_fn(self, scheduler_fn: Optional[Callable[[torch.optim.Optimizer], SchedulerType]]):
        self.scheduler_fn = scheduler_fn

    def set_step_scheduler_fn(self, step_scheduler_fn: Optional[Callable[[torch.optim.Optimizer], StepSchedulerType]]):
        self.per_step_scheduler_fn = step_scheduler_fn

    def __call__(self, datasets: Datasets, model: nn.Module) -> Tuple[Dict[str, torch.optim.Optimizer], Optional[Dict[str, SchedulerType]], Optional[Dict[str, StepSchedulerType]]]:
        per_step_schedulers = None
        schedulers = None
        if self.scheduler_fn is not None:
            schedulers = {}
        if self.per_step_scheduler_fn is not None:
            per_step_schedulers = {}

        optimizers = {}
        for dataset_name in datasets.keys():
            if isinstance(model, torch.nn.ModuleDict):
                # this is a collection of model. Assumed we have a different model
                # per dataset to be optimized
                sub_model = model[dataset_name]
                optimizer = self.optimizer_fn(sub_model.parameters())
            else:
                optimizer = self.optimizer_fn(model.parameters())

            if self.clipping_fn is not None:
                # apply gradient clipping
                optimizer = self.clipping_fn(optimizer)

            optimizers[dataset_name] = optimizer

            if self.scheduler_fn is not None and optimizer is not None:
                scheduler = self.scheduler_fn(optimizer)
                schedulers[dataset_name] = scheduler

            if self.per_step_scheduler_fn is not None and optimizer is not None:
                per_step_scheduler = self.per_step_scheduler_fn(optimizer)
                per_step_schedulers[dataset_name] = per_step_scheduler

        return optimizers, schedulers, per_step_schedulers

    def scheduler_step_lr(self, step_size: int, gamma: float = 0.1) -> 'Optimizer':
        """
        Apply a scheduler on the learning rate.

        Decays the learning rate of each parameter group by gamma every
        step_size epochs.
        """
        scheduler_fn = partial(torch.optim.lr_scheduler.StepLR, step_size=step_size, gamma=gamma)
        self.set_scheduler_fn(scheduler_fn)
        return self

    def scheduler_cosine_annealing_warm_restart(self, T_0: int, T_mult: int = 1, eta_min: float = 0, last_epoch=-1) -> 'Optimizer':
        """
        Apply a scheduler on the learning rate.

        Restart the learning rate every T_0 * (T_mult)^(#restart) epochs.
        
        References:
            https://arxiv.org/pdf/1608.03983v5.pdf

        """
        scheduler_fn = partial(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch)
        self.set_scheduler_fn(scheduler_fn)
        return self

    def scheduler_cosine_annealing_warm_restart_decayed(self, T_0: int, T_mult: int = 1, eta_min: float = 0, last_epoch=-1, decay_factor=0.7) -> 'Optimizer':
        """
        Apply a scheduler on the learning rate. Each time the learning rate is restarted, the base learning rate is decayed 

        Restart the learning rate every T_0 * (T_mult)^(#restart) epochs.
        
        References:
            https://arxiv.org/pdf/1608.03983v5.pdf

        """
        scheduler_fn = partial(CosineAnnealingWarmRestartsDecayed, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch, decay_factor=decay_factor)
        self.set_scheduler_fn(scheduler_fn)
        return self

    @torch_requires(min_version='1.3')
    def scheduler_one_cycle(
            self,
            max_learning_rate: float, 
            epochs: int, 
            steps_per_epoch: int,
            learning_rate_start_div_factor: float = 25.0,
            learning_rate_end_div_factor: float = 10000.0,
            percentage_cycle_increase: float = 0.3,
            anneal_strategy: str = 'cos', 
            cycle_momentum: bool = True, 
            base_momentum: float = 0.85, 
            max_momentum: float = 0.95):
        """
        This scheduler should not be used with another scheduler!

        The learning rate or momentum provided by the Optimizer will
        be overriden by this scheduler.
        """
        assert self.scheduler_fn is None, 'this scheduler cannot be chained!'
        step_scheduler_fn = partial(torch.optim.lr_scheduler.OneCycleLR,
            max_lr=max_learning_rate,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=percentage_cycle_increase,
            anneal_strategy=anneal_strategy,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=learning_rate_start_div_factor,
            final_div_factor=learning_rate_end_div_factor,
            cycle_momentum=cycle_momentum
        )
        
        self.set_step_scheduler_fn(step_scheduler_fn)
        return self

    def clip_gradient_norm(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        Clips the gradient norm during optimization

        Args:
            max_norm: the maximum norm of the concatenated gradients of the optimizer. Note: the gradient is modulated
                by the learning rate
            norm_type: type of the used p-norm. Can be ``'inf'`` for infinity norm

        See:
            :func:`torch.nn.utils.clip_grad_norm_`
        """
        if self.clipping_fn is not None:
            warnings.warn('`self.clipping_fn` is already set and will be replaced!')

        self.clipping_fn = partial(ClippingGradientNorm, max_norm=max_norm, norm_type=norm_type)
        return self

class OptimizerSGD(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.9, weight_decay: float = 0, nesterov: bool = False):
        super().__init__(optimizer_fn=partial(torch.optim.SGD, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov))

class OptimizerAdam(Optimizer):
    def __init__(self, learning_rate: float, weight_decay: float = 0, betas: Tuple[float, float] = (0.9, 0.999), eps: float=1e-8):
        super().__init__(optimizer_fn=partial(torch.optim.Adam, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps))

class OptimizerAdamW(Optimizer):
    def __init__(self, learning_rate: float, weight_decay: float = 1e-2, betas: Tuple[float, float] = (0.9, 0.999), eps: float=1e-8):
        super().__init__(optimizer_fn=partial(torch.optim.AdamW, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps))