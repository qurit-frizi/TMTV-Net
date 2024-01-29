from typing import List, Optional
import torch
from functools import partial
from .transforms import CriteriaFn, criteria_is_tensor, TransformBatchWithCriteria
from basic_typing import Batch


def move_to_device(feature_names: List[str], batch: Batch, device: torch.device, non_blocking: bool) -> Batch:
    new_batch = {name: value for name, value in batch.items()}
    for name in feature_names:
        new_batch[name] = batch[name].to(device, non_blocking=non_blocking)
    
    return new_batch


class TransformMoveToDevice(TransformBatchWithCriteria):
    """
    Move a tensor to a specified device.

    Transfert from CPU to GPU can can't significant time. This transfer time
    can be masked by transferring the data as part of the data preprocessing
    on a single GPU system.

    Note:
        This requires to start torch using `torch.multiprocessing.set_start_method('spawn')`

    Only :class:`torch.Tensor` types will be considered
    """
    def __init__(
            self, 
            device: torch.device,
            non_blocking: bool = False,
            criteria_fn: Optional[CriteriaFn] = None):
        """

        Args:
            feature_names:
            cast_type: must be one of `float`, `long`, `byte`
            non_blocking: wether the data transfer to the device
                is blocking or not
        """
        if criteria_fn is None:
            criteria_fn = criteria_is_tensor

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=partial(move_to_device, device=device, non_blocking=non_blocking)
        )
