import copy

from typing import Optional, Sequence

import numpy as np
import torch
from functools import partial
from .transforms import CriteriaFn, TransformBatchWithCriteria, criteria_is_array_4_or_above
from basic_typing import Batch


def unsqueeze_fn(feature_names: Sequence[str], batch: Batch, axis: int) -> Batch:
    # soft copy so that we don't modify the original batch values
    batch_copy = copy.copy(batch)
    for name in feature_names:
        t = batch_copy[name]
        if isinstance(t, np.ndarray):
            new_shape = list(t.shape[:axis]) + [1] + list(t.shape[axis:])
            batch_copy[name] = np.reshape(t, new_shape)
        elif isinstance(t, torch.Tensor):
            batch_copy[name] = torch.unsqueeze(t, axis)

    return batch_copy


class TransformUnsqueeze(TransformBatchWithCriteria):
    """
    Unsqueeze a dimension of a tensor.

    Only :class:`numpy.ndarray` and :class:`torch.Tensor` types will be transformed
    """
    def __init__(self, axis: int, criteria_fn: Optional[CriteriaFn] = criteria_is_array_4_or_above):
        """

        Args:
            axis: create a new dimension of size 1 for axis `axis`
            criteria_fn: specify what features needs to be transformed
        """
        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=partial(unsqueeze_fn, axis=axis)
        )
