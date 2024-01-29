import copy

from typing import Optional, Sequence

import numpy as np
import torch
from functools import partial
from .transforms import CriteriaFn, TransformBatchWithCriteria, criteria_is_array_4_or_above
from basic_typing import Batch


def squeeze_fn(feature_names: Sequence[str], batch: Batch, axis: int) -> Batch:
    # soft copy so that we don't modify the original batch values
    batch_copy = copy.copy(batch)
    for name in feature_names:
        t = batch_copy[name]
        assert t.shape[axis] == 1, f'squeezed axis must have size==1, got={t.shape[axis]}'
        if isinstance(t, np.ndarray):
            batch_copy[name] = np.squeeze(t, axis)
        elif isinstance(t, torch.Tensor):
            batch_copy[name] = torch.squeeze(t, axis)

    return batch_copy


class TransformSqueeze(TransformBatchWithCriteria):
    """
    Squeeze a dimension of a tensor (i.e., remove one dimension of size 1 of a specifed axis)

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
            transform_fn=partial(squeeze_fn, axis=axis)
        )
