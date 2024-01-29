import collections

import functools
from .transforms import CriteriaFn
from typing import Optional

from basic_typing import ShapeX

from transforms import transforms
from transforms.resize import resize


def _transform_resize(feature_names, batch, size, mode):
    new_batch = collections.OrderedDict()
    for feature_name, feature_value in batch.items():
        if feature_name in feature_names:
            assert len(feature_value.shape) == len(size) + 2, \
                'unexpected shape! `size` should not include samples or filter!'
            new_batch[feature_name] = resize(feature_value, size=size, mode=mode)
        else:
            new_batch[feature_name] = feature_value

    return new_batch


class TransformResize(transforms.TransformBatchWithCriteria):
    """
    Resize a tensor to a fixed size
    """
    def __init__(self, size: ShapeX, criteria_fn: Optional[CriteriaFn] = None, mode='linear'):
        """
        Args:
            size: the size to reshape to. Excluding the sample and filter
            criteria_fn: how to select the features to transform. If `None` transform all arrays with dim >= 3
            mode: the resampling method. Can be `linear` or `nearest`
        """
        size = list(size)

        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_4_or_above

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=functools.partial(_transform_resize, size=size, mode=mode))
