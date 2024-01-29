import collections

import functools
from .transforms import CriteriaFn
from typing import Optional, Callable, Union

from basic_typing import ShapeCX, TensorNCX
from transforms import transforms
from transforms import cutout_function
from transforms.copy import copy
import torch


def _transform_random_cutout(feature_names, batch, cutout_size, cutout_value_fn, probability):
    # make sure we do NOT modify the original images
    assert len(feature_names) == 1, 'joint CUTOUT is not yet implemented!'  # TODO implement joint cutout

    feature_value = copy(batch[feature_names[0]])

    apply_cutout = torch.rand(len(feature_value)) <= probability
    for with_cutout, sample in zip(apply_cutout, feature_value):
        if with_cutout:
            cutout_function.cutout(sample, cutout_size=cutout_size, cutout_value_fn=cutout_value_fn)
    transformed_arrays = [feature_value]

    new_batch = collections.OrderedDict(zip(feature_names, transformed_arrays))
    for feature_name, feature_value in batch.items():
        if feature_name not in feature_names:
            # not in the transformed features, so copy the original value
            new_batch[feature_name] = feature_value
    return new_batch


class TransformRandomCutout(transforms.TransformBatchWithCriteria):
    """
    Randomly flip the axis of selected features
    """
    def __init__(
            self,
            cutout_size: Union[ShapeCX, Callable[[], ShapeCX]],
            criteria_fn: Optional[CriteriaFn] = None,
            probability: float = 1.0,
            cutout_value_fn: Callable[[TensorNCX], None] = functools.partial(
                cutout_function.cutout_value_fn_constant,
                value=0)
    ):
        """
        Args:
            cutout_size: the size of the regions to occlude or a callable function with no argument returning
                a tuple representing the size of the region to occlude (with ``N`` dimension removed)
            cutout_value_fn: a function to fill the cutout images. Should directly modify the image
            criteria_fn: how to select the features to transform. If `None` transform all arrays with dim >= 3
            probability: the probability of cutout
        """
        assert isinstance(cutout_size, tuple) or callable(cutout_size), 'must be a tuple or a callable!'
        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_4_or_above

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=functools.partial(
                _transform_random_cutout,
                cutout_size=cutout_size,
                probability=probability,
                cutout_value_fn=cutout_value_fn))
