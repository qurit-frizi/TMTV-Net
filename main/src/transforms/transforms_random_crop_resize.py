import collections

import functools
from .transforms import CriteriaFn
from typing import Optional

from basic_typing import ShapeX
from transforms import transforms
from transforms import crop
from transforms import resize
from typing_extensions import Literal


def _transform_random_crop_resize(features_names, batch, crop_size, resize_mode):
    arrays = [batch[name] for name in features_names]
    assert len(arrays) > 0, 'no features found!'

    shape = arrays[0].shape[2:]
    cropped_arrays = crop.transform_batch_random_crop_joint(arrays, [arrays[0].shape[1]] + list(crop_size))
    if resize_mode == 'none':
        resized_arrays = cropped_arrays
    else:
        resized_arrays = [resize(cropped_array, shape, resize_mode) for cropped_array in cropped_arrays]

    new_batch = collections.OrderedDict(zip(features_names, resized_arrays))
    for feature_name, feature_value in batch.items():
        if feature_name not in features_names:
            # not in the transformed features, so copy the original value
            new_batch[feature_name] = feature_value

    return new_batch


class TransformRandomCropResize(transforms.TransformBatchWithCriteria):
    """
    Randomly crop a tensor and resize to its original shape.

    Args:
        crop_size: a sequence of size `len(array.shape)-2` indicating the width of crop, excluding
            the ``N`` and ``C`` components
        criteria_fn: function applied on each feature. If satisfied, the feature will be transformed, if not
            the original feature is returned
        resize_mode: string among ('nearest', 'linear', 'none') specifying the resampling method

    Returns:
        a transformed batch
    """
    def __init__(self,
                 crop_size: ShapeX,
                 criteria_fn: Optional[CriteriaFn] = None,
                 resize_mode: Literal['nearest', 'linear', 'none'] = 'linear'):

        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_4_or_above

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=functools.partial(
                _transform_random_crop_resize,
                crop_size=crop_size,
                resize_mode=resize_mode)
         )
        self.criteria_fn = transforms.criteria_is_array_4_or_above
