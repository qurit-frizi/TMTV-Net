import collections

import functools
from .transforms import CriteriaFn
from typing import Optional

from typing_extensions import Literal

from utils import batch_pad_joint
from transforms import transforms
from transforms import crop
from basic_typing import ShapeCX, Numeric


def _transform_random_crop_pad(features_names, batch, padding, mode='edge', constant_value=0, shape=None):
    """
    Add a specified padding to the image and randomly crop it so that we have the same shape as the original
    image

    This support joint padding & cropping of multiple arrays (e.g., to support segmentation maps)

    Args:
        features_names: the name of the features to be jointly random cropped
        batch: the batch to transform
        padding: a sequence of shape `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode. Currently supported are ('constant', 'edge', 'symmetric')
        shape: if `None`, the image will be cropped to the original shape, else it must be a list of the shape to crop
            for each dimension except for dimension 0

    Returns:
        a padded and cropped image to original shape
    """

    # we have joint arrays, padding and cropping must be identical for all arrays
    if shape is None:
        shape = batch[features_names[0]].shape[1:]

    arrays = [batch[name] for name in features_names]
    if padding is not None:
        padded_arrays = batch_pad_joint(
            arrays,
            padding=padding,
            mode=mode,
            constant_value=constant_value)
    else:
        padded_arrays = arrays

    cropped_arrays = crop.transform_batch_random_crop_joint(padded_arrays, shape)

    new_batch = collections.OrderedDict(zip(features_names, cropped_arrays))
    for feature_name, feature_value in batch.items():
        if feature_name not in features_names:
            # not in the transformed features, so copy the original value
            new_batch[feature_name] = feature_value

    return new_batch


class TransformRandomCropPad(transforms.TransformBatchWithCriteria):
    """
    Add padding on a numpy array of samples and random crop to original size

    Args:
        padding: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0).
            If `None`, no padding added
        criteria_fn: function applied on each feature. If satisfied, the feature will be transformed, if not
            the original feature is returned
        mode: `numpy.pad` mode. Currently supported are ('constant', 'edge', 'symmetric')
        shape: the size of the cropped image. If `None`, same size as input image

    Returns:
        a randomly cropped batch
    """
    def __init__(
            self,
            padding: Optional[ShapeCX],
            criteria_fn: Optional[CriteriaFn] = None,
            mode: Literal['constant', 'edge', 'symmetric'] = 'constant',
            constant_value: Numeric = 0,
            shape: Optional[ShapeCX] = None):

        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_4_or_above

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=functools.partial(
                _transform_random_crop_pad,
                padding=padding,
                mode=mode,
                constant_value=constant_value,
                shape=shape)
         )
        self.criteria_fn = transforms.criteria_is_array_4_or_above

