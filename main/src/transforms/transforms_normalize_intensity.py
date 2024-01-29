import functools
from numbers import Number
from .transforms import CriteriaFn
from typing import Optional, Sequence

from transforms import transforms
from transforms.normalize import normalize
import collections


def _transform_normalize(features_names, batch, mean, std):
    new_batch = collections.OrderedDict()
    for feature_name, feature_value in batch.items():
        if feature_name in features_names:
            new_batch[feature_name] = normalize(feature_value, mean=mean, std=std)
        else:
            new_batch[feature_name] = feature_value
    return new_batch


class TransformNormalizeIntensity(transforms.TransformBatchWithCriteria):
    """
    Normalize a tensor image with mean and standard deviation.

    Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel
    of the input torch.Tensor, input[channel] = (input[channel] - mean[channel]) / std[channel]

    Args:
        array: the torch array to normalize. Expected layout is (sample, filter, d0, ... dN)
        mean: a N-dimensional sequence
        std: a N-dimensional sequence
        criteria_fn: function applied on each feature. If satisfied, the feature will be transformed, if not
            the original feature is returned

    Returns:
        A normalized batch such that the mean is 0 and std is 1 for the selected features
    """
    def __init__(
            self,
            mean: Sequence[Number],
            std: Sequence[Number],
            criteria_fn: Optional[CriteriaFn] = None):

        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_4_or_above

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=functools.partial(_transform_normalize, mean=mean, std=std)
         )
        self.criteria_fn = criteria_fn
