from typing import Sequence, List, Optional

import numpy as np
import torch
from basic_typing import Tensor, TensorNCX
from transforms.stack import stack


def flip(array: Tensor, axis: int) -> Tensor:
    """
    Flip an axis of an array

    Args:
        array: a :class:`numpy.ndarray` or :class:`torch.Tensor` n-dimensional array
        axis: the xis to flip

    Returns:
        an array with specified axis flipped
    """
    if isinstance(array, np.ndarray):
        return np.flip(array, axis=axis)
    elif isinstance(array, torch.Tensor):
        return torch.flip(array, [axis])
    else:
        raise NotImplementedError()


def transform_batch_random_flip(
        array: TensorNCX,
        axis: int,
        flip_probability: Optional[float] = 0.5,
        flip_choices: Sequence[bool] = None) -> TensorNCX:
    """
    Randomly flip an image with a given probability

    Args:
        array: a :class:`numpy.ndarray` or :class:`torch.Tensor` n-dimensional array. Samples are stored on axis 0
        axis: the axis to flip
        flip_probability: the probability that a sample is flipped
        flip_choices: for each sample, `True` or `False` to indicate if the sample is flipped or not

    Returns:
        an array
    """
    if flip_choices is None:
        r = np.random.rand(array.shape[0])
        flip_choices = r <= flip_probability
    else:
        assert len(flip_choices) == len(array)

    samples = []
    for flip_choice, sample in zip(flip_choices, array):
        if flip_choice:
            samples.append(flip(sample, axis=axis - 1))  # `-1` since the `N` axis is removed 
        else:
            samples.append(sample)

    return stack(samples)


def transform_batch_random_flip_joint(
        arrays: List[TensorNCX],
        axis: int,
        flip_probability: float = 0.5) -> List[TensorNCX]:
    """
    Randomly flip a joint images with a given probability

    Args:
        arrays: a list of a :class:`numpy.ndarray` or :class:`torch.Tensor` n-dimensional array. Samples for
            each array are stored on axis 0
        axis: the axis to flip
        flip_probability: the probability that a sample is flipped

    Returns:
        an array
    """
    assert isinstance(arrays, list), 'must be a list of arrays'
    nb_samples = len(arrays[0])
    for a in arrays[1:]:
        assert len(a) == nb_samples

    r = np.random.rand(nb_samples)
    flip_choices = r <= flip_probability

    transformed_arrays = [transform_batch_random_flip(a, axis=axis, flip_probability=None, flip_choices=flip_choices) for a in arrays]
    return transformed_arrays
