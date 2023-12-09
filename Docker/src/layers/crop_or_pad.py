import torch.nn.functional as F
import numpy as np
import torch
from typing import Sequence
from typing import Union

"""Generic Tensor as numpy or torch. Must be shaped as [N, C, D, H, W, ...]"""
TensorNCX = Union[np.ndarray, torch.Tensor]


def batch_crop(images: TensorNCX, min_index: Sequence[int], max_index_exclusive: Sequence[int]) -> TensorNCX:
    """
    Crop an image
    Args:
        images: images with shape [N * ...]
        min_index: a sequence of size `len(array.shape)-1` indicating cropping start
        max_index_exclusive: a sequence of size `len(array.shape)-1` indicating cropping end (excluded)

    Returns:
        a cropped images
    """
    nb_dims = len(images.shape)

    if nb_dims == 1:
        crop_fn = _crop_1d
    elif nb_dims == 2:
        crop_fn = _crop_2d
    elif nb_dims == 3:
        crop_fn = _crop_3d
    elif nb_dims == 4:
        crop_fn = _crop_4d
    elif nb_dims == 5:
        crop_fn = _crop_5d
    else:
        assert 0, 'TODO implement for generic dimension'

    return crop_fn(images, [0] + list(min_index), [len(images)] + list(max_index_exclusive))


def crop_or_pad_fun(x: torch.Tensor, shape: Sequence[int], padding_default_value=0) -> torch.Tensor:
    """
    Crop or pad a tensor to the specified shape (``N`` and ``C`` excluded)

    Args:
        x: the tensor shape
        shape: the shape of x to be returned. ``N`` and ``C`` channels must not be specified
        padding_default_value: the padding value to be used

    Returns:
        torch.Tensor
    """
    assert len(shape) + 2 == len(x.shape), f'Expected dim={len(x.shape) - 2} got={len(shape)}. ' \
                                           f'`N` and `C components should not be included!`'

    shape_x = np.asarray(x.shape[2:])
    shape_difference = np.asarray(shape) - shape_x
    assert (shape_difference >= 0).all() or (shape_difference <= 0).all(), \
        f'Not implemented. Expected the decoded shape to ' \
        f'be smaller than x! Shape difference={shape_difference}'

    if np.abs(shape_difference).max() == 0:
        # x has already the right shape!
        return x

    if shape_difference.max() > 0:
        # here we need to add padding
        left_padding = shape_difference // 2
        right_padding = shape_difference - left_padding

        # padding must remove N, C channels & reversed order
        padding = []
        for left, right in zip(left_padding, right_padding):
            padding += [right, left]
        padding = list(padding[::-1])
        padded_decoded_x = F.pad(x, padding, mode='constant', value=padding_default_value)
        return padded_decoded_x
    else:
        # we need to crop the image
        shape_difference = - shape_difference
        left_crop = shape_difference // 2
        right_crop = shape_x - (shape_difference - left_crop)
        cropped_decoded_x = batch_crop(x, [0] + list(left_crop), [x.shape[1]] + list(right_crop))
        return cropped_decoded_x
