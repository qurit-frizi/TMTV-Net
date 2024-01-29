from typing import Union, Tuple, List

import numpy as np
import torch
from typing import Sequence

from basic_typing import TensorNCX


def _crop_5d(image, min, max):
    return image[min[0]:max[0], min[1]:max[1], min[2]:max[2], min[3]:max[3], min[4]:max[4]]


def _crop_4d(image, min, max):
    return image[min[0]:max[0], min[1]:max[1], min[2]:max[2], min[3]:max[3]]


def _crop_3d(image, min, max):
    return image[min[0]:max[0], min[1]:max[1], min[2]:max[2]]


def _crop_2d(image, min, max):
    return image[min[0]:max[0], min[1]:max[1]]


def _crop_1d(image, min, max):
    return image[min[0]:max[0]]


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


def transform_batch_random_crop_offset(array: TensorNCX, crop_shape: Sequence[Union[None, int]]) -> np.ndarray:
    """
    Calculate the offsets of array to randomly crop it with shape `crop_shape`

    Examples:
        Crop a 3D arrays stored as NCX. Use `None` to keep dim=1:
        >>> arrays = torch.zeros([10, 1, 64, 64, 64])
        >>> cropped_arrays = transform_batch_random_crop_offset(array, crop_shape=[None, 16, 16, 16])
        >>> cropped_arrays.shape
        [10, 1, 16, 16, 16]

    Args:
        array: a numpy array. Samples are stored in the first dimension
        crop_shape: a sequence of size `len(array.shape)-1` indicating the shape of the crop. If a dimension
            is `None`, the whole axis is kept for this dimension.

    Returns:
        a offsets to crop the array
    """
    nb_dims = len(array.shape) - 1
    assert len(crop_shape) == nb_dims, 'padding must have shape size of {}, got={}'.format(nb_dims, len(crop_shape))
    for index, size in enumerate(crop_shape):
        assert size is None or array.shape[index + 1] >= size, \
            'crop_size is larger than array size! shape={}, crop_size={}, index={}'.\
            format(array.shape[1:], crop_shape, index)

    # calculate the maximum offset per dimension. We can then
    # use `max_offsets` and `crop_shape` to calculate the cropping
    max_offsets = []
    for size, crop_size in zip(array.shape[1:], crop_shape):
        if crop_size is None:
            # crop ALL the dimension
            max_offset = 0
        else:
            assert size is not None
            max_offset = size - crop_size
        max_offsets.append(max_offset)

    nb_samples = array.shape[0]

    # calculate the offset per dimension
    offsets = []
    for max_offset in max_offsets:
        offset = np.random.randint(0, max_offset + 1, nb_samples)
        offsets.append(offset)
    offsets = np.stack(offsets, axis=-1)

    return offsets


def transform_batch_random_crop(
        array: TensorNCX,
        crop_shape: Sequence[Union[int, None]],
        offsets: Sequence[Sequence[int]] = None,
        return_offsets: bool = False) -> Union[TensorNCX, Tuple[TensorNCX, Sequence[Sequence[int]]]]:
    """
    Randomly crop a numpy array of samples given a target size. This works for an arbitrary number of dimensions

    Args:
        array: a numpy or Torch array. Samples are stored in the first dimension
        crop_shape: a sequence of size `len(array.shape)-1` indicating the shape of the crop. If `None` in one
            of the element of the shape, take the whole dimension
        offsets: if `None`, offsets will be randomly created to crop with `crop_shape`, else an array indicating
            the crop position for each sample
        return_offsets: if `True`, returns a tuple (cropped array, offsets)

    Returns:
        a cropped array and optionally the crop positions
    """
    nb_dims = len(array.shape) - 1
    nb_samples = array.shape[0]

    is_numpy = isinstance(array, np.ndarray)
    is_torch = isinstance(array, torch.Tensor)
    assert is_numpy or is_torch, 'must be a numpy array or pytorch.Tensor!'

    if offsets is None:
        offsets = transform_batch_random_crop_offset(array, crop_shape)
    else:
        assert len(offsets) == len(array)
        assert len(offsets[0]) == len(array.shape[1:])

    # select the crop function according to dimension
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

    # handle the `None` in crop shape. In that case crop the whole dimension
    crop_shape = [s if s is not None else array.shape[d + 1] for d, s in enumerate(crop_shape)]

    cropped_array = []
    for n in range(nb_samples):
        min_corner = np.asarray(offsets[n])
        cropped_array.append(crop_fn(array[n], min_corner, min_corner + crop_shape))

    if is_numpy:
        output = np.asarray(cropped_array)
    elif is_torch:
        output = torch.stack(cropped_array)
    else:
        assert 0, 'unreachable!'

    if return_offsets:
        return output, offsets
    return output


def transform_batch_random_crop_joint(
        arrays: Sequence[TensorNCX],
        crop_shape: Sequence[Union[None, int]],
        return_offsets: bool = False) -> Union[List[TensorNCX], Tuple[List[TensorNCX], TensorNCX]]:
    """
    Randomly crop a list of arrays. Apply the same cropping for each array element

    Args:
        arrays: a list of numpy or Torch arrays. Samples are stored in the first dimension
        crop_shape: a sequence of size `len(array.shape)-1` indicating the size of the cropped tensor
        return_offsets: if true, returns where the cropping started

    Returns:
        a cropped array, [cropping offset]
    """

    assert isinstance(arrays, list), 'must be a list of arrays'
    assert isinstance(arrays[0], (torch.Tensor, np.ndarray)), 'must be a list of arrays'

    shape = arrays[0].shape
    for a in arrays[1:]:
        # number of filters may be different (e.g., segmentation map & color image)
        assert a.shape[2:] == shape[2:], 'joint crop MUST have the same shape! ' \
                                         'Found={} expected={}'.format(a.shape, shape)
        assert shape[0] == a.shape[0], 'must have the same number of volumes!'

    offsets = transform_batch_random_crop_offset(arrays[0], crop_shape)
    cropped_arrays = [transform_batch_random_crop(a, crop_shape=crop_shape, offsets=offsets) for a in arrays]
    if return_offsets:
        return cropped_arrays, offsets
    return cropped_arrays
