from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from basic_typing import NumpyTensorNCX, ShapeCX, TorchTensorNCX, TensorNCX, Numeric


def batch_pad_minmax_numpy(
        array: NumpyTensorNCX,
        padding_min: ShapeCX,
        padding_max: ShapeCX,
        mode: str = 'edge',
        constant_value: Numeric = 0) -> NumpyTensorNCX:
    """
    Add padding on a numpy array of samples. This works for an arbitrary number of dimensions

    Args:
        array: a numpy array. Samples are stored in the first dimension
        padding_min: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning of each dimension (except for dimension 0)
        padding_max: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode
        constant_value: constant used if mode == `constant`

    Returns:
        a padded array
    """
    assert isinstance(array, np.ndarray), 'must be a numpy array!'
    assert len(padding_min) == len(array.shape) - 1, 'padding must have shape size of {}'.format(len(array.shape) - 1)
    assert len(padding_max) == len(array.shape) - 1, 'padding must have shape size of {}'.format(len(array.shape) - 1)
    full_padding_min = [0] + list(padding_min)
    full_padding_max = [0] + list(padding_max)
    full_padding = list(zip(full_padding_min, full_padding_max))

    if mode == 'constant':
        constant_values = [(constant_value, constant_value)] * len(array.shape)
        padded_array = np.pad(array, full_padding, mode='constant', constant_values=constant_values)
    else:
        padded_array = np.pad(array, full_padding, mode=mode)
    return padded_array


def batch_pad_minmax_torch(
        array: TorchTensorNCX,
        padding_min: ShapeCX,
        padding_max: ShapeCX,
        mode: str = 'edge',
        constant_value: Numeric = 0) -> TorchTensorNCX:
    """
    Add padding on a numpy array of samples. This works for an arbitrary number of dimensions

    This function mimics the API of `transform_batch_pad_numpy` so they can be easily interchanged.

    Args:
        array: a Torch array. Samples are stored in the first dimension
        padding_min: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning of each dimension (except for dimension 0)
        padding_max: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode. Currently supported are ('constant', 'edge', 'symmetric')
        constant_value: constant used if mode == `constant`

    Returns:
        a padded array
    """
    assert isinstance(array, torch.Tensor), 'must be a torch.Tensor!'
    assert len(padding_min) == len(array.shape) - 1, 'padding must have shape size of {}'.format(len(array.shape) - 1)
    assert len(padding_max) == len(array.shape) - 1, 'padding must have shape size of {}'.format(len(array.shape) - 1)

    full_padding = []
    # pytorch start from last dimension to first dimension so reverse the padding
    for p_min, p_max in zip(reversed(padding_min), reversed(padding_max)):
        full_padding.append(p_min)
        full_padding.append(p_max)

    if mode == 'edge':
        mode = 'replicate'
    if mode == 'symmetric':
        mode = 'reflect'

    if mode != 'constant':
        # for reflect and replicate we MUST remove the `component` padding
        full_padding = full_padding[:-2]

    if mode == 'constant':
        padded_array = F.pad(array, full_padding, mode='constant', value=constant_value)
    else:
        padded_array = F.pad(array, full_padding, mode=mode)
    return padded_array


def batch_pad_minmax(
        array: TensorNCX,
        padding_min: ShapeCX,
        padding_max: ShapeCX,
        mode: str = 'edge',
        constant_value: Numeric = 0) -> TensorNCX:
    """
    Add padding on a numpy array of samples. This works for an arbitrary number of dimensions

    Args:
        array: a numpy array. Samples are stored in the first dimension
        padding_min: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning of each dimension (except for dimension 0)
        padding_max: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode
        constant_value: constant used if mode == `constant`

    Returns:
        a padded array
    """
    if isinstance(array, np.ndarray):
        return batch_pad_minmax_numpy(array, padding_min, padding_max, mode, constant_value)
    elif isinstance(array, torch.Tensor):
        return batch_pad_minmax_torch(array, padding_min, padding_max, mode, constant_value)

    raise NotImplementedError()


def batch_pad_minmax_joint(
        arrays: List[TensorNCX],
        padding_min: ShapeCX,
        padding_max: ShapeCX,
        mode: str = 'edge',
        constant_value: Numeric = 0) -> List[TensorNCX]:
    """
    Add padding on a list of numpy or tensor array of samples. Supports arbitrary number of dimensions

    Args:
        arrays: a numpy array. Samples are stored in the first dimension
        padding_min: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning of each dimension (except for dimension 0)
        padding_max: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode
        constant_value: constant used if mode == `constant`

    Returns:
        a list of padded arrays
    """
    assert isinstance(arrays, list), 'must be a list of arrays'
    padded_arrays = [
        batch_pad_minmax(
            a,
            padding_min=padding_min,
            padding_max=padding_max,
            mode=mode,
            constant_value=constant_value) for a in arrays
    ]
    return padded_arrays
