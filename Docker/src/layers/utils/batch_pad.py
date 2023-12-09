from numbers import Number
from typing import List

from basic_typing import NumpyTensorNCX, ShapeCX, TorchTensorNCX, TensorNCX, Numeric
from .batch_pad_minmax import batch_pad_minmax, batch_pad_minmax_joint, \
    batch_pad_minmax_numpy, batch_pad_minmax_torch


def batch_pad_numpy(array: NumpyTensorNCX, padding: ShapeCX, mode: str = 'edge', constant_value: Numeric = 0):
    """
    Add padding on a numpy array of samples. This works for an arbitrary number of dimensions

    Args:
        array: a numpy array. Samples are stored in the first dimension
        padding: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode
        constant_value: constant used if mode == `constant`

    Returns:
        a padded array
    """
    return batch_pad_minmax_numpy(array, padding, padding, mode, constant_value)


def batch_pad_torch(array: TorchTensorNCX, padding: ShapeCX, mode: str = 'edge', constant_value: Numeric = 0):
    """
    Add padding on a numpy array of samples. This works for an arbitrary number of dimensions

    This function mimics the API of `transform_batch_pad_numpy` so they can be easily interchanged.

    Args:
        array: a Torch array. Samples are stored in the first dimension
        padding: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode. Currently supported are ('constant', 'edge', 'symmetric')
        constant_value: constant used if mode == `constant`

    Returns:
        a padded array
    """
    return batch_pad_minmax_torch(array, padding, padding, mode, constant_value)


def batch_pad(array: TensorNCX, padding: ShapeCX, mode: str = 'edge', constant_value: Numeric = 0):
    """
    Add padding on a numpy array of samples. This works for an arbitrary number of dimensions

    Args:
        array: a numpy array. Samples are stored in the first dimension
        padding: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode
        constant_value: constant used if mode == `constant`

    Returns:
        a padded array
    """
    return batch_pad_minmax(array, padding, padding, mode, constant_value)


def batch_pad_joint(arrays: List[TensorNCX], padding: ShapeCX, mode: str = 'edge', constant_value: Numeric = 0):
    """
    Add padding on a list of numpy or tensor array of samples. Supports arbitrary number of dimensions

    Args:
        arrays: a numpy array. Samples are stored in the first dimension
        padding: a sequence of size `len(array.shape)-1` indicating the width of the
            padding to be added at the beginning and at the end of each dimension (except for dimension 0)
        mode: `numpy.pad` mode
        constant_value: constant used if mode == `constant`

    Returns:
        a list of padded arrays
    """
    return batch_pad_minmax_joint(arrays, padding, padding, mode, constant_value)
