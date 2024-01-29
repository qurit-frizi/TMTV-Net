from typing import Sequence

import numpy as np
import torch
from basic_typing import NumpyTensorNCX, TorchTensorNCX, TensorNCX


def normalize_numpy(array: NumpyTensorNCX, mean: Sequence[float], std: Sequence[float]) -> NumpyTensorNCX:
    """
    Normalize a tensor image with mean and standard deviation.

    Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel
    of the input torch.Tensor, input[channel] = (input[channel] - mean[channel]) / std[channel]

    Args:
        array: the numpy array to normalize. Expected layout is (sample, filter, d0, ... dN)
        mean: a N-dimensional sequence
        std: a N-dimensional sequence

    Returns:
        A normalized tensor such that the mean is 0 and std is 1
    """
    assert isinstance(array, np.ndarray)
    assert len(mean) == len(std), 'mean and std must have the same size!'
    assert len(mean) == array.shape[1], 'dimension do not match! `mean` or `std` is per channel!'

    shape = [1, array.shape[1]] + [1] * (len(array.shape) - 2)
    mean_np = np.reshape(np.asarray(mean, dtype=array.dtype), shape)
    std_np = np.reshape(np.asarray(std, dtype=array.dtype), shape)
    normalized = (array - mean_np) / std_np
    return normalized


def normalize_torch(array: TorchTensorNCX, mean: Sequence[float], std: Sequence[float]) -> TorchTensorNCX:
    """
    Normalize a tensor image with mean and standard deviation.

    Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel
    of the input torch.Tensor, input[channel] = (input[channel] - mean[channel]) / std[channel]

    Args:
        array: the torch array to normalize. Expected layout is (sample, filter, d0, ... dN)
        mean: a N-dimensional sequence
        std: a N-dimensional sequence

    Returns:
        A normalized tensor such that the mean is 0 and std is 1
    """
    assert isinstance(array, torch.Tensor)
    assert len(mean) == len(std), 'mean and std must have the same size!'
    assert len(mean) == array.shape[1], 'dimension do not match! `mean` or `std` is per channel!'

    shape = [1, array.shape[1]] + [1] * (len(array.shape) - 2)
    mean_np = torch.from_numpy(np.reshape(np.asarray(mean), shape))
    std_np = torch.from_numpy(np.reshape(np.asarray(std), shape))
    normalized = (array - mean_np) / std_np
    return normalized


def normalize(array: TensorNCX, mean: Sequence[float], std: Sequence[float]) -> TensorNCX:
    """
    Normalize a tensor image with mean and standard deviation.

    Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel
    of the input torch.Tensor, input[channel] = (input[channel] - mean[channel]) / std[channel]

    Args:
        array: the torch array to normalize. Expected layout is (sample, filter, d0, ... dN)
        mean: a N-dimensional sequence
        std: a N-dimensional sequence

    Returns:
        A normalized tensor such that the mean is 0 and std is 1
    """
    if isinstance(array, np.ndarray):
        return normalize_numpy(array, mean, std)
    elif isinstance(array, torch.Tensor):
        return normalize_torch(array, mean, std)
    else:
        raise NotImplementedError()
