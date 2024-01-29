import numpy as np
import torch
from basic_typing import Tensor


def copy(array: Tensor) -> Tensor:
    """
    Copy an array

    Args:
        array: a :class:`numpy.ndarray` or :class:`torch.Tensor` n-dimensional array

    Returns:
        an array with specified axis flipped
    """
    if isinstance(array, np.ndarray):
        return array.copy()
    elif isinstance(array, torch.Tensor):
        return array.clone()
    else:
        raise NotImplementedError()
