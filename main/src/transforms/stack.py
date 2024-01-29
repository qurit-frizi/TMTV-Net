import numpy as np
import torch
import collections


def stack(sequence, axis=0):
    """
    stack an array

    Args:
        sequence: a :class:`numpy.ndarray` or :class:`torch.Tensor` n-dimensional array
        axis: the xis to flip

    Returns:
        an array stacked
    """
    assert isinstance(sequence, collections.Sequence), 'muse be a list!'

    if isinstance(sequence[0], np.ndarray):
        return np.stack(sequence, axis=axis)
    elif isinstance(sequence[0], torch.Tensor):
        return torch.stack(sequence, axis)
    else:
        raise NotImplemented()
