from typing import Sequence, Any
import torch


def clamp_n(tensor: torch.Tensor, min_values: Sequence[Any], max_values: Sequence[Any]) -> torch.Tensor:
    """
    Clamp a tensor with axis dependent values.

    Args:
        tensor: a N-d torch.Tensor
        min_values: a 1D torch.Tensor. Min value is axis dependent
        max_values: a 1D torch.Tensor. Max value is axis dependent

    Returns:
        tensor with values clamped to min_values and max_values

    Examples:
        >>> t = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
        >>> min_values = torch.LongTensor([3, 2, 4])
        >>> max_values = torch.LongTensor([3, 4, 8])
        >>> clamped_t = clamp_n(t, min_values, max_values)
    """
    assert isinstance(min_values, torch.Tensor)
    assert isinstance(max_values, torch.Tensor)
    assert min_values.shape == max_values.shape
    if len(min_values.shape) == 1:
        min_values = min_values.unsqueeze(dim=0)
        max_values = max_values.unsqueeze(dim=0)
    else:
        assert min_values.shape[0] == 1, 'must be broadcastable to tensor shape'
        assert max_values.shape[0] == 1, 'must be broadcastable to tensor shape'
    return torch.max(torch.min(tensor, max_values), min_values)
