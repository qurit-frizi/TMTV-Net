import torch
from .typing import Shape


def sub_tensor(tensor: torch.Tensor, min_indices: Shape, max_indices_exclusive: Shape) -> torch.Tensor:
    """
    Select a region of a tensor (without copy)

    Examples:
        >>> t = torch.randn([5, 10])
        >>> sub_t = sub_tensor(t, [2, 3], [4, 8])
        Returns the t[2:4, 3:8]

        >>> t = torch.randn([5, 10])
        >>> sub_t = sub_tensor(t, [2], [4])
        Returns the t[2:4]

    Args:
        tensor: a tensor
        min_indices: the minimum indices to select for each dimension
        max_indices_exclusive: the maximum indices (excluded) to select for each dimension

    Returns:
        torch.tensor
    """
    assert len(min_indices) == len(max_indices_exclusive)
    assert len(tensor.shape) >= len(min_indices)

    for dim, (min_index, max_index) in enumerate(zip(min_indices, max_indices_exclusive)):
        size = max_index - min_index
        tensor = tensor.narrow(dim, min_index, size)
    return tensor