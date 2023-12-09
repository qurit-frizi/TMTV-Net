from typing import Union, Sequence

import torch


def div_shape(shape: Union[Sequence[int], int], div: int = 2) -> Union[Sequence[int], int]:
    """
    Divide the shape by a constant

    Args:
        shape: the shape
        div: a divisor

    Returns:
        a list
    """
    if isinstance(shape, (list, tuple, torch.Size)):
        return [s // div for s in shape]
    assert isinstance(shape, int)
    return shape // div
