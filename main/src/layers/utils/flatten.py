from basic_typing import TorchTensorNCX, TorchTensorNX


def flatten(x: TorchTensorNCX) -> TorchTensorNX:
    """
    Flatten a tensor

    Example, a tensor of shape[N, Z, Y, X] will be reshaped [N, Z * Y * X]

    Args:
        x: a tensor

    Returns: return a flattened tensor

    """
    dim = 1
    for d in x.shape[1:]:
        dim *= d
    return x.view((-1, dim))
