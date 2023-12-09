import torch.nn as nn
import torch


class Flatten(nn.Module):
    """
    Flatten a tensor

    For example, a tensor of shape[N, Z, Y, X] will be reshaped [N, Z * Y * X]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a tensor

        Returns: return a flattened tensor
        """
        dim = 1
        for d in x.shape[1:]:
            dim *= d
        return x.view((-1, dim))
