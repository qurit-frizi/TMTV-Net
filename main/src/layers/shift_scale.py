import torch.nn as nn
import torch
from typing import Union


def transfer_to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if x.device != device:
            return x.to(device)
    return x


class ShiftScale(nn.Module):
    """
    Normalize a tensor with a mean and standard deviation

    The output tensor will be (x - mean) / standard_deviation

    This layer simplify the preprocessing for the `trw.simple_layers` package
    """
    
    def __init__(self,
                 mean: Union[float, torch.Tensor],
                 standard_deviation: Union[float, torch.Tensor],
                 output_dtype: torch.dtype = torch.float32):
        """

        Args:
            mean:
            standard_deviation:
        """
        super().__init__()
        self.mean = torch.tensor(mean)
        self.standard_deviation = torch.tensor(standard_deviation)
        self.output_dtype = output_dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a tensor

        Returns: return a flattened tensor
        """
        mean = transfer_to_device(self.mean, x.device)
        standard_deviation = transfer_to_device(self.standard_deviation, x.device)
        o = (x - mean) / standard_deviation

        self.mean = mean
        self.standard_deviation = standard_deviation
        return o.type(self.output_dtype)
