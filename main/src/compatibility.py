from typing import List, Optional

import warnings

import torch.nn.functional
from torch import nn
from packaging.version import Version

torch_version = Version(torch.__version__)


def affine_grid(theta: torch.Tensor, size: List[int], align_corners: Optional[bool]) -> torch.Tensor:
    """
    Compatibility layer for new arguments introduced in pytorch 1.3

    See :func:`torch.nn.functional.affine_grid`
    """
    if torch_version >= Version('1.3'):
        return torch.nn.functional.affine_grid(theta=theta, size=size, align_corners=align_corners)
    else:
        if not align_corners:
            warnings.warn('`align_corners` argument is not supported in '
                          'this version and is ignored. Results may differ')

        return torch.nn.functional.affine_grid(theta=theta, size=size)


def grid_sample(
        input: torch.Tensor,
        grid: torch.Tensor,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros', align_corners: bool = None) -> torch.Tensor:
    """
    Compatibility layer for argument change between pytorch <= 1.2 and pytorch > 1.3

    See :func:`torch.nn.functional.grid_sample`
    """
    if torch_version < Version('1.3'):
        if not align_corners:
            warnings.warn('`align_corners` argument is not supported in '
                          'this version and is ignored. Results may differ')

        return torch.nn.functional.grid_sample(
            input=input,
            grid=grid,
            mode=mode,
            padding_mode=padding_mode)
    else:
        return torch.nn.functional.grid_sample(
            input=input,
            grid=grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)


class SwishCompat(nn.Module):
    """
    For compatibility with old PyTorch versions
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


if hasattr(nn, 'SiLU'):
    Swish = nn.SiLU
else:
    Swish = SwishCompat  # type: ignore

try:
    inv = torch.linalg.inv
except:
    inv = torch.inverse

try:
    torch_linalg_norm = torch.linalg.norm
except:
    torch_linalg_norm = lambda input, ord, dim: torch.norm(input, p=ord, dim=dim)

if hasattr(torch.nn, 'Identity'):
    Identity = torch.nn.Identity
else:
    class Identity(nn.Module):  # type: ignore
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x
