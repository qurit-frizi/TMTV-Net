import collections

from typing import Union, Optional, Any, Sequence

import math
import torch.nn as nn
from torch.nn import functional as F
import torch
import functools
import numpy as np
from basic_typing import TorchTensorNCX
from typing_extensions import Literal


class FilterFixed(nn.Module):
    """
    Apply a fixed filter to n-dimensional images
    """
    def __init__(self, kernel: torch.Tensor, groups: int = 1, padding: int = 0):
        """
        Args:
            kernel: the kernel. format is expected to be [input_channels, output_channels, filter_size_n, ... filter_size_0]. For example,
            groups: the number of groups (e.g., for gaussian filtering, each channel must be treated as independent)
            padding: the padding to be applied
        """
        super().__init__()
        assert isinstance(kernel, torch.Tensor), 'must be a tensor!'

        # make sure this kernel is not trainable
        cloned_kernel = kernel.data.clone()
        cloned_kernel.requires_grad = False
        self.kernel = cloned_kernel

        if len(kernel.shape) == 3:
            self.conv = functools.partial(F.conv1d, groups=groups, weight=self.kernel, padding=padding)
        elif len(kernel.shape) == 4:
            self.conv = functools.partial(F.conv2d, groups=groups, weight=self.kernel, padding=padding)
        elif len(kernel.shape) == 5:
            self.conv = functools.partial(F.conv3d, groups=groups, weight=self.kernel, padding=padding)
        else:
            raise NotImplementedError()

    def __call__(self, value: TorchTensorNCX) -> TorchTensorNCX:
        return self.conv(value)


class FilterGaussian(FilterFixed):
    """
    Implement a gaussian filter as a :class:`torch.nn.Module`
    """
    def __init__(self,
                 input_channels: int,
                 nb_dims: int,
                 sigma: Union[float, Sequence[float]],
                 kernel_sizes: Optional[Union[int, Sequence[int]]] = None,
                 padding: Literal['same', 'none'] = 'same',
                 device: Optional[torch.device] = None):
        """

        Args:
            input_channels: the number of channels expected as input
            kernel_sizes: the size of the gaussian kernel in each dimension. **Beware** if the kernel is
                too small, the gaussian approximation will be inaccurate. If too large, the computation
                will be much slower for limited accuracy gain. If `None`, an appropriate guess will
                be used based on ``sigma``
            sigma: the variance of the gaussian kernel
            padding: one of `same`, `none`
            nb_dims: the number of dimension of the image, excluding sample & channels dimensions
            device: the memory location of the kernel
        """
        # see https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8
        if not isinstance(sigma, collections.Sequence):
            sigma = [sigma] * nb_dims
        else:
            assert len(sigma) == nb_dims

        if kernel_sizes is not None and not isinstance(kernel_sizes, collections.Sequence):
            kernel_sizes = [kernel_sizes] * nb_dims

        if kernel_sizes is None:
            # estimate the kernel size
            kernel_sizes = []
            for s in sigma:
                k = 5 * s + s
                if k % 2 == 0:
                    k += 1
                kernel_sizes.append(int(k))
        else:
            assert len(kernel_sizes) == nb_dims

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = torch.tensor(1.0)
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_sizes
            ]
        )
        for size, s, mgrid in zip(kernel_sizes, sigma, meshgrids):
            std = s * s
            mean = (size - 1) / 2
            kernel = kernel * 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-0.5 / std * (mgrid - mean) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(input_channels, *[1] * (kernel.dim() - 1))

        if padding == 'same':
            padding_value: Any = tuple(np.asarray(kernel.shape[2:]) // 2)
        elif padding == 'none':
            padding_value = 0
        else:
            raise NotImplemented()

        if device is not None:
            kernel = kernel.to(device)
        super().__init__(kernel=kernel, groups=input_channels, padding=padding_value)
