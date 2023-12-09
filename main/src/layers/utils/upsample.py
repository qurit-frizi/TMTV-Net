import torch
import torch.nn as nn
from typing_extensions import Literal

from basic_typing import ShapeNCX, TensorNCX, TorchTensorNCX, ShapeX


def _upsample_int_1d(tensor: TorchTensorNCX, size: ShapeNCX) -> TorchTensorNCX:
    # this is just a workaround! TODO assess the speed impact!
    # see https://discuss.pytorch.org/t/what-is-the-good-way-to-interpolate-int-tensor/29490
    assert len(size) + 2 == len(tensor.shape), 'shape must be only the resampled components, ' \
                                               'WITHOUT the batch and filter channels'
    assert len(tensor.shape) == 3

    dx = torch.linspace(0 + 0.5, tensor.shape[2] - 1 + 0.5, steps=size[0], dtype=torch.float32).long()

    tensor_interp = tensor[:, :, dx]
    return tensor_interp


def _upsample_int_2d(tensor: TorchTensorNCX, size: ShapeNCX) -> TorchTensorNCX:
    # this is just a workaround! TODO assess the speed impact!
    # see https://discuss.pytorch.org/t/what-is-the-good-way-to-interpolate-int-tensor/29490
    assert len(size) + 2 == len(tensor.shape), 'shape must be only the resampled components, ' \
                                               'WITHOUT the batch and filter channels'
    assert len(tensor.shape) == 4

    dx = torch.linspace(0 + 0.5, tensor.shape[2] - 1 + 0.5, steps=size[0], dtype=torch.float32).long()
    dy = torch.linspace(0 + 0.5, tensor.shape[3] - 1 + 0.5, steps=size[1], dtype=torch.float32).long()

    tensor_interp = tensor[:, :, dx[:, None], dy]
    return tensor_interp


def _upsample_int_3d(tensor: TorchTensorNCX, size: ShapeNCX) -> TorchTensorNCX:
    # this is just a workaround! TODO assess the speed impact!
    # see https://discuss.pytorch.org/t/what-is-the-good-way-to-interpolate-int-tensor/29490
    assert len(size) + 2 == len(tensor.shape), 'shape must be only the resampled components, ' \
                                               'WITHOUT the batch and filter channels'
    assert len(tensor.shape) == 5

    dx = torch.linspace(0 + 0.5, tensor.shape[2] - 1 + 0.5, steps=size[0], dtype=torch.float32).long()
    dy = torch.linspace(0 + 0.5, tensor.shape[3] - 1 + 0.5, steps=size[1], dtype=torch.float32).long()
    dz = torch.linspace(0 + 0.5, tensor.shape[4] - 1 + 0.5, steps=size[2], dtype=torch.float32).long()

    tensor_interp = tensor[:, :, dx[:, None, None], dy[:, None], dz]
    return tensor_interp


def upsample(tensor: TensorNCX, size: ShapeX, mode: Literal['linear', 'nearest'] = 'linear') -> TensorNCX:
    """
    Upsample a 1D, 2D, 3D tensor

    This is a wrapper around `torch.nn.Upsample` to make it more practical. Support integer based tensors.

    Note:
        PyTorch as of version 1.3 doesn't support non-floating point upsampling
        (see https://github.com/pytorch/pytorch/issues/13218 and https://github.com/pytorch/pytorch/issues/5580).
        Instead use a workaround (TODO assess the speed impact!).


    Args:
        tensor: 1D (shape = b x c x n), 2D (shape = b x c x h x w) or 3D (shape = b x c x d x h x w)
        size: if 1D, shape = n, if 2D shape = h x w, if 3D shape = d x h x w
        mode: `linear` or `nearest`

    Returns:
        an up-sampled tensor with same batch size and filter size as the input
    """

    assert len(size) + 2 == len(tensor.shape), 'shape must be only the resampled components, ' \
                                               'WITHOUT the batch and filter channels'
    assert len(tensor.shape) >= 3, 'only 1D, 2D, 3D tensors are currently handled!'
    assert len(tensor.shape) <= 5, 'only 1D, 2D, 3D tensors are currently handled!'

    size = tuple(size)
    if not torch.is_floating_point(tensor):
        # Workaround for non floating point tensors. Ignore `mode`
        if len(tensor.shape) == 3:
            return _upsample_int_1d(tensor, size)
        elif len(tensor.shape) == 4:
            return _upsample_int_2d(tensor, size)
        elif len(tensor.shape) == 5:
            return _upsample_int_3d(tensor, size)
        else:
            raise NotImplementedError('dimension not implemented!')

    if mode == 'linear':
        align_corners = True
        if len(tensor.shape) == 4:
            # 2D case
            return nn.Upsample(mode='bilinear', size=size, align_corners=align_corners).forward(tensor)
        elif len(tensor.shape) == 5:
            # 3D case
            return nn.Upsample(mode='trilinear', size=size, align_corners=align_corners).forward(tensor)
        elif len(tensor.shape) == 3:
            # 1D case
            return nn.Upsample(mode='linear', size=size, align_corners=align_corners).forward(tensor)
        else:
            assert 0, 'impossible or bug!'

    elif mode == 'nearest':
        return nn.Upsample(mode='nearest', size=size).forward(tensor)
    else:
        assert 0, 'upsample mode ({}) is not handled'.format(mode)
