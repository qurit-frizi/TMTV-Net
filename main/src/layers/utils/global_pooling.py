import torch.nn.functional as F
from basic_typing import TorchTensorNCX


def global_max_pooling_2d(tensor: TorchTensorNCX) -> TorchTensorNCX:
    """
    2D Global max pooling.

    Calculate the max value per sample per channel of a tensor.

    Args:
        tensor: tensor with shape NCHW

    Returns:
        a tensor of shape NC
    """
    assert len(tensor.shape) == 4, 'must be a NCHW tensor!'
    return F.max_pool2d(tensor, tensor.shape[2:]).squeeze(2).squeeze(2)


def global_average_pooling_2d(tensor: TorchTensorNCX) -> TorchTensorNCX:
    """
    2D Global average pooling.

    Calculate the average value per sample per channel of a tensor.

    Args:
        tensor: tensor with shape NCHW

    Returns:
        a tensor of shape NC
    """
    assert len(tensor.shape) == 4, 'must be a NCHW tensor!'
    return F.avg_pool2d(tensor, tensor.shape[2:]).squeeze(2).squeeze(2)


def global_max_pooling_3d(tensor: TorchTensorNCX) -> TorchTensorNCX:
    """
    3D Global max pooling.

    Calculate the max value per sample per channel of a tensor.

    Args:
        tensor: tensor with shape NCDHW

    Returns:
        a tensor of shape NC
    """
    assert len(tensor.shape) == 5, 'must be a NCDHW tensor!'
    return F.max_pool3d(tensor, tensor.shape[2:]).squeeze(2).squeeze(2).squeeze(2)


def global_average_pooling_3d(tensor: TorchTensorNCX) -> TorchTensorNCX:
    """
    3D Global average pooling.

    Calculate the average value per sample per channel of a tensor.

    Args:
        tensor: tensor with shape NCDHW

    Returns:
        a tensor of shape NC
    """
    assert len(tensor.shape) == 5, 'must be a NCDHW tensor!'
    return F.avg_pool3d(tensor, tensor.shape[2:]).squeeze(2).squeeze(2).squeeze(2)
