from typing import Optional

import torch.nn as nn
import functools

from basic_typing import ModuleCreator
from typing_extensions import Literal


def upsample_mode(mode: Literal['nearest', 'linear'], dim: int) -> ModuleCreator:
    if mode == 'linear':
        if dim == 1:
            return functools.partial(nn.Upsample, mode='linear')
        elif dim == 2:
            return functools.partial(nn.Upsample, mode='bilinear')
        elif dim == 3:
            return functools.partial(nn.Upsample, mode='trilinear')
        else:
            raise ValueError(f'mode not implemented for mode={mode}, dim={dim}')
    elif mode == 'nearest':
        return functools.partial(nn.Upsample, mode='nearest')
    else:
        raise ValueError(f'mode not implemented={mode}')


class OpsConversion:
    """
    Helper to create standard N-d operations
    """
    def __init__(self, upsample_mode: Literal['nearest', 'linear'] = 'linear'):
        self.dim: Optional[int] = None
        self.group_norm_fn: Optional[ModuleCreator] = None
        self.upsample_mode = upsample_mode
        try:
            self.group_norm_fn = nn.GroupNorm
        except:
            pass

        self.sync_bn_fn = None
        try:
            self.sync_bn_fn = nn.SyncBatchNorm
        except:
            pass

        self.layer_norm = None
        try:
            self.layer_norm = nn.LayerNorm
        except:
            pass

        self.lrn_fn: Optional[ModuleCreator] = nn.LocalResponseNorm

        self.conv_fn: Optional[ModuleCreator] = None
        self.decon_fn: Optional[ModuleCreator] = None

        self.max_pool_fn: Optional[ModuleCreator] = None
        self.avg_pool_fn: Optional[ModuleCreator] = None
        self.fractional_max_pool_fn: Optional[ModuleCreator] = None
        self.adaptative_max_pool_fn: Optional[ModuleCreator] = None
        self.adaptative_avg_pool_fn: Optional[ModuleCreator] = None

        self.dropout_fn: Optional[ModuleCreator] = None
        self.dropout1d_fn: Optional[ModuleCreator] = nn.Dropout

        self.alpha_dropout: Optional[ModuleCreator] = None
        try:
            self.alpha_dropout = nn.AlphaDropout
        except:
            pass

        self.upsample_fn: Optional[ModuleCreator] = None
        self.instance_norm: Optional[ModuleCreator] = None
        self.bn_fn: Optional[ModuleCreator] = None
        self.constant_padding: Optional[ModuleCreator] = None

    def set_dim(self, dim: int):
        self.dim = dim

        self.upsample_fn = upsample_mode(self.upsample_mode, dim=dim)

        if dim == 3:
            self.conv_fn = nn.Conv3d
            self.decon_fn = nn.ConvTranspose3d
            self.max_pool_fn = nn.MaxPool3d
            self.avg_pool_fn = nn.AvgPool3d

            self.fractional_max_pool_fn = None
            try:
                self.fractional_max_pool_fn = nn.FractionalMaxPool3d
            except:
                pass

            self.adaptative_max_pool_fn = nn.AdaptiveMaxPool3d
            self.adaptative_avg_pool_fn = nn.AdaptiveAvgPool3d
            self.dropout_fn = nn.Dropout3d

            self.instance_norm = nn.InstanceNorm3d
            self.bn_fn = nn.BatchNorm3d
            self.constant_padding = nn.ConstantPad3d

        elif dim == 2:
            self.conv_fn = nn.Conv2d
            self.decon_fn = nn.ConvTranspose2d
            self.max_pool_fn = nn.MaxPool2d
            self.avg_pool_fn = nn.AvgPool2d

            self.fractional_max_pool_fn = None
            try:
                self.fractional_max_pool_fn = nn.FractionalMaxPool2d
            except:
                pass

            self.adaptative_max_pool_fn = nn.AdaptiveMaxPool2d
            self.adaptative_avg_pool_fn = nn.AdaptiveAvgPool2d
            self.dropout_fn = nn.Dropout2d
            self.instance_norm = nn.InstanceNorm2d
            self.bn_fn = nn.BatchNorm2d
            self.constant_padding = nn.ConstantPad2d

        elif dim == 1:
            self.conv_fn = nn.Conv1d
            self.decon_fn = nn.ConvTranspose1d
            self.max_pool_fn = nn.MaxPool1d
            self.avg_pool_fn = nn.AvgPool1d

            self.fractional_max_pool_fn = None
            self.adaptative_max_pool_fn = nn.AdaptiveMaxPool1d
            self.adaptative_avg_pool_fn = nn.AdaptiveAvgPool1d
            self.dropout_fn = nn.Dropout
            self.instance_norm = nn.InstanceNorm1d
            self.bn_fn = nn.BatchNorm1d
            self.constant_padding = nn.ConstantPad1d

        else:
            raise NotImplementedError()
