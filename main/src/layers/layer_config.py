from enum import Enum
import torch.nn as nn
from basic_typing import ModuleCreator

from .ops_conversion import OpsConversion
from typing import Dict, Any, Optional


class NormType(Enum):
    """
    Representation of the normalization layer
    """
    BatchNorm = 'BatchNorm'
    InstanceNorm = 'InstanceNorm'
    GroupNorm = 'GroupNorm'
    SyncBatchNorm = 'SyncBatchNorm'
    LocalResponseNorm = 'LocalResponseNorm'


class PoolType(Enum):
    """
    Representation of the pooling layer
    """
    MaxPool = 'MaxPool'
    AvgPool = 'AvgPool'
    FractionalMaxPool = 'FractionalMaxPool'
    AdaptiveMaxPool = 'AdaptiveMaxPool'
    AdaptiveAvgPool = 'AdaptiveAvgPool'


class DropoutType(Enum):
    """
    Representation of the dropout types
    """
    Dropout1d = 'Dropout1d'  # force 1D dropout
    Dropout = 'Dropout'  # N-D dropout
    AlphaDropout = 'AlphaDropout'


def create_dropout_fn(ops: OpsConversion, dropout: Optional[DropoutType]) -> Optional[ModuleCreator]:
    """
    Create the norm function from the ops and norm type

    Args:
        ops: the operations to be used
        dropout: the norm type to create

    Returns:
        a normalization layer
    """
    assert ops.dim is not None, 'call `ops.set_dim` to set the dimension first!'

    if dropout is None:
        return None
    elif dropout == DropoutType.Dropout1d:
        return ops.dropout1d_fn
    elif dropout == DropoutType.Dropout:
        return ops.dropout_fn
    elif dropout == DropoutType.AlphaDropout:
        return ops.alpha_dropout

    return None


def create_pool_fn(ops: OpsConversion, pool: Optional[PoolType]) -> Optional[ModuleCreator]:
    """
    Create the norm function from the ops and pool type

    Args:
        ops: the operations to be used
        pool: the pool type to create

    Returns:
        a normalization layer
    """
    assert ops.dim is not None, 'call `ops.set_dim` to set the dimension first!'

    if pool is None:
        return None
    elif pool == PoolType.MaxPool:
        return ops.max_pool_fn
    elif pool == PoolType.AvgPool:
        return ops.avg_pool_fn
    elif pool == PoolType.FractionalMaxPool:
        return ops.fractional_max_pool_fn
    elif pool == PoolType.AdaptiveMaxPool:
        return ops.adaptative_max_pool_fn
    elif pool == PoolType.AdaptiveAvgPool:
        return ops.adaptative_avg_pool_fn

    return None


def create_norm_fn(ops: OpsConversion, norm: Optional[NormType]) -> Optional[ModuleCreator]:
    """
    Create the norm function from the ops and norm type

    Args:
        ops: the operations to be used
        norm: the norm type to create

    Returns:
        a normalization layer
    """
    assert ops.dim is not None, 'call `ops.set_dim` to set the dimension first!'

    if norm is None:
        return None
    elif norm == NormType.BatchNorm:
        return ops.bn_fn
    elif norm == NormType.InstanceNorm:
        return ops.instance_norm
    elif norm == NormType.GroupNorm:
        return ops.group_norm_fn
    elif norm == NormType.SyncBatchNorm:
        return ops.sync_bn_fn
    elif norm == NormType.LocalResponseNorm:
        return ops.lrn_fn

    return None


class LayerConfig:
    """
    Generic configuration of the layers_legacy
    """
    def __init__(
            self,
            ops: OpsConversion,
            norm_type: Optional[NormType] = NormType.BatchNorm,
            norm_kwargs: Dict = {},
            pool_type: Optional[PoolType] = PoolType.MaxPool,
            pool_kwargs: Dict = {},
            activation: Optional[Any] = nn.ReLU,
            activation_kwargs: Dict = {},
            dropout_type: Optional[DropoutType] = DropoutType.Dropout1d,
            dropout_kwargs: Dict = {},
            conv_kwargs: Dict = {'padding': 'same'},
            deconv_kwargs: Dict = {'padding': 'same'}):

        self.ops = ops
        self.norm_kwargs = norm_kwargs
        self.activation = activation
        self.activation_kwargs = activation_kwargs
        self.conv_kwargs = conv_kwargs
        self.deconv_kwargs = deconv_kwargs
        self.norm_type = norm_type
        self.pool_type = pool_type
        self.pool_kwargs = pool_kwargs
        self.dropout_type = dropout_type
        self.dropout_kwargs = dropout_kwargs

        # types depends on the dimensionality
        self.norm: Optional[ModuleCreator] = None
        self.conv: Optional[ModuleCreator] = None
        self.deconv: Optional[ModuleCreator] = None
        self.pool: Optional[ModuleCreator] = None
        self.dropout: Optional[ModuleCreator] = None

    def set_dim(self, dimensionality: int):
        self.ops.set_dim(dimensionality)
        self.norm = create_norm_fn(self.ops, self.norm_type)
        self.pool = create_pool_fn(self.ops, self.pool_type)
        self.conv = self.ops.conv_fn
        self.deconv = self.ops.decon_fn
        self.dropout = create_dropout_fn(self.ops, self.dropout_type)


def default_layer_config(
        dimensionality: Optional[int] = None,
        norm_type: Optional[NormType] = NormType.BatchNorm,
        norm_kwargs: Dict = {},
        pool_type: Optional[PoolType] = PoolType.MaxPool,
        pool_kwargs: Dict = {},
        activation: Optional[Any] = nn.ReLU,
        activation_kwargs: Dict = {},
        dropout_type: Optional[DropoutType] = DropoutType.Dropout1d,
        dropout_kwargs: Dict = {},
        conv_kwargs: Dict = {'padding': 'same'},
        deconv_kwargs: Dict = {'padding': 'same'}) -> LayerConfig:
    """
    Default layer configuration

    Args:
        dimensionality: the number of dimensions of the input (without the `N` and `C` components)
        norm_type: the type of normalization
        norm_kwargs: additional normalization parameters
        activation: the activation
        activation_kwargs: additional activation parameters
        dropout_kwargs: if not None, dropout parameters
        conv_kwargs: additional parameters for the convolutional layer
        deconv_kwargs: additional arguments for the transposed convolutional layer
        pool_type: the type of pooling
        pool_kwargs: additional parameters for the pooling layers_legacy
        dropout_type: the type of dropout
    """
    conf = LayerConfig(
        OpsConversion(),
        norm_type=norm_type,
        norm_kwargs=norm_kwargs,
        activation=activation,
        activation_kwargs=activation_kwargs,
        conv_kwargs=conv_kwargs,
        pool_type=pool_type,
        pool_kwargs=pool_kwargs,
        dropout_type=dropout_type,
        dropout_kwargs=dropout_kwargs,
        deconv_kwargs=deconv_kwargs,
    )

    if dimensionality is not None:
        conf.set_dim(dimensionality)
    return conf
