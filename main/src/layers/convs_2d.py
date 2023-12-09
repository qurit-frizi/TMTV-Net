from typing import Union, Any, Dict, Optional, Sequence, Callable

import torch.nn as nn
from basic_typing import PoolingSizes, ConvStrides, ConvKernels, Activation, Paddings
from .blocks import BlockConvNormActivation, ConvBlockType
from .layer_config import NormType, default_layer_config, LayerConfig
from .convs import ConvsBase


def convs_2d(
        input_channels: int,
        channels: Sequence[int],
        convolution_kernels: ConvKernels = 5,
        strides: ConvStrides = 1,
        pooling_size: Optional[PoolingSizes] = 2,
        convolution_repeats: Union[int, Sequence[int]] = 1,
        activation: Activation = nn.ReLU,
        padding: Paddings = 'same',
        with_flatten: bool = False,
        dropout_probability: Optional[float] = None,
        norm_type: Optional[NormType] = None,
        norm_kwargs: Dict[str, Any] = {},
        pool_kwargs: Dict[str, Any] = {},
        last_layer_is_output: bool = False,
        conv_block_fn: ConvBlockType = BlockConvNormActivation,
        config: LayerConfig = default_layer_config(dimensionality=None)):
    """

    Args:
        input_channels: the number of input channels
        channels: the number of channels
        convolution_kernels: for each convolution group, the kernel of the convolution
        strides: for each convolution group, the stride of the convolution
        pooling_size: the pooling size to be inserted after each convolution group
        convolution_repeats: the number of repeats of a convolution
        activation: the activation function
        with_flatten: if True, the last output will be flattened
        dropout_probability: if None, not dropout. Else the probability of dropout after each convolution
        padding: 'same' will add padding so that convolution output as the same size as input
        last_layer_is_output: if True, the last convolution will NOT have activation, dropout, batch norm, LRN
        norm_type: the normalization layer (e.g., BatchNorm)
        norm_kwargs: additional arguments for normalization
        pool_kwargs: additional argument for pool
        conv_block_fn: the base blocks convolutional
        config: defines the allowed operations
    """

    return ConvsBase(
        dimensionality=2,
        input_channels=input_channels,
        channels=channels,
        convolution_kernels=convolution_kernels,
        strides=strides,
        pooling_size=pooling_size,
        convolution_repeats=convolution_repeats,
        activation=activation,
        padding=padding,
        with_flatten=with_flatten,
        dropout_probability=dropout_probability,
        norm_type=norm_type,
        norm_kwargs=norm_kwargs,
        pool_kwargs=pool_kwargs,
        last_layer_is_output=last_layer_is_output,
        conv_block_fn=conv_block_fn,
        config=config)

