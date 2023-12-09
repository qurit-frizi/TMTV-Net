import collections

import copy
from numbers import Number

import torch
from basic_typing import Activation, ConvKernels, ConvStrides, PoolingSizes, Paddings, IntListList
# from .utils import div_shape
from .blocks import BlockConvNormActivation, BlockPool, ConvBlockType
from .layer_config import default_layer_config, NormType, LayerConfig
import torch.nn as nn
from typing import Union, Dict, Sequence, Optional, List

# from ..utils import flatten


def div_shape(shape: Union[Sequence[int], int], div: int = 2) -> Union[Sequence[int], int]:
    """
    Divide the shape by a constant

    Args:
        shape: the shape
        div: a divisor

    Returns:
        a list
    """
    if isinstance(shape, (list, tuple, torch.Size)):
        return [s // div for s in shape]
    assert isinstance(shape, int)
    return shape // div

class ModuleWithIntermediate:
    """
    Represent a module with intermediate results
    """
    def forward_with_intermediate(self, x: torch.Tensor, **kwargs) -> Sequence[torch.Tensor]:
        raise NotImplementedError()


class ConvsBase(nn.Module, ModuleWithIntermediate):
    def __init__(
            self,
            dimensionality: int,
            input_channels: int,
            *,
            channels: Sequence[int],
            convolution_kernels: ConvKernels = 5,
            strides: ConvStrides = 1,
            pooling_size: Optional[PoolingSizes] = 2,
            convolution_repeats: Union[int, Sequence[int], IntListList] = 1,
            activation: Optional[Activation] = nn.ReLU,
            padding: Paddings = 'same',
            with_flatten: bool = False,
            dropout_probability: Optional[float] = None,
            norm_type: Optional[NormType] = None,
            norm_kwargs: Dict = {},
            pool_kwargs: Dict = {},
            activation_kwargs: Dict = {},
            last_layer_is_output: bool = False,
            conv_block_fn: ConvBlockType = BlockConvNormActivation,
            config: LayerConfig = default_layer_config(dimensionality=None)):
        """
        Args:
            dimensionality: the dimension of the  CNN (2 for 2D or 3 for 3D)
            input_channels: the number of input channels
            channels: the number of channels for each convolutional layer
            convolution_kernels: for each convolution group, the kernel of the convolution
            strides: for each convolution group, the stride of the convolution
            pooling_size: the pooling size to be inserted after each convolution group
            convolution_repeats: the number of repeats of a convolution. ``1`` means no repeat.
            activation: the activation function
            with_flatten: if True, the last output will be flattened
            norm_kwargs: additional arguments to be used for the normalization layer
            padding: 'same' will add padding so that convolution output as the same size as input
            last_layer_is_output: if True, the last convolution will NOT have activation, dropout, batch norm, LRN
            dropout_probability: dropout probability
        """
        super().__init__()

        # update the configuration locally
        config = copy.copy(config)
        if norm_type is not None:
            config.norm_type = norm_type
        if activation is not None:
            config.activation = activation
        config.set_dim(dimensionality)
        config.pool_kwargs = {**pool_kwargs, **config.pool_kwargs}
        config.norm_kwargs = {**norm_kwargs, **config.norm_kwargs}
        config.activation_kwargs = {**activation_kwargs, **config.activation_kwargs}
        self.dimensionality = dimensionality

        # normalize the arguments
        nb_convs = len(channels)
        if not isinstance(convolution_kernels, collections.Sequence):
            convolution_kernels = [convolution_kernels] * nb_convs
        if not isinstance(strides, collections.Sequence):
            assert isinstance(strides, int) or (isinstance(strides, tuple) and isinstance(strides[0], int)), \
                'stride must be a number or a tuple!'
            strides = [strides] * nb_convs  # type: ignore

        if not isinstance(pooling_size, collections.Sequence) and pooling_size is not None:
            pooling_size = [pooling_size] * nb_convs
        if isinstance(convolution_repeats, int):
            convolution_repeats = [convolution_repeats] * nb_convs
        if isinstance(padding, int):
            padding = [padding] * nb_convs
        elif isinstance(padding, str):
            pass
        else:
            assert len(padding) == nb_convs

        assert nb_convs == len(convolution_kernels), 'must be specified for each convolutional layer'
        assert nb_convs == len(strides), 'must be specified for each convolutional layer'
        assert pooling_size is None or nb_convs == len(pooling_size), 'must be specified for each convolutional layer'
        assert nb_convs == len(convolution_repeats)
        assert isinstance(convolution_repeats, collections.Sequence)

        self.with_flatten = with_flatten

        layers = nn.ModuleList()
        prev = input_channels
        for n in range(len(channels)):
            is_last_layer = (n + 1) == len(channels)
            current = channels[n]

            if padding == 'same':
                p = div_shape(convolution_kernels[n], 2)
            else:
                assert not isinstance(padding, str)
                p = padding[n]  # type: ignore

            ops = []

            nb_repeats = convolution_repeats[n]
            assert isinstance(nb_repeats, int)
            for r in range(nb_repeats):
                is_last_repetition = (r + 1) == nb_repeats
                if is_last_repetition:
                    stride = strides[n]
                else:
                    stride = 1

                if last_layer_is_output and is_last_layer and is_last_repetition:
                    # Last layer layer should not have dropout/normalization/activation
                    config.activation = None
                    config.norm = None
                    config.dropout = None
                ops.append(conv_block_fn(config, prev, current, kernel_size=convolution_kernels[n], padding=p, stride=stride))  # type: ignore
                prev = current

            if pooling_size is not None:
                ops.append(BlockPool(config, kernel_size=pooling_size[n]))

            if not is_last_layer or not last_layer_is_output:
                if dropout_probability is not None and config.dropout is not None:
                    ops.append(config.dropout(p=dropout_probability, **config.dropout_kwargs))

            layers.append(nn.Sequential(*ops))

        self.layers = layers

    def forward_simple(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        if self.with_flatten:
            x = flatten(x)
        return x

    def forward_with_intermediate(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        assert len(kwargs) == 0, f'unsupported arguments={kwargs}'
        assert len(x.shape) == self.dimensionality + 2, \
            f'Expecting a tensor (batch and channel included) of dim={self.dimensionality + 2}, got={len(x.shape)}'

        r = []
        for layer in self.layers:
            x = layer(x)
            r.append(x)

        return r

    def forward(self, x):
        return self.forward_simple(x)

