import collections
import numbers
from typing import Sequence, Union, Optional, Any, Dict, Callable, List

import torch
import torch.nn as nn
from ..basic_typing import IntTupleList, ConvKernels, ConvStrides, Paddings
from .utils import div_shape
from .convs import ModuleWithIntermediate, NormType, LayerConfig, default_layer_config
from .blocks import BlockDeconvNormActivation, ConvTransposeBlockType
import copy

from .crop_or_pad import crop_or_pad_fun


class ConvsTransposeBase(nn.Module, ModuleWithIntermediate):
    """
    Helper class to create sequence of transposed convolution

    This can be used to map an embedding back to image space.
    """
    def __init__(
            self,
            dimensionality: int,
            input_channels: int,
            channels: Sequence[int],
            *,
            convolution_kernels: ConvKernels = 5,
            strides: ConvStrides = 2,
            paddings: Optional[Paddings] = None,
            activation: Any = nn.ReLU,
            activation_kwargs: Dict = {},
            dropout_probability: Optional[float] = None,
            norm_type: Optional[NormType] = None,
            norm_kwargs: Dict = {},
            last_layer_is_output: bool = False,
            squash_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            deconv_block_fn: ConvTransposeBlockType = BlockDeconvNormActivation,
            config: LayerConfig = default_layer_config(dimensionality=None),
            target_shape: Optional[Sequence[int]] = None):
        """

        Args:
            dimensionality: the dimension of the  CNN (2 for 2D or 3 for 3D)
            input_channels: the number of input channels
            channels: the number of channels for each convolutional layer
            convolution_kernels: for each convolution group, the kernel of the convolution
            strides: for each convolution group, the stride of the convolution
            dropout_probability: if None, not dropout. Else the probability of dropout after each convolution
            norm_kwargs: the normalization additional arguments. See the original torch functions for description.
            last_layer_is_output: if True, the last convolution will NOT have activation, dropout, batch norm, LRN
            squash_function: a function to be applied on the reconstuction. It is common to apply
                for example ``torch.sigmoid``. If ``None``, no function applied
            paddings: the paddings added. If ``None``, half the convolution kernel will be used.
            target_shape: if not ``None``, the output layer will be cropped or padded to mach the target
                (N, C components excluded)
        """
        super().__init__()

        # update the configuration locally
        config = copy.copy(config)
        if norm_type is not None:
            config.norm_type = norm_type
        if activation is not None:
            config.activation = activation
        config.set_dim(dimensionality)
        config.norm_kwargs = {**norm_kwargs, **config.activation_kwargs}
        config.activation_kwargs = {**activation_kwargs, **config.activation_kwargs}

        # normalize the arguments
        nb_convs = len(channels)
        if not isinstance(convolution_kernels, collections.Sequence):
            convolution_kernels = [convolution_kernels] * nb_convs
        if not isinstance(strides, collections.Sequence):
            strides = [strides] * nb_convs
        if paddings is None:
            paddings = [div_shape(kernel, 2) for kernel in convolution_kernels]  # type: ignore
        elif isinstance(paddings, int):
            paddings = [paddings] * nb_convs
        else:
            assert isinstance(paddings, collections.Sequence) and len(paddings) == nb_convs

        assert nb_convs == len(convolution_kernels), 'must be specified for each convolutional layer'
        assert nb_convs == len(strides), 'must be specified for each convolutional layer'
        assert isinstance(paddings, collections.Sequence)
        assert len(paddings) == nb_convs

        layers = nn.ModuleList()

        prev = input_channels
        for n in range(len(channels)):
            current = channels[n]
            currently_last_layer = n + 1 == len(channels)

            p = paddings[n]
            if last_layer_is_output and currently_last_layer:
                # Last layer layer should not have dropout/normalization/activation
                config.norm = None
                config.activation = None
                config.dropout = None

            ops = []
            ops.append(deconv_block_fn(
                config,
                prev,
                current,
                kernel_size=convolution_kernels[n],
                stride=strides[n],
                padding=p,
                output_padding=strides[n] - 1,  # type: ignore
            ))

            if config.dropout is not None and dropout_probability is not None:
                ops.append(config.dropout(p=dropout_probability, **config.dropout_kwargs))

            layers.append(nn.Sequential(*ops))
            prev = current
        self.layers = layers
        self.squash_function = squash_function
        self.target_shape = target_shape

    def forward_with_intermediate(self, x):
        r = []
        for layer in self.layers:
            x = layer(x)
            r.append(x)

        if self.squash_function is not None:
            r[-1] = self.squash_function(r[-1])

        if self.target_shape is not None:
            r[-1] = crop_or_pad_fun(r[-1], self.target_shape)
        return r

    def forward_simple(self, x):
        for layer in self.layers:
            x = layer(x)

        if self.squash_function is not None:
            x = self.squash_function(x)

        if self.target_shape is not None:
            x = crop_or_pad_fun(x, self.target_shape)

        return x

    def forward(self, x):
        return self.forward_simple(x)
