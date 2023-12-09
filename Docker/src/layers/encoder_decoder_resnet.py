import collections

from functools import partial

import copy
from typing import Sequence, Optional, Any, List

import torch.nn as nn
from ..basic_typing import ConvStrides, Activation, TorchTensorNCX
from .layer_config import LayerConfig, default_layer_config, NormType
from .blocks import ConvBlockType, BlockConvNormActivation, \
    ConvTransposeBlockType, BlockDeconvNormActivation, BlockRes


class EncoderDecoderResnet(nn.Module):
    def __init__(
            self,
            dimensionality: int,
            input_channels: int,
            output_channels: int,
            encoding_channels: Sequence[int],
            decoding_channels: Sequence[int],
            *,
            nb_residual_blocks: int = 9,
            convolution_kernel: int = 3,
            encoding_strides: ConvStrides = 2,
            decoding_strides: ConvStrides = 2,
            activation: Optional[Activation] = None,
            encoding_block: ConvBlockType = BlockConvNormActivation,
            decoding_block: ConvTransposeBlockType = BlockDeconvNormActivation,
            init_block=partial(BlockConvNormActivation, kernel_size=7),
            middle_block: Any = BlockRes,
            out_block=partial(BlockConvNormActivation, kernel_size=7),
            config: LayerConfig = default_layer_config(
                conv_kwargs={'padding': 'same', 'bias': False, 'padding_mode': 'reflect'},
                deconv_kwargs={'padding': 'same', 'bias': False},
                norm_type=NormType.BatchNorm,
                activation=nn.ReLU)):
        super().__init__()

        #
        # encoding path
        #
        nb_convs = len(encoding_channels)
        if not isinstance(encoding_strides, collections.Sequence):
            encoding_strides = [encoding_strides] * nb_convs
        assert len(encoding_strides) == nb_convs

        config_enc = copy.copy(config)
        config_enc.set_dim(dimensionality)
        if activation is not None:
            config_enc.activation = activation

        intermediate_channels = encoding_channels[0] // 2

        cur = input_channels
        prev = input_channels
        if init_block is not None:
            self.initial = init_block(
                config=config_enc,
                input_channels=input_channels,
                output_channels=intermediate_channels,
                stride=1)
            prev = intermediate_channels

        self.encoders = nn.ModuleList()  # do NOT store in a list, else the layer parameters will not be found!
        for cur, stride in zip(encoding_channels, encoding_strides):
            assert isinstance(stride, (int, tuple))
            block = encoding_block(config_enc, prev, cur, kernel_size=convolution_kernel, stride=stride)
            prev = cur
            self.encoders.append(block)

        self.residuals = nn.ModuleList()  # do NOT store in a list, else the layer parameters will not be found!
        for n in range(nb_residual_blocks):
            self.residuals.append(middle_block(config=config_enc, input_channels=cur, kernel_size=convolution_kernel))

        #
        # decoding path
        #

        config_dec = copy.copy(config)
        config_dec.set_dim(dimensionality)
        if activation is not None:
            config_dec.activation = activation

        nb_convs = len(decoding_channels)
        if not isinstance(decoding_strides, collections.Sequence):
            decoding_strides = [decoding_strides] * nb_convs
        assert len(decoding_strides) == nb_convs

        self.decoders = nn.ModuleList()  # do NOT store in a list, else the layer parameters will not be found!
        for cur, stride in zip(decoding_channels, encoding_strides):
            assert isinstance(stride, int), 'todo: tuple not handled'
            block = decoding_block(
                config_dec,
                prev,
                cur,
                kernel_size=convolution_kernel,
                stride=stride,
                output_padding=stride - 1)
            prev = cur
            self.decoders.append(block)

        config_dec.activation = None  # no activation for the last layer
        config_dec.norm = None
        self.out = out_block(config_dec, prev, output_channels)

    def forward(self, x: TorchTensorNCX) -> TorchTensorNCX:
        intermediates = self.forward_with_intermediate(x)
        return intermediates[-1]

    def forward_with_intermediate(self, x: TorchTensorNCX) -> List[TorchTensorNCX]:
        intermediates = []

        x = self.initial(x)
        for encoder in self.encoders:
            x = encoder(x)
            intermediates.append(x)
        for residual in self.residuals:
            x = residual(x)
            intermediates.append(x)
        for decoder in self.decoders:
            x = decoder(x)
            intermediates.append(x)
        x = self.out(x)
        intermediates.append(x)
        return intermediates
