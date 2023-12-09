from typing import Sequence, Union, Optional, Dict, Callable, Tuple, List

import torch
import torch.nn as nn
from .layer_config import NormType, LayerConfig, default_layer_config
from .convs_transpose import ConvsTransposeBase
from .crop_or_pad import crop_or_pad_fun
from ..layers.convs import ModuleWithIntermediate, ConvsBase
from ..basic_typing import IntTupleList, Activation, ConvKernels, ConvStrides, PoolingSizes


class AutoencoderConvolutional(nn.Module, ModuleWithIntermediate):
    """
    Convolutional autoencoder

    Examples:
        Create an encoder taking 1 channel with [4, 8, 16] filters and a decoder taking as input 16 channels
        of 4x4 with [8, 4, 1] filters:
        >>> model = AutoencoderConvolutional(2, 1, [4, 8, 16], [8, 4, 1])
    """
    def __init__(
            self,
            dimensionality: int,
            input_channels: int,
            encoder_channels: Sequence[int],
            decoder_channels: Sequence[int],
            convolution_kernels: ConvKernels = 5,
            encoder_strides: Union[ConvStrides] = 1,
            decoder_strides: Union[ConvStrides] = 2,
            pooling_size: Optional[PoolingSizes] = 2,
            convolution_repeats: Union[int, Sequence[int]] = 1,
            activation: Optional[Activation] = nn.ReLU,
            dropout_probability: Optional[float] = None,
            norm_type: NormType = NormType.BatchNorm,
            norm_kwargs: Dict = {},
            activation_kwargs: Dict = {},
            last_layer_is_output: bool = False,
            force_decoded_size_same_as_input: bool = True,
            squash_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            config: LayerConfig = default_layer_config(dimensionality=None)):

        super().__init__()
        self.force_decoded_size_same_as_input = force_decoded_size_same_as_input

        self.encoder = ConvsBase(
            dimensionality=dimensionality,
            input_channels=input_channels,
            channels=encoder_channels,
            convolution_kernels=convolution_kernels,
            strides=encoder_strides,
            pooling_size=pooling_size,
            convolution_repeats=convolution_repeats,
            activation=activation,
            dropout_probability=dropout_probability,
            norm_type=norm_type,
            activation_kwargs=activation_kwargs,
            norm_kwargs=norm_kwargs,
            last_layer_is_output=False,
            config=config
        )

        self.decoder = ConvsTransposeBase(
            dimensionality=dimensionality,
            input_channels=encoder_channels[-1],
            channels=decoder_channels,
            strides=decoder_strides,
            activation=activation,
            dropout_probability=dropout_probability,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            activation_kwargs=activation_kwargs,
            last_layer_is_output=last_layer_is_output,
            squash_function=squash_function,
            config=config
        )

    def forward_simple(self, x: torch.Tensor) -> torch.Tensor:
        encoded_x = self.encoder(x)
        return encoded_x

    def forward_with_intermediate(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(kwargs) == 0, f'unsupported arguments={kwargs}'

        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)

        if self.force_decoded_size_same_as_input:
            decoded_x = crop_or_pad_fun(decoded_x, x.shape[2:])

        return encoded_x, decoded_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_simple(x)
