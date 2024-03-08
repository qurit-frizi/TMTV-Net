import copy
from functools import partial
from typing import Sequence, Optional, Union, Any, List
import torch
import torch.nn as nn
from typing_extensions import Literal

from ..basic_typing import ShapeNCX, KernelSize
from .convs import ModuleWithIntermediate
from ..utils import upsample
from .blocks import BlockConvNormActivation, ConvBlockType
from .layer_config import LayerConfig, default_layer_config
from .unet_base import UpType, MiddleType, LatentConv
from ..train.utilities import get_device


class BlockUpResizeDeconvSkipConv(nn.Module):
    """
    Reshape the bottom features to match the transverse feature using linear interpolation
    and apply a block on the concatenated (transverse, resampled bottom features)
    """
    def __init__(
            self,
            layer_config: LayerConfig,
            bloc_level: int,
            skip_channels: int,
            input_channels: int,
            output_channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            resize_mode: Literal['linear', 'nearest'] = 'linear',
            block=BlockConvNormActivation):
        super().__init__()

        self.bloc_level = bloc_level
        self.block = block(
            layer_config,
            input_channels=input_channels + skip_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
        )

        self.skip_channels = skip_channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.resize_mode = resize_mode

    def forward(self, skip: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        assert skip.shape[1] == self.skip_channels
        assert previous.shape[1] == self.input_channels

        if previous.shape[2:] != skip.shape[2:]:
            previous = upsample(previous, skip.shape[2:], mode=self.resize_mode)

        x = torch.cat([skip, previous], dim=1)
        x = self.block(x)
        return x


class BackboneDecoder(nn.Module, ModuleWithIntermediate):
    """
    U-net like model with backbone used as encoder.

    Examples:
        >>> encoder = layers.convs_3d(1, channels=[64, 128, 256])
        >>> segmenter = layers.BackboneDecoder([256, 128, 64], 3, encoder, [0, 1, 2], [1, 1, 64, 64, 64])
    """
    def __init__(self,
                 decoding_channels: Sequence[int],
                 output_channels: int,
                 backbone: ModuleWithIntermediate,
                 backbone_transverse_connections: Sequence[int],
                 backbone_input_shape: ShapeNCX,
                 *,
                 up_block_fn: UpType = BlockUpResizeDeconvSkipConv,
                 middle_block_fn: MiddleType = partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=5)),
                 output_block_fn: ConvBlockType = BlockConvNormActivation,
                 latent_channels: Optional[int] = None,
                 kernel_size: Optional[int] = 3,
                 strides: Union[int, Sequence[int]] = 2,
                 activation: Optional[Any] = None,
                 config: LayerConfig = default_layer_config(dimensionality=None),
                 ):
        """
        Args:
            decoding_channels: the channels used for the decoding. Must have the same length
                as ``backbone_transverse_connections``. From bottom layer to top layer (i.e., reverse of the
                encoder)
            output_channels: the number of desired output channels
            backbone: the module used to encode the input
            backbone_transverse_connections: specify the intermediate layers of the backbone to be used
                as transverse connection
            backbone_input_shape: an example of shape accepted by the backbone. This will be used to
                calculate the number of filters of each lateral connection
            up_block_fn: defines how to decode a feature using transverse feature and previsouly decoded feature
            middle_block_fn: middle block to bridge the encoding and decoding, as well as potentially
                managing given embedding
            output_block_fn: final block to be applied
            latent_channels: number of latent channels to be used
            kernel_size: override the given kernel's configuration
            strides: override the given stride's configuration
            activation: override the given activation's configuration
            config: the configuration to be used for the decoding path
        """
        super().__init__()
        self.decoding_channels = decoding_channels
        self.backbone_input_shape = backbone_input_shape

        backbone_device = get_device(backbone)
        backbone_intermediate = backbone.forward_with_intermediate(
            torch.zeros(tuple(backbone_input_shape), dtype=torch.float32, device=backbone_device),
        )

        self.backbone = backbone
        self.backbone_output_channels = backbone_intermediate[-1].shape[1]
        self.backbone_transverse_connections = list(backbone_transverse_connections)
        assert max(self.backbone_transverse_connections) < len(backbone_intermediate), \
            f'only {len(backbone_intermediate)} backbone intermediate layers,' \
            f'but lateral connection={len(backbone_intermediate)}'
        assert len(self.backbone_transverse_connections) == len(decoding_channels)

        # make a new config in case this was shared among multiple
        # modules
        config = copy.copy(config)
        dim = len(backbone_input_shape) - 2
        config.set_dim(dim)
        if kernel_size is not None:
            config.conv_kwargs['kernel_size'] = kernel_size
            config.deconv_kwargs['kernel_size'] = kernel_size
        if activation is not None:
            config.activation = activation

        self.activation = activation
        self.strides = strides
        self.kernel_size = kernel_size
        self.latent_channels = latent_channels
        self.up_block_fn = up_block_fn
        self.output_channels = output_channels

        last_encoder_channels = backbone_intermediate[-1].shape[1]

        self.middle_block = middle_block_fn(
            layer_config=config,
            bloc_level=-1,
            input_channels=last_encoder_channels,
            output_channels=last_encoder_channels,
            latent_channels=latent_channels
        )

        decoder_blocks = nn.ModuleList()
        last_channels = last_encoder_channels
        assert len(decoding_channels) > 0
        block_n = 0
        for block_n, channels in enumerate(decoding_channels):
            intermediate_n = self.backbone_transverse_connections[-(block_n+1)]
            skip_channels = backbone_intermediate[intermediate_n].shape[1]
            block = up_block_fn(
                layer_config=config,
                bloc_level=block_n,
                skip_channels=skip_channels,
                input_channels=last_channels,
                output_channels=channels,
            )
            last_channels = channels
            decoder_blocks.append(block)
        self.decoder_blocks = decoder_blocks

        # recover the final shape from the input
        self.final_decoder_block = up_block_fn(
            layer_config=config,
            bloc_level=block_n + 1,
            skip_channels=backbone_input_shape[1],
            input_channels=last_channels,
            output_channels=decoding_channels[-1],
        )

        # here we need to have a special output block: this
        # is because we do NOT want to add the activation for the
        # result layer (i.e., often, the output is normalized [-1, 1]
        # and we would discard the negative portion)
        config = copy.copy(config)
        config.norm = None
        config.activation = None
        config.dropout = None

        self.output_block = output_block_fn(
            config=config,
            input_channels=decoding_channels[-1],
            output_channels=self.output_channels,
        )

    def forward_with_intermediate(self, x: torch.Tensor, latent: Optional[torch.Tensor] = None, **kwargs) -> List[torch.Tensor]:
        assert len(kwargs) == 0
        intermediates = self.backbone.forward_with_intermediate(x)
        last = self.middle_block(intermediates[-1], latent=latent)

        decoder_intermediates = [last]
        for b_n, b in enumerate(self.decoder_blocks):
            transverse = intermediates[self.backbone_transverse_connections[-(b_n + 1)]]
            last = b(transverse, last)
            decoder_intermediates.append(last)

        last = self.final_decoder_block(x, last)
        decoder_intermediates.append(last)

        last = self.output_block(last)
        decoder_intermediates.append(last)

        return decoder_intermediates

    def forward(self, x: torch.Tensor, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: the input image
            latent: a latent variable appended by the middle block
        """
        intermediates = self.forward_with_intermediate(x, latent)
        assert len(intermediates)
        return intermediates[-1]
