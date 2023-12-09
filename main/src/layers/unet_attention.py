from functools import partial
from typing import Callable, Sequence
import torch
import torch.nn as nn
from .blocks import BlockConv, BlockMerge, BlockUpDeconvSkipConv, BlockUpsampleNnConvNormActivation

from .layer_config import LayerConfig
from .unet_base import Up, UNetBase


class BlockAttention(nn.Module):
    """
    Attention UNet style of attention.

    See:
        "Attention U-Net: Learning Where to Look for the Pancreas", https://arxiv.org/pdf/1804.03999.pdf

    Args:
        nn (_type_): _description_
    """
    def __init__(
            self,
            config: LayerConfig,
            gating_channels: int,
            input_channels: int,
            intermediate_channels: int):
        super().__init__()

        W_g = [
            BlockConv(config=config, input_channels=gating_channels, output_channels=intermediate_channels, kernel_size=1)
        ]
        if config.norm is not None:
            W_g.append(config.norm(num_features=intermediate_channels, **config.norm_kwargs))

        W_x = [
            BlockConv(config=config, input_channels=input_channels, output_channels=intermediate_channels, kernel_size=1)
        ]
        if config.norm is not None:
            W_x.append(config.norm(num_features=intermediate_channels, **config.norm_kwargs))

        psi = [
            BlockConv(config=config, input_channels=intermediate_channels, output_channels=1, kernel_size=1)
        ]
        if config.norm is not None:
            psi.append(config.norm(num_features=1, **config.norm_kwargs))
        psi.append(nn.Sigmoid())

        if config.activation is not None:
            self.activation = config.activation(**config.activation_kwargs)
        else:
            self.activation = nn.Identity()

        self.W_x = nn.Sequential(*W_x)
        self.W_g = nn.Sequential(*W_g)
        self.psi = nn.Sequential(*psi)
        self.input_channels = input_channels
        self.gating_channels = gating_channels
        self.intermediate_channels = intermediate_channels

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert g.shape[2:] == x.shape[2:], f'invalid shape. Must be the same. Got g={g.shape}, x={x.shape}'
        assert g.shape[1] == self.gating_channels, f'Got={g.shape[1]}, expected={self.gating_channels}'
        assert x.shape[1] == self.input_channels, f'Got={x.shape[1]}, expected={self.input_channels}'

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class MergeBlockAttention_Gating_Input(nn.Module):
    def __init__(
            self, 
            config: LayerConfig,
            layer_channels: Sequence[int], 
            attention_block_fn=BlockAttention,
            num_intermediate_fn: Callable[[Sequence[int]], int] = lambda layer_channels: max(8, layer_channels[0] // 2)) -> None:
        super().__init__()
        assert len(layer_channels) == 2
        self.output_channels = sum(layer_channels)
        self.attention_block = attention_block_fn(
            config=config,
            gating_channels=layer_channels[0],
            input_channels=layer_channels[1],
            intermediate_channels=num_intermediate_fn(layer_channels)
        )
    
    def get_output_channels(self):
        return self.output_channels

    def forward(self, layers: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(layers) == 2
        gating_layer = layers[0]
        x = layers[1]
        attention = self.attention_block(gating_layer, x)
        return torch.concat([attention, x], dim=1)


UNetAttention = partial(
    UNetBase,
    up_block_fn=partial(Up, block=partial(
        BlockUpDeconvSkipConv, 
        merge_layer_fn=MergeBlockAttention_Gating_Input, 
        deconv_block=BlockUpsampleNnConvNormActivation)
    )
)
