import copy
import math
from functools import partial
from typing import Optional, Sequence
import torch

import torch.nn as nn
from .flatten import Flatten
from .convs import ModuleWithIntermediate
from ..basic_typing import KernelSize, Stride, TorchTensorNCX, ModuleCreator
from .layer_config import LayerConfig, default_layer_config
from .blocks import BlockConvNormActivation, BlockSqueezeExcite
from ..train.compatibility import Swish, Identity


class DropSample(nn.Module):
    """
    Drops each sample in `x` with probability p during training
    """
    def __init__(self, p: float = 0):
        """
        Args:
            p: probability of dropping a connection
        """
        super().__init__()
        self.p = p

    def forward(self, x: TorchTensorNCX) -> TorchTensorNCX:
        if (not self.p) or (not self.training):
            return x

        batch_size = len(x)
        random_tensor = torch.rand([batch_size] + [1] * (len(x.shape) - 1), device=x.device, dtype=x.dtype)
        bit_mask = self.p < random_tensor
        return x / (1 - self.p) * bit_mask


class MBConvN(nn.Module):
    """
    MBConv with an expansion factor of N, plus squeeze-and-excitation

    References:
        [1] "Searching for MobileNetV3", https://arxiv.org/pdf/1905.02244.pdf
    """
    def __init__(self,
                 config: LayerConfig,
                 input_channels: int,
                 output_channels: int,
                 *,
                 expansion_factor: int,
                 kernel_size: Optional[KernelSize] = 3,
                 stride: Optional[Stride] = None,
                 r: int = 24,
                 p: float = 0):

        super().__init__()

        expanded = expansion_factor * input_channels
        self.skip_connection = (input_channels == output_channels) and (stride == 1)

        self.expand_pw = Identity() if (expansion_factor == 1) else BlockConvNormActivation(
            config=config,
            input_channels=input_channels,
            output_channels=expanded,
            kernel_size=1,
            bias=False
        )

        self.depthwise = BlockConvNormActivation(
            config=config,
            input_channels=expanded,
            output_channels=expanded,
            kernel_size=kernel_size,
            stride=stride,
            groups=expanded,
            bias=False
        )
        self.se = BlockSqueezeExcite(config=config, input_channels=expanded, r=r)

        config_reduce = copy.copy(config)
        config_reduce.activation = None
        self.reduce_pw = BlockConvNormActivation(
            config=config_reduce,
            input_channels=expanded,
            output_channels=output_channels,
            kernel_size=1,
            bias=False
        )

        self.dropsample = DropSample(p)

    def forward(self, x):
        residual = x

        x = self.expand_pw(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.reduce_pw(x)

        if self.skip_connection:
            x = self.dropsample(x)
            x = x + residual

        return x


MBConv1 = partial(MBConvN, expansion_factor=1)

MBConv6 = partial(MBConvN, expansion_factor=6)


def create_stage(
        config,
        input_channels,
        output_channels,
        num_layers,
        layer_type=MBConv6,
        kernel_size=3,
        stride=1,
        r=24,
        p=0):

    """
    Creates a Sequential consisting of [num_layers] layer_type
    """
    layers = [layer_type(
        config,
        input_channels,
        output_channels,
        kernel_size=kernel_size,
        stride=stride,
        r=r,
        p=p
    )]

    layers += [layer_type(
        config,
        output_channels,
        output_channels,
        kernel_size=kernel_size,
        r=r,
        p=p) for _ in range(num_layers-1)
    ]

    return nn.Sequential(*layers)


def scale_width(w, w_factor):
    """
    Scales width given a scale factor
    """
    w *= w_factor
    new_w = (int(w + 4) // 8) * 8
    new_w = max(8, new_w)
    if new_w < 0.9 * w:
        new_w += 8
    return int(new_w)


class EfficientNet(nn.Module, ModuleWithIntermediate):
    """
    Generic EfficientNet that takes in the width and depth scale factors and scales accordingly.

    With default settings, it operates on 224x224 images.

    References:
        [1] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
        https://arxiv.org/abs/1905.11946
    """
    def __init__(self,
                 dimensionality: int,
                 input_channels: int,
                 output_channels: int,
                 *,
                 w_factor: float = 1,
                 d_factor: float = 1,
                 activation: Optional[ModuleCreator] = Swish,
                 base_widths=(
                     (32, 16),
                     (16, 24),
                     (24, 40),
                     (40, 80),
                     (80, 112),
                     (112, 192),
                     (192, 320),
                     (320, 1280)
                 ),
                 base_depths=(1, 2, 2, 3, 3, 4, 1),
                 kernel_sizes=(3, 3, 5, 3, 5, 5, 3),
                 strides=(1, 2, 2, 2, 1, 2, 1),
                 config: LayerConfig = default_layer_config(dimensionality=None)):
        super().__init__()

        assert len(base_widths) - 1 == len(base_depths)
        assert len(base_widths) - 1 == len(kernel_sizes)
        assert len(base_widths) - 1 == len(strides)

        self.dimensionality = dimensionality
        config = copy.copy(config)
        config.set_dim(dimensionality)

        if activation is not None:
            config.activation = activation

        # default parameters for B0
        scaled_widths = [(scale_width(w[0], w_factor), scale_width(w[1], w_factor)) for w in base_widths]
        scaled_depths = [math.ceil(d_factor * d) for d in base_depths]
        ps = [0, 0.029, 0.057, 0.086, 0.114, 0.143, 0.171]

        self.stem = BlockConvNormActivation(
            config=config,
            input_channels=input_channels,
            output_channels=scaled_widths[0][0],
            kernel_size=3,
            stride=2,
            bias=False
        )

        stages = nn.ModuleList()
        for i in range(7):
            i_ch = scaled_widths[i][0]
            o_ch = scaled_widths[i][1]

            layer_type = MBConv1 if (i == 0) else MBConv6
            r = 4 if (i == 0) else 24
            stage = create_stage(
                config=config,
                input_channels=i_ch,
                output_channels=o_ch,
                num_layers=scaled_depths[i],
                layer_type=layer_type,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                r=r,
                p=ps[i]
            )

            stages.append(stage)

        self.stages = stages

        self.pre_head = BlockConvNormActivation(
            config=config,
            input_channels=scaled_widths[-1][0],
            output_channels=scaled_widths[-1][1],
            kernel_size=1
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(scaled_widths[-1][1], output_channels)
        )

    def forward_with_intermediate(self, x: torch.Tensor, **kwargs) -> Sequence[torch.Tensor]:
        assert len(kwargs) == 0, f'unexpected arguments={kwargs.keys()}'
        intermediates = []

        x = self.stem(x)
        intermediates.append(x)
        for stage in self.stages:
            x = stage(x)
            intermediates.append(x)
        x = self.pre_head(x)
        intermediates.append(x)

        return intermediates

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediate(x)[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.head(x)
        return x


EfficientNetB0 = partial(EfficientNet, w_factor=1, d_factor=1)
EfficientNetB1 = partial(EfficientNet, w_factor=1, d_factor=1.1)
EfficientNetB2 = partial(EfficientNet, w_factor=1.1, d_factor=1.2)
EfficientNetB3 = partial(EfficientNet, w_factor=1.2, d_factor=1.4)
EfficientNetB5 = partial(EfficientNet, w_factor=1.6, d_factor=2.2)
EfficientNetB6 = partial(EfficientNet, w_factor=1.8, d_factor=2.6)
EfficientNetB7 = partial(EfficientNet, w_factor=2, d_factor=3.1)
