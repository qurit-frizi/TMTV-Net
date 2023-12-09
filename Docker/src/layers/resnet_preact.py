import copy
from functools import partial
from typing import Sequence, List, Optional

import torch
import torch.nn as nn

from .convs import ModuleWithIntermediate
from .layer_config import PoolType, default_layer_config, LayerConfig
from ..basic_typing import Stride
from .blocks import BlockResPreAct, BlockConvNormActivation, BlockPoolClassifier


class PreActResNet(nn.Module, ModuleWithIntermediate):
    """
    Pre-activation Resnet model

    Examples:
        >>> pre_act_resnet18 = PreActResNet(2, 3, 10)
        >>> c = pre_act_resnet18(torch.zeros(10, 3, 32, 32))

    References:
        [1] https://arxiv.org/pdf/1603.05027.pdf

    Notes:
        The default pooling kernel has been adapted to fit CIFAR10 rather than
        imagenet image size (kernel size=4 instead of 7)
    """
    def __init__(
            self,
            dimensionality: int,
            input_channels: int,
            output_channels: Optional[int],
            *,
            block=BlockResPreAct,
            num_blocks: Sequence[int] = (2, 2, 2, 2),
            strides: Sequence[Stride] = (1, 2, 2, 2),
            channels: Sequence[int] = (64, 128, 256, 512),
            init_block_fn=partial(BlockConvNormActivation, kernel_size=3, stride=1, bias=False),
            output_block_fn=BlockPoolClassifier,
            config: LayerConfig = default_layer_config(dimensionality=None, pool_type=PoolType.AvgPool)):

        super().__init__()
        if isinstance(strides, int):
            strides = [strides] * len(num_blocks)

        assert len(num_blocks) == len(strides)
        assert len(num_blocks) == len(channels)

        self.channels = channels
        self.strides = strides
        config = copy.copy(config)
        config.set_dim(dimensionality)

        self.conv1 = init_block_fn(config, input_channels, channels[0])
        blocks = nn.ModuleList()
        self.in_planes = channels[0]
        for i in range(len(num_blocks)):
            b = self._make_layer(config, block, channels[i], num_blocks[i], stride=strides[i])
            blocks.append(b)
        self.blocks = blocks

        if output_channels is not None:
            self.classifier = output_block_fn(config, channels[-1], output_channels)
        else:
            self.classifier = None

    def _make_layer(self, config, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            b = block(config, self.in_planes, planes, stride)
            layers.append(b)
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_with_intermediate(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        assert len(kwargs) == 0, f'unsupported arguments={kwargs}'

        intermediates = []
        out = self.conv1(x)
        intermediates.append(out)
        for block in self.blocks:
            out = block(out)
            intermediates.append(out)

        return intermediates

    def forward(self, x):
        intermediates = self.forward_with_intermediate(x)
        if self.classifier is not None:
            out = self.classifier(intermediates[-1])
            return out
        return intermediates[-1]


PreActResNet18 = partial(
    PreActResNet,
    dimensionality=2,
    input_channels=3,
    output_channels=10,
    block=BlockResPreAct,
    num_blocks=[2, 2, 2, 2]
)

PreActResNet34 = partial(
    PreActResNet,
    dimensionality=2,
    input_channels=3,
    output_channels=10,
    block=BlockResPreAct,
    num_blocks=[3, 4, 6, 3]
)

