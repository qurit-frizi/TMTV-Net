import collections
import warnings
from numbers import Number

import torch

# from ..utils import flatten
from basic_typing import TorchTensorNCX, Padding, KernelSize, Stride
# from .utils import div_shape
from .layer_config import LayerConfig
import torch.nn as nn
from typing import Union, Dict, Optional, Sequence, List
from typing_extensions import Protocol, Literal  # backward compatibility for python 3.6-3.7
import copy
import numpy as np


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
    
class BlockPool(nn.Module):
    def __init__(
            self,
            config: LayerConfig,
            kernel_size: Optional[KernelSize] = 2):

        super().__init__()

        pool_kwargs = copy.copy(config.pool_kwargs)
        if kernel_size is not None:
            pool_kwargs['kernel_size'] = kernel_size

        assert config.pool is not None
        self.op = config.pool(**pool_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


def _posprocess_padding(config: LayerConfig, conv_kwargs: Dict, ops: List[nn.Module]) -> None:
    """
    Note:
        conv_kwargs will be modified in-place. Make a copy before!
    """
    padding_same = False
    padding = conv_kwargs.get('padding')
    if padding is not None and padding == 'same':
        padding_same = True
        kernel_size = conv_kwargs.get('kernel_size')
        assert kernel_size is not None, 'missing argument `kernel_size` in convolutional arguments!'
        padding = div_shape(kernel_size)
        conv_kwargs['padding'] = padding

    # if the padding is even, it needs to be asymmetric: one side has less padding
    # than the other. Here we need to add an additional ops to perform the padding
    # since we can't do it in the convolution
    if padding is not None:
        assert config.ops.dim is not None
        if isinstance(padding, int):
            padding = [padding] * config.ops.dim
        else:
            assert isinstance(padding, collections.Sequence)

        kernel_size = conv_kwargs.get('kernel_size')
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * config.ops.dim
        assert kernel_size is not None
        assert len(kernel_size) == len(padding)

        is_even = 1 - np.mod(kernel_size, 2)
        if padding_same and any(is_even):
            # make sure we don't have conflicting padding info:
            # here we do not support all the possible options: if kernel is even
            # no problem but if we use even kernel, <nn.functional.pad> doesn't support all combinations.
            # this will need to be revisited when we have more support.
            padding_mode = conv_kwargs.get('padding_mode')
            if padding_mode is not None:
                if padding_mode != 'zeros':
                    warnings.warn(f'padding mode={padding_mode} is not supported with even padding!')

            #  there is even padding, add special padding op
            full_padding = []
            for k, p in zip(kernel_size, padding):
                left = k // 2
                right = p - left // 2
                # we need to reverse the dimensions, so reverse also the left/right components
                # and then reverse the whole sequence
                full_padding += [right, left]

            assert config.ops.constant_padding is not None
            ops.append(config.ops.constant_padding(padding=tuple(full_padding[::-1]), value=0))
            # we have explicitly added padding, so now set to
            # convolution padding to none
            conv_kwargs['padding'] = 0

    # handle differences with pytorch <= 1.0
    # where the convolution doesn't have argument `padding_mode`
    version = torch.__version__[:3]
    if 'padding_mode' in conv_kwargs:
        if version == '1.0':
            warnings.warn('convolution doesn\'t have padding_mode as argument in  pytorch <= 1.0. Argument is deleted!')
            del conv_kwargs['padding_mode']


class BlockConv(nn.Module):
    def __init__(self,
                 config: LayerConfig,
                 input_channels: int,
                 output_channels: int,
                 *,
                 kernel_size: Optional[KernelSize] = None,
                 padding: Optional[Padding] = None,
                 stride: Optional[Stride] = None,
                 padding_mode: Optional[str] = None,
                 groups: int = 1,
                 bias: Optional[bool] = None):

        super().__init__()

        # local override of the default config
        conv_kwargs = copy.copy(config.conv_kwargs)
        if kernel_size is not None:
            conv_kwargs['kernel_size'] = kernel_size
        if padding is not None:
            conv_kwargs['padding'] = padding
        if stride is not None:
            conv_kwargs['stride'] = stride
        if padding_mode is not None:
            conv_kwargs['padding_mode'] = padding_mode
        if bias is not None:
            conv_kwargs['bias'] = bias

        ops: List[nn.Module] = []
        _posprocess_padding(config, conv_kwargs, ops)

        assert config.conv is not None
        conv = config.conv(
            in_channels=input_channels,
            out_channels=output_channels,
            groups=groups,
            **conv_kwargs)
        ops.append(conv)

        self.ops = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


class BlockConvNormActivation(nn.Module):
    def __init__(
            self,
            config: LayerConfig,
            input_channels: int,
            output_channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            stride: Optional[Stride] = None,
            padding_mode: Optional[str] = None,
            groups: int = 1,
            bias: Optional[bool] = None):

        super().__init__()

        conv = BlockConv(
            config=config,
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            padding_mode=padding_mode,
            groups=groups,
            bias=bias
        )

        ops: List[nn.Module] = [conv]

        if config.norm is not None:
            ops.append(config.norm(num_features=output_channels, **config.norm_kwargs))

        if config.activation is not None:
            ops.append(config.activation(**config.activation_kwargs))

        self.ops = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


class BlockDeconvNormActivation(nn.Module):
    def __init__(
            self,
            config: LayerConfig,
            input_channels: int,
            output_channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            output_padding: Optional[Union[int, Sequence[int]]] = None,
            stride: Optional[Stride] = None,
            padding_mode: Optional[str] = None):

        super().__init__()

        # local override of the default config
        deconv_kwargs = copy.copy(config.deconv_kwargs)
        if kernel_size is not None:
            deconv_kwargs['kernel_size'] = kernel_size
        if padding is not None:
            deconv_kwargs['padding'] = padding
        if stride is not None:
            deconv_kwargs['stride'] = stride
        if output_padding is not None:
            deconv_kwargs['output_padding'] = output_padding
        if padding_mode is not None:
            deconv_kwargs['padding_mode'] = padding_mode

        ops: List[nn.Module] = []
        _posprocess_padding(config, deconv_kwargs, ops)

        assert config.deconv is not None
        deconv = config.deconv(
            in_channels=input_channels,
            out_channels=output_channels,
            **deconv_kwargs)
        ops.append(deconv)

        if config.norm is not None:
            ops.append(config.norm(num_features=output_channels, **config.norm_kwargs))

        if config.activation is not None:
            ops.append(config.activation(**config.activation_kwargs))
        self.ops = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


class BlockUpsampleNnConvNormActivation(nn.Module):
    """
    The standard approach of producing images with deconvolution — despite its successes! — 
    has some conceptually simple issues that lead to checkerboard artifacts in produced images.

    This is an alternative block using nearest neighbor upsampling + convolution.

    See Also:
        https://distill.pub/2016/deconv-checkerboard/

    """
    def __init__(
            self,
            config: LayerConfig,
            input_channels: int,
            output_channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            # unused, just to follow the other upsampling interface
            output_padding: Optional[Union[int, Sequence[int]]] = None,
            stride: Optional[Stride] = None,
            padding_mode: Optional[str] = None):

        super().__init__()

        # local override of the default config
        conv_kwargs = copy.copy(config.conv_kwargs)
        if kernel_size is not None:
            conv_kwargs['kernel_size'] = kernel_size
        if padding is not None:
            conv_kwargs['padding'] = padding
        # stride is used in the upsampling
        if padding_mode is not None:
            conv_kwargs['padding_mode'] = padding_mode

        if stride is None:
            stride = config.deconv_kwargs.get('stride')

        assert stride is not None

        ops = []
        stride_np = np.asarray(stride)
        if (isinstance(stride, Number) and stride != 1) or (stride_np.max() != 1 or stride_np.min() != 1):
            # if stride is 1, don't upsample!
            assert config.ops.upsample_fn is not None
            ops.append(config.ops.upsample_fn(scale_factor=stride))

        _posprocess_padding(config, conv_kwargs, ops)
        assert config.conv is not None
        ops.append(config.conv(in_channels=input_channels,
                               out_channels=output_channels,
                               **conv_kwargs))

        if config.norm is not None:
            ops.append(config.norm(num_features=output_channels, **config.norm_kwargs))

        if config.activation is not None:
            ops.append(config.activation(**config.activation_kwargs))
        self.ops = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


class BlockMerge(nn.Module):
    """
    Merge multiple layers (e.g., concatenate, sum...)
    """
    def __init__(self, config: LayerConfig, layer_channels: Sequence[int], mode: Literal['concatenation', 'sum'] = 'concatenation') -> None:
        super().__init__()
        self.layer_channels = layer_channels
        self.mode = mode
        assert len(layer_channels) > 0
        if self.mode == 'sum':
            assert len(set(layer_channels)) == 1, 'all layers must have the same number of channels to sum!'
            self.output_channels = self.layer_channels[0]
        elif self.mode == 'concatenation':
            self.output_channels = sum(self.layer_channels)
        else:
            raise ValueError(f'unsupported mode={mode}')

    def get_output_channels(self):
        return self.output_channels

    def forward(self, layers: Sequence[torch.Tensor]):
        assert len(layers) == len(self.layer_channels)
        if self.mode == 'sum':
            return torch.add(*layers)
        elif self.mode == 'concatenation':
            return torch.concat(layers, dim=1)
        else:
            raise ValueError(f'unsupported mode={self.mode}')


class BlockUpDeconvSkipConv(nn.Module):
    def __init__(
            self,
            config: LayerConfig,
            skip_channels: int,
            input_channels: int,
            output_channels: int,
            *,
            nb_repeats: int = 1,
            kernel_size: Optional[KernelSize] = None,
            deconv_kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            output_padding: Optional[Union[int, Sequence[int]]] = None,
            deconv_block=BlockDeconvNormActivation,
            stride: Optional[Stride] = None,
            merge_layer_fn=BlockMerge):
        super().__init__()

        self.merge_layer = merge_layer_fn(config=config, layer_channels=[skip_channels, output_channels])

        if deconv_kernel_size is None:
            deconv_kernel_size = kernel_size

        self.ops_deconv = deconv_block(
            config,
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=deconv_kernel_size,
            padding=padding,
            output_padding=output_padding,
            stride=stride
        )

        convs = [
            BlockConvNormActivation(
                config,
                input_channels=self.merge_layer.get_output_channels(),
                output_channels=output_channels,
                kernel_size=kernel_size,
                stride=1
            )
        ]

        for _ in range(1, nb_repeats):
            conv = BlockConvNormActivation(
                config,
                input_channels=output_channels,
                output_channels=output_channels,
                kernel_size=kernel_size,
                stride=1
            )
            convs.append(conv)

        self.ops_conv: Union[nn.Sequential, BlockConvNormActivation]
        if len(convs) == 1:
            self.ops_conv = convs[0]
        else:
            self.ops_conv = nn.Sequential(*convs)

        self.skip_channels = skip_channels
        self.input_channels = input_channels
        self.output_channels = output_channels

    def forward(self, skip: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        assert skip.shape[1] == self.skip_channels
        assert previous.shape[1] == self.input_channels
        x = self.ops_deconv(previous)
        assert x.shape[2:] == skip.shape[2:], f'got shape={x.shape[2:]}, expected={skip.shape[2:]}'

        x = self.merge_layer([skip, x])
        x = self.ops_conv(x)
        return x


class ConvTransposeBlockType(Protocol):
    def __call__(
            self,
            config: LayerConfig,
            input_channels: int,
            output_channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            output_padding: Optional[Union[int, Sequence[int]]] = None,
            stride: Optional[Stride] = None,
            padding_mode: Optional[str] = None) -> nn.Module:

        ...


class ConvBlockType(Protocol):
    def __call__(
            self,
            config: LayerConfig,
            input_channels: int,
            output_channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            stride: Optional[Stride] = None,
            padding_mode: Optional[str] = None) -> nn.Module:
        ...


class BlockSqueezeExcite(nn.Module):
    """
    Squeeze-and-excitation block

    References:
        [1] "Squeeze-and-Excitation Networks", https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self,
                 config: LayerConfig,
                 input_channels: int,
                 r: int = 24):
        super().__init__()

        assert config.ops.adaptative_avg_pool_fn is not None
        self.squeeze = config.ops.adaptative_avg_pool_fn(1)

        squeezed_channels = input_channels // r
        assert squeezed_channels > 1, f'invalid channels! input_channels={input_channels}, r={r}'

        conv_1 = BlockConv(
            config=config,
            input_channels=input_channels,
            output_channels=squeezed_channels,
            kernel_size=1)

        assert config.activation is not None
        activation = config.activation()

        conv_2 = BlockConv(
            config=config,
            input_channels=squeezed_channels,
            output_channels=input_channels,
            kernel_size=1)

        self.excitation = nn.Sequential(conv_1, activation, conv_2, nn.Sigmoid())

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


class BlockRes(nn.Module):
    """
    Original Residual block design

    References:
        [1] "Deep Residual Learning for Image Recognition", https://arxiv.org/abs/1512.03385
    """
    def __init__(
            self,
            config: LayerConfig,
            input_channels: int,
            *,
            kernel_size: Optional[KernelSize] = None,
            padding: Optional[Padding] = None,
            padding_mode: Optional[str] = None,
            base_block: ConvBlockType = BlockConvNormActivation):
        super().__init__()

        config = copy.copy(config)
        conv_kwargs = copy.copy(config.conv_kwargs)
        if kernel_size is not None:
            conv_kwargs['kernel_size'] = kernel_size
        if padding is not None:
            conv_kwargs['padding'] = padding
        if padding_mode is not None:
            conv_kwargs['padding_mode'] = padding_mode
        config.conv_kwargs = conv_kwargs

        # DO NOT use _posprocess_padding here. This is specific to a convolution!

        stride = 1
        self.block_1 = base_block(
            config, input_channels, input_channels,
            kernel_size=kernel_size, padding=padding,
            stride=stride, padding_mode=padding_mode)

        assert config.activation is not None
        self.activation = config.activation(**config.activation_kwargs)

        config.activation = None
        self.block_2 = base_block(
            config, input_channels, input_channels,
            kernel_size=kernel_size, padding=padding,
            stride=stride, padding_mode=padding_mode)

    def forward(self, x: TorchTensorNCX) -> TorchTensorNCX:
        o = self.block_1(x)
        o = self.block_2(o)
        return self.activation(x + o)


class BlockResPreAct(nn.Module):
    """
    Pre-activation residual block
    """
    def __init__(self,
                 config: LayerConfig,
                 input_channels: int,
                 planes: int,
                 stride: Optional[Stride] = None,
                 kernel_size: Optional[KernelSize] = 3):

        super().__init__()

        config = copy.copy(config)

        assert config.activation is not None
        self.act1 = config.activation()
        assert config.norm is not None
        self.bn1 = config.norm(input_channels)
        self.conv1 = BlockConv(
            config,
            input_channels,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding='same',
            bias=False
        )

        self.act2 = config.activation()
        self.bn2 = config.norm(planes)
        self.conv2 = BlockConv(
            config,
            planes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            bias=False
        )

        if stride is None:
            stride = 1
        stride_np = np.asarray(stride)

        self.shortcut: Optional[nn.Sequential]
        if (stride_np == 1).all() or input_channels != planes:
            self.shortcut = nn.Sequential(
                BlockConv(
                    config,
                    input_channels,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                )
            )
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.act1(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x

        out = self.conv1(out)
        out = self.conv2(self.act2(self.bn2(out)))
        out += shortcut
        return out


class BlockPoolClassifier(nn.Module):
    def __init__(self, config: LayerConfig, input_channels: int, output_channels: int, pooling_kernel=4):
        super().__init__()
        self.linear = nn.Linear(input_channels, output_channels)
        assert config.pool is not None
        self.pooling = config.pool(pooling_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pooling(x)
        x = flatten(x)
        x = self.linear(x)
        return x
