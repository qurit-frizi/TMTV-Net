import copy
from typing import Callable, Sequence, Optional, List, Tuple, Union

import torch
import torch.nn as nn

from basic_typing import ShapeCX, TorchTensorNCX, TensorNCX
from outputs import Output, OutputSegmentation
from utilities import get_device
from resize import resize
from typing_extensions import Literal, Protocol

from layers.layer_config import LayerConfig, default_layer_config
from layers.convs import ModuleWithIntermediate
from layers.blocks import ConvBlockType, BlockConvNormActivation
import numpy as np



def adaptative_weighting(outputs: Sequence[TorchTensorNCX]) -> np.ndarray:
    """
    Weight the outputs proportionally to their spatial extent
    """
    elements = np.asarray([np.prod(o.shape[2:]) for o in outputs], dtype=np.float32)
    return elements / elements.max()


def select_third_to_last_skip_before_last(s: Sequence[TorchTensorNCX]) -> Sequence[TorchTensorNCX]:
    assert len(s) >= 4
    last: TorchTensorNCX = s[-2]  # use before the last (the last layer will have nb_features == nb_classes)
    return list(s[1:-2]) + [last]  # typing: convert to list so that we have operator `+`


class OutputCreator(Protocol):
    def __call__(self, output: TensorNCX, output_truth: TensorNCX, loss_scaling: float) -> Output: ...


class DeepSupervision(nn.Module):
    """
    Apply a deep supervision layer to help the flow of gradient reach top level layers.

    This is mostly used for segmentation tasks.

    Example:
        >>> backbone = layers.UNetBase(dim=2, input_channels=3, channels=[2, 4, 8], output_channels=2)
        >>> deep_supervision = DeepSupervision(backbone, [3, 8, 16])
        >>> i = torch.zeros([1, 3, 8, 16], dtype=torch.float32)
        >>> t = torch.zeros([1, 1, 8, 16], dtype=torch.long)
        >>> outputs = deep_supervision(i, t)
    """
    def __init__(
            self,
            backbone: ModuleWithIntermediate,
            input_target_shape: ShapeCX,
            output_creator: OutputCreator = OutputSegmentation,
            output_block: ConvBlockType = BlockConvNormActivation,
            select_outputs_fn: Callable[[Sequence[TorchTensorNCX]], Sequence[TorchTensorNCX]] = select_third_to_last_skip_before_last,
            resize_mode: Literal['nearest', 'linear'] = 'linear',
            weighting_fn: Optional[Callable[[Sequence[TorchTensorNCX]], Sequence[float]]] = adaptative_weighting,
            config: LayerConfig = default_layer_config(dimensionality=None),
            return_intermediate: bool = False):
        """

        Args:
            backbone: the backbone that will create a hierarchy of features. Must inherit
                from :class:`layer.ModuleWithIntermediate`
            input_target_shape: a shape that will be used to instantiate the outputs of the backbone. Internally,
                this is used to create output layers compatible with the backbone
            output_creator: specify what type of output and criterion to optimize
            output_block: the block to be used to calculate the output
            select_outputs_fn: function that returns intermediates to apply deep supervision
            resize_mode: how to resize the outputs to match the target
            config: default layer configuration
            weighting_fn: a weighting function to scale the loss of the different outputs
            return_intermediate: if `True`, intermediate layer tensors will be returned in the
                `forward` method
        """

        super().__init__()
        self.backbone = backbone
        self.return_intermediate = return_intermediate

        device = get_device(backbone)
        self.select_outputs_fn = select_outputs_fn

        # dummy test to get the intermediate layer shapes
        # use `2` in case we have 1xCx1x1 and batch norm or instance norm layers
        # it will throw an exception...
        dummy_input = torch.zeros([2] + list(input_target_shape), device=device)
        outputs = backbone.forward_with_intermediate(dummy_input)
        assert isinstance(outputs, Sequence), '`outputs` must be a sequence!'
        dim = len(outputs[0].shape) - 2

        # no activation, these are all output nodes!
        config = copy.copy(config)
        config.set_dim(dim)
        config.activation = None
        config.norm = None

        # this is what we expect as target `C`
        self.output_channels = outputs[-1].shape[1]

        self.outputs = nn.ModuleList()
        selected_outputs = select_outputs_fn(outputs)
        if weighting_fn is not None:
            self.weights = weighting_fn(selected_outputs)
            assert len(self.weights) == len(selected_outputs), 'must have one weight per output!'
        else:
            self.weights = np.ones([len(selected_outputs)], dtype=np.float32)

        for o in selected_outputs:
            output = output_block(
                config,
                o.shape[1],
                self.output_channels,
                kernel_size=1,
                stride=1
            )
            self.outputs.append(output)

        self.output_creator = output_creator
        self.resize_mode = resize_mode

    def forward(self, x: torch.Tensor, target: torch.Tensor, latent: Optional[torch.Tensor] = None) -> Union[List[Output], Tuple[List[Output], List[torch.Tensor]]]:
        os = self.backbone.forward_with_intermediate(x, latent=latent)

        outputs = []
        seq = self.select_outputs_fn(os)
        assert len(seq) == len(self.weights), 'unexpected number of outputs!'
        for n, o in enumerate(seq):
            # reshape the output to have the expected target channels
            # here we have 2 choices:
            # 1) resize the target to match the output. This is problematic since it
            # will remove small objects
            # 2) resize the output to match the target. This is slower but will not remove
            # the small objects
            # Option 2) is better.
            o_tfm = self.outputs[n](o)
            o_tfm_resized = resize(o_tfm, target.shape[2:], mode=self.resize_mode)
            loss_scaling = self.weights[n]
            output = self.output_creator(o_tfm_resized, target, loss_scaling=loss_scaling)
            outputs.append(output)

        if self.return_intermediate:
            return outputs, os
        return outputs
