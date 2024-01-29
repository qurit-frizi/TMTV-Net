from functools import partial
from timeit import repeat
from torch import nn
from basic_typing import Batch
from layers.unet_base import LatentConv
from layers.blocks import BlockConvNormActivation
from typing import Dict
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_unet_multiclass import ModelUNetMulticlass, LossDiceCrossEntropyFocal
from model_unet_multiclass_deepsupervision import ModelUNetMulticlassDeepSupervision, LossDiceCrossEntropyFocal2

from layers.blocks import BlockConvNormActivation, BlockUpDeconvSkipConv, BlockRes
from layers.layer_config import LayerConfig, default_layer_config, NormType
from layers.unet_base import BlockConvType, BlockTypeConvSkip, Up
from layers.deep_supervision import DeepSupervision


class Refiner(nn.Module):
    def __init__(self, nb_inputs, nb_blocks, nb_features, config, output_channels=2) -> None:
        super().__init__()

        init_blocks = [
            BlockConvNormActivation(
                config=config, 
                input_channels=nb_inputs, 
                output_channels=nb_features, 
                kernel_size=9)
        ]

        blocks = [
            BlockRes(config=config, input_channels=nb_features, kernel_size=3) for n in range(nb_blocks)
        ]

        output_blocks = [
            BlockConvNormActivation(
                config=config, 
                input_channels=nb_features, 
                output_channels=output_channels, 
                kernel_size=1)
        ]

        self.ops = nn.Sequential(*init_blocks, *blocks, *output_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


Refiner_dice_ce_fov_v1_ds_lung_soft_hot_boundary = partial(
    ModelUNetMulticlassDeepSupervision,
    model=Refiner(
        nb_inputs=6,
        nb_blocks=4,
        nb_features=32,
        config=default_layer_config(
            dimensionality=3,
            norm_type=NormType.InstanceNorm,
            conv_kwargs={'padding': 'same', 'bias': False},
            deconv_kwargs={'padding': 'same'},
            norm_kwargs={'affine': True},
            activation=partial(nn.LeakyReLU, negative_slope=0.01)
        )
    ),
    with_deep_supervision=False,
    boundary_loss_factor=1.0,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=0.5, gamma=1.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    with_additional_features=('cascade.inference.output_found',),
    input_target_shape=(6, 96, 96, 96)
)

Refiner_dice_ce_fov_v1_ds_lung_soft_hot_boundary_sensitivity = partial(
    ModelUNetMulticlassDeepSupervision,
    model=Refiner(
        nb_inputs=6,
        nb_blocks=4,
        nb_features=32,
        config=default_layer_config(
            dimensionality=3,
            norm_type=NormType.InstanceNorm,
            conv_kwargs={'padding': 'same', 'bias': False},
            deconv_kwargs={'padding': 'same'},
            norm_kwargs={'affine': True},
            activation=partial(nn.LeakyReLU, negative_slope=0.01)
        )
    ),
    with_deep_supervision=False,
    boundary_loss_factor=1.0,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=0.5, gamma=1.0, sensitivity=0.5),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    with_additional_features=('cascade.inference.output_found',),
    input_target_shape=(6, 96, 96, 96)
)

Refiner_dice_ce_fov_v1_ds_lung_soft_hot_boundary_8 = partial(
    ModelUNetMulticlassDeepSupervision,
    model=Refiner(
        nb_inputs=6,
        nb_blocks=8,
        nb_features=32,
        config=default_layer_config(
            dimensionality=3,
            norm_type=NormType.InstanceNorm,
            conv_kwargs={'padding': 'same', 'bias': False},
            deconv_kwargs={'padding': 'same'},
            norm_kwargs={'affine': True},
            activation=partial(nn.LeakyReLU, negative_slope=0.01)
        )
    ),
    with_deep_supervision=False,
    boundary_loss_factor=1.0,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=0.5, gamma=1.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    with_additional_features=('cascade.inference.output_found',),
    input_target_shape=(6, 96, 96, 96)
)

