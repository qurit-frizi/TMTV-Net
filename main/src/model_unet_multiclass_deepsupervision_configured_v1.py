from functools import partial
from timeit import repeat
from torch import nn
from basic_typing import Batch
from layers.unet_base import LatentConv, UNetBase
from layers.blocks import BlockConvNormActivation, BlockUpsampleNnConvNormActivation
from typing import Dict, Optional
# import trw
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_unet_multiclass import ModelUNetMulticlass, LossDiceCrossEntropyFocal
from model_unet_multiclass_deepsupervision import ModelUNetMulticlassDeepSupervision, LossDiceCrossEntropyFocal2

from layers.blocks import BlockConvNormActivation, BlockUpDeconvSkipConv
from layers.layer_config import LayerConfig, default_layer_config, NormType
from layers.unet_base import BlockConvType, BlockTypeConvSkip, Up
from layers.unet_attention import MergeBlockAttention_Gating_Input


class Down(nn.Module):
    def __init__(
            self,
            layer_config: LayerConfig,
            bloc_level: int,
            input_channels: int,
            output_channels: int,
            block: BlockConvType = BlockConvNormActivation,
            **block_kwargs):
        super().__init__()

        ops = [
            block(
                config=layer_config,
                input_channels=input_channels,
                output_channels=output_channels,
                **block_kwargs
            ),
            block(
                config=layer_config,
                input_channels=output_channels,
                output_channels=output_channels,
            )
        ]

        self.ops = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


SimpleMulticlassUNet_dice_ce_fov_v5 = partial(
    ModelUNetMulticlass,
    model=UNetBase(
        dim=3,
        input_channels=2,
        channels=[32, 64, 128, 256, 320],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            conv_kwargs={'padding': 'same', 'bias': False},
            deconv_kwargs={'padding': 'same'},
            norm_kwargs={'affine': True},
            activation=partial(nn.LeakyReLU, negative_slope=0.01)
        ),
        activation=None,
        down_block_fn=Down,
        # , deconv_kernel_size=2 
        up_block_fn=partial(Up, block=partial(BlockUpDeconvSkipConv, nb_repeats=2)),
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=3))
    ),
    loss_fn=partial(LossDiceCrossEntropyFocal, ce_factor=1.0)
)


SimpleMulticlassUNet_dice_ce_fov_v6 = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=2,
        channels=[48, 64, 128, 256, 320],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            #conv_kwargs={'padding': 'same', 'bias': False},
            conv_kwargs={'padding': 'same'},
            deconv_kwargs={'padding': 'same'},
            #norm_kwargs={'affine': True},
            activation=partial(nn.LeakyReLU, negative_slope=0.01)
        ),
        activation=None,
        down_block_fn=Down,
        up_block_fn=partial(Up, block=partial(BlockUpDeconvSkipConv, nb_repeats=2)),
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=3))
    ),
    with_deep_supervision=False,
    #loss_fn=LossDiceCrossEntropyFocal2_nnUnet
    loss_fn=LossDiceCrossEntropyFocal2
)


SimpleMulticlassUNet_dice_ce_fov_v6_ds = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=2,
        channels=[48, 64, 128, 256, 320],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            #conv_kwargs={'padding': 'same', 'bias': False},
            conv_kwargs={'padding': 'same'},
            deconv_kwargs={'padding': 'same'},
            #norm_kwargs={'affine': True},
            activation=partial(nn.LeakyReLU, negative_slope=0.01)
        ),
        activation=None,
        down_block_fn=Down,
        up_block_fn=partial(Up, block=partial(BlockUpDeconvSkipConv, nb_repeats=2)),
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=3))
    ),
    with_deep_supervision=True,
    #loss_fn=LossDiceCrossEntropyFocal2_nnUnet
    loss_fn=LossDiceCrossEntropyFocal2
)


SimpleMulticlassUNet_dice_ce_fov_v7_ds_lung = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=3,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    with_deep_supervision=True,
    loss_fn=LossDiceCrossEntropyFocal2,
    with_ct_lung=True,
    input_target_shape=(3, 96, 96, 96)
)

SimpleMulticlassUNet_dice_ce_fov_v8_ds_lung = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=3,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    with_deep_supervision=True,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=1.0, gamma=1.0),
    with_ct_lung=True,
    input_target_shape=(3, 96, 96, 96)
)

SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=5,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    with_deep_supervision=True,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=1.0, gamma=1.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    input_target_shape=(5, 96, 96, 96)
)

SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_boundary = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=5,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    with_deep_supervision=True,
    boundary_loss_factor=1.0,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=0.5, gamma=1.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    input_target_shape=(5, 96, 96, 96)
)

SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_boundary_sensitivity = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=5,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    with_deep_supervision=True,
    boundary_loss_factor=1.0,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=0.5, gamma=1.0, sensitivity=2.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    input_target_shape=(5, 96, 96, 96)
)

SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_boundary_coord = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=8,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    with_deep_supervision=True,
    boundary_loss_factor=1.0,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=0.5, gamma=1.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    with_additional_features=('z_coords', 'y_coords', 'x_coords'),
    input_target_shape=(8, 96, 96, 96)
)

SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_boundary_composite = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=5,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    with_deep_supervision=False,
    with_deep_supervision_composite=True,
    boundary_loss_factor=1.0,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=0.5, gamma=1.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    input_target_shape=(5, 96, 96, 96)
)


SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_larger = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=5,
        channels=[96, 128, 156, 192],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    with_deep_supervision=True,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=1.0, gamma=1.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    input_target_shape=(5, 96, 96, 96)
)

SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_larger2 = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=5,
        channels=[128, 156, 192, 224],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=7))
    ),
    with_deep_supervision=True,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=1.0, gamma=1.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    input_target_shape=(5, 96, 96, 96)
)


SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_larger2_sensitivity = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=5,
        channels=[128, 156, 192, 224],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=7))
    ),
    with_deep_supervision=True,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=1.0, gamma=1.0, sensitivity=1.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    input_target_shape=(5, 96, 96, 96)
)


SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_focal_composite = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=5,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    with_deep_supervision=False,
    with_deep_supervision_composite=True,
    boundary_loss_factor=None,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=0.5, gamma=2.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    input_target_shape=(5, 96, 96, 96)
)


SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_focal = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=5,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    with_deep_supervision=True,
    with_deep_supervision_composite=False,
    boundary_loss_factor=None,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=0.5, gamma=2.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    input_target_shape=(5, 96, 96, 96)
)


SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_boundary_attention = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=5,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9)),
        up_block_fn=partial(Up, block=partial(
            BlockUpDeconvSkipConv, 
            merge_layer_fn=MergeBlockAttention_Gating_Input, 
            deconv_block=BlockUpsampleNnConvNormActivation)
        )
    ),
    with_deep_supervision=True,
    boundary_loss_factor=1.0,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=0.5, gamma=1.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    input_target_shape=(5, 96, 96, 96)
)


from basic_typing import KernelSize, Padding, Stride
from typing import Union, Sequence
from layers.blocks import BlockMerge, BlockDeconvNormActivation
from layers.non_local import BlockNonLocal, linear_embedding

class BlockUpDeconvSkipConvNonLocal(nn.Module):
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

        self.non_local = BlockNonLocal(
            config=config,
            input_channels=output_channels,
            f_mapping_fn=linear_embedding,
            g_mapping_fn=linear_embedding,
            intermediate_channels=max(16, output_channels // 2)
        )

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
        if x.shape[2] <= 32:
            # else it is just too slow...
            x = self.non_local(x)
        return x


SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_boundary_non_local = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNetBase(
        dim=3,
        input_channels=5,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9)),
        up_block_fn=partial(Up, block=BlockUpDeconvSkipConvNonLocal)
    ),
    with_deep_supervision=True,
    boundary_loss_factor=1.0,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=0.5, gamma=1.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    input_target_shape=(5, 96, 96, 96)
)