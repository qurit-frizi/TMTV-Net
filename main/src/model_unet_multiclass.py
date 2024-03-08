from functools import partial
from torch import nn
from basic_typing import Batch
from layers.unet_base import LatentConv
from layers.unet_base import UNetBase
from layers.layer_config import default_layer_config, NormType
from layers.blocks import BlockConvNormActivation
from typing import Dict
import torch
from losses import LossDiceMulticlass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelUNetMulticlass(nn.Module):
    def __init__(self, model, loss_fn, with_softmax=True, with_ct_lung=False) -> None:
        super().__init__()
        self.model = model
        self.output_postprocessing = lambda x: torch.argmax(x, dim=1, keepdim=True).type(torch.long)
        self.loss_fn = loss_fn
        self.with_softmax = with_softmax
        self.with_ct_lung = with_ct_lung

    def forward(self, batch: Batch) -> Dict:
        suv = batch['suv']
        ct = batch['ct']
        seg = batch['seg']
        assert len(ct.shape) == 5
        assert ct.shape[1] == 1

        if self.with_ct_lung:
            ct_lung = batch['ct_lung']
            features = torch.cat([ct, ct_lung, suv], dim=1)
        else:
            features = torch.cat([ct, suv], dim=1)

        o = self.model(features)

        if self.with_softmax:
            o = F.softmax(o, dim=1)

        #z_half = ct.shape[2] // 2 

        return {
            'seg': train.OutputSegmentation(o, seg, criterion_fn=self.loss_fn, output_postprocessing=self.output_postprocessing),
            #'2d_ct': train.OutputEmbedding(ct[:, :, z_half]),
            #'2d_suv': train.OutputEmbedding(suv[:, :, z_half]),
            #'2d_seg': train.OutputEmbedding(seg[:, :, z_half]),
            #'2d_found': train.OutputEmbedding(self.output_postprocessing((o[:, :, z_half])).type(torch.float32)),
        }


class LossDiceCrossEntropyFocal:
    def __init__(self, ce_factor=0.5, gamma=2.0) -> None:
        self.focal = train.LossFocalMulticlass(gamma=gamma)
        self.dice = train.LossDiceMulticlass(normalization_fn=None, discard_background_loss=False, smooth=1.0)
        self.ce_factor = ce_factor

    def __call__(self, output, target):
        return self.ce_factor * self.focal(output, target) + self.dice(output, target)


SimpleMulticlassUNet = partial(
    ModelUNetMulticlass,
    model=UNetBase(
        dim=3,
        input_channels=2,
        channels=[32, 64, 128, 256],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=5))
    ),
    loss_fn=partial(LossDiceMulticlass, normalization_fn=None)
)


SimpleMulticlassUNet_dice_ce = partial(
    ModelUNetMulticlass,
    model=UNetBase(
        dim=3,
        input_channels=2,
        channels=[32, 64, 128, 256],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=5))
    ),
    loss_fn=LossDiceCrossEntropyFocal
)

SimpleMulticlassUNet_dice_ce_fov = partial(
    ModelUNetMulticlass,
    model=UNetBase(
        dim=3,
        input_channels=2,
        channels=[32, 64, 128, 256],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    loss_fn=LossDiceCrossEntropyFocal
)

SimpleMulticlassUNet_dice_ce_fov_v2 = partial(
    ModelUNetMulticlass,
    model=UNetBase(
        dim=3,
        input_channels=2,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    loss_fn=LossDiceCrossEntropyFocal
)

SimpleMulticlassUNet_dice_ce_fov_v3_lung = partial(
    ModelUNetMulticlass,
    model=UNetBase(
        dim=3,
        input_channels=3,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    loss_fn=LossDiceCrossEntropyFocal,
    with_ct_lung=True
)

SimpleMulticlassUNet_dice_ce_fov_v3_lung_large_init = partial(
    ModelUNetMulticlass,
    model=UNetBase(
        dim=3,
        input_channels=3,
        channels=[64, 96, 128, 156],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU
        ),
        activation=None,
        init_block_fn=partial(BlockConvNormActivation, kernel_size=7),
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    loss_fn=LossDiceCrossEntropyFocal,
    with_ct_lung=True
)

BlockConvNormActivation

SimpleMulticlassUNet_dice_ce_fov_v4 = partial(
    ModelUNetMulticlass,
    model=UNetBase(
        dim=3,
        input_channels=2,
        channels=[48, 64, 96, 128, 128],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9))
    ),
    loss_fn=LossDiceCrossEntropyFocal
)


from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.networks.layers.factories import Act

SimpleMulticlassUNet_dice_ce_monai = partial(
    ModelUNetMulticlass,
    model=UNet(
            # dimensions=3,
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            #norm=Norm.BATCH,
    ),
    loss_fn=LossDiceCrossEntropyFocal
)








def loss_debug(output, target):
    return torch.nn.functional.cross_entropy(output, target.squeeze(dim=1), reduction='none') 

SimpleMulticlassUNet_dice_ce_DEBUG = partial(
    ModelUNetMulticlass,
    model=UNetBase(
        dim=3,
        input_channels=2,
        channels=[32, 64, 128, 256],
        output_channels=2,
        config=default_layer_config(
            norm_type=NormType.InstanceNorm,
            activation=nn.LeakyReLU
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=5))
    ),
    loss_fn=lambda: loss_debug
)


def make_resnet_backbone_decoder():
    config = default_layer_config(
        dimensionality=None,
        norm_type=NormType.InstanceNorm,
        activation=torch.nn.LeakyReLU
    )

    encoder = PreActResNet(
                dimensionality=3, 
                input_channels=3, 
                output_channels=None,
                num_blocks=(2, 2, 2, 2),
                strides=(1, 2, 2, 2),
                channels=(32, 48, 64, 96),
                config=config
    )

    decoder = BackboneDecoder(
        decoding_channels=(96, 64, 48, 32),
        output_channels=2,
        backbone=encoder,
        backbone_transverse_connections=(0, 1, 2, 3),
        backbone_input_shape=(1, 3, 96, 96, 96),
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=9)),
        config=config
    )

    model = ModelUNetMulticlass(
        model=decoder,
        loss_fn=LossDiceCrossEntropyFocal,
        with_ct_lung=True
    )
    return model