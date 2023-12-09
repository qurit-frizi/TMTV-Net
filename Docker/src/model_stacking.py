from functools import partial
from torch import nn
from basic_typing import Batch
from typing import Dict, Sequence, Callable
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from model_unet_multiclass_deepsupervision import LossDiceCrossEntropyFocal2, output_postprocessing_fn
from model_refiner_multiclass_deepsupervision_configured_v1 import Refiner
from layers.layer_config import default_layer_config, NormType
from outputs import OutputSegmentation

def linear_stacking_fn(nb_inputs: int, nb_outputs: int = 2):
    c = nn.Conv3d(nb_inputs, nb_outputs, kernel_size=1)
    
    # defaults to averaging
    c.weight.data[:] = 1.0 / nb_inputs
    c.bias.data[:] = 0.0
    return c


def resblocks_stacking_fn(nb_inputs: int, nb_outputs: int = 2, nb_blocks=4, nb_features=32):
    refiner = Refiner(
        nb_inputs=nb_inputs,
        output_channels=nb_outputs,
        nb_blocks=nb_blocks,
        nb_features=nb_features,
        config=default_layer_config(
            dimensionality=3,
            norm_type=NormType.InstanceNorm,
            conv_kwargs={'padding': 'same', 'bias': False},
            deconv_kwargs={'padding': 'same'},
            norm_kwargs={'affine': True},
            activation=partial(nn.LeakyReLU, negative_slope=0.01)
        )
    )
    return refiner


class ModelStacking(nn.Module):
    def __init__(self,
            base_models: Sequence[nn.Module],
            stacking_block_fn: Callable[[int], nn.Module] = linear_stacking_fn,
            loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=0.5, gamma=1.0),
            additional_inputs: Sequence[str]=()) -> None:
        super().__init__()

        # use a ModuleList so that all parameters are moved/halved correcly
        self.base_models = nn.ModuleList()
        for m in base_models:
            self.base_models.append(copy.deepcopy(m))
        self.staking_blocks = stacking_block_fn(len(base_models) + len(additional_inputs))
        self.output_fn = partial(OutputSegmentation, criterion_fn=loss_fn, output_postprocessing=output_postprocessing_fn)
        self.additional_inputs = additional_inputs

    def forward(self, batch: Batch) -> Batch:
        outputs = []
        seg = batch.get('seg')
        for m in self.base_models:
            # we don't want to update the base model
            # so don't calculate the gradients and
            # detach the output
            with torch.no_grad():
                # calculate the probability of lesion
                batch_copy = copy.deepcopy(batch)
                o = F.softmax(m(batch_copy)['seg'].output, dim=1)
                #o = m(batch_copy)['seg'].output
                assert o.shape[1] == 2, 'expecting binary output'
                outputs.append(o[:, 1:2])

        """
        np.save('/mnt/datasets/ludovic/AutoPET/tmp/output_0_expected.npy', seg[0, 0].detach().cpu().numpy())
        for model_n, output in enumerate(outputs):
            np.save(f'/mnt/datasets/ludovic/AutoPET/tmp/output_{model_n}.npy', output[0, 0].detach().cpu().numpy())

        np.save(f'/mnt/datasets/ludovic/AutoPET/tmp/last_output_{model_n}_s0.npy', o[0, 0].detach().cpu().numpy())
        np.save(f'/mnt/datasets/ludovic/AutoPET/tmp/last_output_{model_n}_s1.npy', o[0, 1].detach().cpu().numpy())
        
        from corelib import write_lz4_pkl, read_lz4_pkl
        write_lz4_pkl('/mnt/datasets/ludovic/AutoPET/tmp/batch.pkl.lz4', batch_copy)
        batch2 = read_lz4_pkl('/mnt/datasets/ludovic/AutoPET/tmp/batch.pkl.lz4')
        """
        
        for feature_name in self.additional_inputs:
            feature = batch.get(feature_name)
            assert feature is not None
            outputs.append(feature)

        stacking_input = torch.cat(outputs, dim=1).detach()
        stacking_output = self.staking_blocks(stacking_input)
        
        return {
            'seg': self.output_fn(stacking_output, seg),
        }


ModelStackingResblocks = partial(ModelStacking, stacking_block_fn=resblocks_stacking_fn)

ModelStackingResblocksPET = partial(ModelStacking, stacking_block_fn=resblocks_stacking_fn, additional_inputs=('suv',))