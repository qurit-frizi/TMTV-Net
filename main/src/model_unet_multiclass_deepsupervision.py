from functools import partial
from torch import nn
from basic_typing import Batch
from layers.unet_base import LatentConv
from layers.blocks import BlockConvNormActivation
from layers.deep_supervision import DeepSupervision
from typing import Dict, Sequence
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from basic_typing import TorchTensorNCX
from outputs import OutputSegmentation, OutputEmbedding, OutputLoss
from losses import LossFocalMulticlass, LossDiceMulticlass

def adaptative_weighting(outputs: Sequence[TorchTensorNCX]) -> np.ndarray:
    """
    Weight the outputs proportionally to their spatial extent
    """
    elements = np.asarray([o.shape[2] for o in outputs], dtype=np.float32)
    return elements / elements.max()


def select_deepsupervision_layers(s: Sequence[TorchTensorNCX]) -> Sequence[TorchTensorNCX]:
    return list(s[0:-2] + [s[-2]])


def output_postprocessing_fn(x):
    return torch.argmax(x, dim=1, keepdim=True).type(torch.long)


try:
    from torch.cuda.amp import autocast
except ModuleNotFoundError:
    # PyTorch version did not support autocast
    def do_nothing_fn():
        pass

    autocast = lambda: lambda x: do_nothing_fn()



from layers.layer_config import LayerConfig, default_layer_config
from layers.convs import ModuleWithIntermediate
from layers.blocks import ConvBlockType, BlockConvNormActivation
from basic_typing import ShapeCX, TorchTensorNCX, TensorNCX
from utilities import get_device
from typing_extensions import Literal, Protocol
import copy
from typing import Callable, Sequence, Optional, List
from resize import resize



class DeepSupervisionComposite(nn.Module):
    """
    Apply a deep supervision layer to help the flow of gradient reach top level layers.

    Selected layers will be resized and summed for the output layers

    This is mostly used for segmentation tasks.

    Example:
        >>> backbone = UNetBase(dim=2, input_channels=3, channels=[2, 4, 8], output_channels=2)
        >>> deep_supervision = DeepSupervisionSum(backbone, [3, 8, 16])
        >>> i = torch.zeros([1, 3, 8, 16], dtype=torch.float32)
        >>> t = torch.zeros([1, 1, 8, 16], dtype=torch.long)
        >>> outputs = deep_supervision(i, t)
    """
    def __init__(
            self,
            backbone: ModuleWithIntermediate,
            input_target_shape: ShapeCX,
            output_block: ConvBlockType = BlockConvNormActivation,
            select_outputs_fn: Callable[[Sequence[TorchTensorNCX]], Sequence[TorchTensorNCX]] = select_deepsupervision_layers,
            resize_mode: Literal['nearest', 'linear'] = 'linear',
            config: LayerConfig = default_layer_config(dimensionality=None)):
        """

        Args:
            backbone: the backbone that will create a hierarchy of features. Must inherit
                from :class:`ModuleWithIntermediate`
            input_target_shape: a shape that will be used to instantiate the outputs of the backbone. Internally,
                this is used to create output layers compatible with the backbone
            output_block: the block to be used to calculate the output
            select_outputs_fn: function that returns intermediates to apply deep supervision
            resize_mode: how to resize the outputs to match the target
            config: default layer configuration
        """

        super().__init__()
        self.backbone = backbone

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

        for o in selected_outputs:
            output = output_block(
                config,
                o.shape[1],
                self.output_channels,
                kernel_size=1,
                stride=1
            )
            self.outputs.append(output)

        self.resize_mode = resize_mode

    def forward(self, x: torch.Tensor, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        os = self.backbone.forward_with_intermediate(x, latent=latent)
        target_shape = os[-1].shape[2:]

        outputs = 0
        seq = self.select_outputs_fn(os)
        assert len(seq) == len(self.outputs)
        for n, o in enumerate(seq):
            o_tfm = self.outputs[n](o)
            o_tfm_resized = resize(o_tfm, target_shape, mode=self.resize_mode)
            outputs += o_tfm_resized

        return outputs


class ModelUNetMulticlassDeepSupervision(nn.Module):
    def __init__(
            self, 
            model, 
            loss_fn, 
            with_ct_lung=False, 
            with_ct_soft=False,
            with_pet_hot=False,
            with_additional_features=(),
            input_target_shape=(2, 128, 128, 128), 
            boundary_loss_factor=None,
            with_deep_supervision=True,
            with_deep_supervision_composite=False,
            return_deep_supervision_intermediate=False) -> None:
        super().__init__()
        self.output_postprocessing = output_postprocessing_fn
        self.loss_fn = loss_fn
        self.boundary_loss_factor = boundary_loss_factor
        self.output_fn = partial(OutputSegmentation, criterion_fn=self.loss_fn, output_postprocessing=self.output_postprocessing)
        
        assert int(with_deep_supervision) + int(with_deep_supervision_composite) <= 1 
        if with_deep_supervision:
            self.model = DeepSupervision(
                model, 
                input_target_shape=input_target_shape, 
                output_creator=self.output_fn,
                config=model.config,
                weighting_fn=adaptative_weighting,
                select_outputs_fn=select_deepsupervision_layers,
                return_intermediate=return_deep_supervision_intermediate,
            )
        elif with_deep_supervision_composite:
            self.model = DeepSupervisionComposite(
                model, 
                input_target_shape=input_target_shape, 
                config=model.config,
                select_outputs_fn=select_deepsupervision_layers
            )
        else:
            self.model = model

        self.with_deep_supervision = with_deep_supervision
        self.features = ['ct', 'suv']
        if with_ct_lung:
            self.features.append('ct_lung')
        if with_ct_soft:
            self.features.append('ct_soft')
        if with_pet_hot:
            self.features.append('suv_hot')
        self.features += list(with_additional_features)

    def forward(self, batch: Batch) -> Dict:
        ct = batch['ct']
        seg = batch.get('seg')
        assert len(ct.shape) == 5
        assert ct.shape[1] == 1

        features = []
        for f in self.features:
            features.append(batch[f])

        features = torch.cat(features, dim=1)

        #np.save('/mnt/datasets/ludovic/AutoPET/tmp/cascade.inference.output_found.npy', batch['cascade.inference.output_found'][0].cpu().numpy())
        #np.save('/mnt/datasets/ludovic/AutoPET/tmp/suv.npy', batch['suv'][0].cpu().numpy())
        #np.save('/mnt/datasets/ludovic/AutoPET/tmp/seg.npy', batch['seg'][0].cpu().numpy())

        if seg is None:
            # inference mode
            if self.with_deep_supervision:
                os = self.model.backbone.forward_with_intermediate(features)
                os = self.model.select_outputs_fn(os)
                o = self.model.outputs[-1](os[-1])
            else:
                o = self.model(features)
            return {
                'seg': OutputEmbedding(o)
            }
        elif self.with_deep_supervision:
            o = self.model(features, seg)
            intermediates = None
            if isinstance(o, tuple):
                o, intermediates = o  
            outputs = {f'seg_{o_n.loss_scaling}': o_n for n, o_n in enumerate(o[:-1])}
            outputs['seg'] = o[-1]
            if intermediates is not None:
                outputs['features'] = OutputEmbedding(intermediates[-2])
        else:
            o = self.model(features)
            outputs = {
                'seg': self.output_fn(o, seg)
            }

        # boundary loss
        dt = batch.get('surface_loss_distance_transform')
        if self.boundary_loss_factor is not None and dt is not None:
            o_pb = torch.softmax(outputs['seg'].output, dim=1)
            boundary_loss = loss_surface(o_pb, dt) * self.boundary_loss_factor
            outputs['boundary'] = OutputLoss(boundary_loss)

            #s = 6
            #np.save('/mnt/datasets/ludovic/tmp4/loss_dt.npy', (dt[s, 1]).detach().cpu().numpy())
            #np.save('/mnt/datasets/ludovic/tmp4/loss.npy', (o_pb[s, 1] * dt[s, 1]).detach().cpu().numpy())
            #np.save('/mnt/datasets/ludovic/tmp4/output.npy', o_pb[s, 1].detach().cpu().numpy())
            #np.save('/mnt/datasets/ludovic/tmp4/seg.npy', o_pb[s, 1].detach().cpu().numpy())

        """
        index = 0
        np.save('/mnt/datasets/ludovic/tmp4/seg.npy', (seg[index]).detach().cpu().numpy())
        np.save('/mnt/datasets/ludovic/tmp4/ct.npy', (ct[index]).detach().cpu().numpy())
        np.save('/mnt/datasets/ludovic/tmp4/suv.npy', (batch['suv'][index]).detach().cpu().numpy())
        np.save('/mnt/datasets/ludovic/tmp4/cascade.npy', batch['cascade.inference.output_found'][index].cpu().numpy())
        np.save('/mnt/datasets/ludovic/tmp4/found.npy', torch.softmax(o[index], 1)[1].cpu().detach().numpy())
        """   
        return outputs


class LossDiceCrossEntropyFocal2:
    def __init__(self, ce_factor=0.5, gamma=2.0, sensitivity=0.0) -> None:
        self.focal = LossFocalMulticlass(gamma=gamma)
        self.dice = LossDiceMulticlass(normalization_fn=partial(nn.Softmax, dim=1), discard_background_loss=True, smooth=1e-3)
        self.ce_factor = ce_factor
        self.sensitivity_factor = sensitivity

    def __call__(self, output, target):
        loss = self.dice(output, target)
        if self.ce_factor > 0:
            loss += self.ce_factor * self.focal(output, target)
            
        if self.sensitivity_factor > 0:
            assert output.shape[1] == 2, 'must be binary classification!' 
            output_softmax = nn.Softmax(dim=1)(output)
            l1_loss = torch.nn.L1Loss(reduction='none')(output_softmax[:, 1:], target)

            l1_loss_flat = l1_loss.view((target.shape[0], -1)) 
            target_flat = target.view((target.shape[0], -1))
            assert l1_loss_flat.shape == target_flat.shape
            nb_elements = target_flat.sum(dim=1)
            sensitivity_loss = (l1_loss_flat * target_flat).sum(dim=1) / (1e-4 + nb_elements)
            loss += self.sensitivity_factor * sensitivity_loss
        return loss 
        
