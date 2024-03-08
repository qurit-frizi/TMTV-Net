from typing import Callable
import torch
from torch import nn
from basic_typing import TorchTensorNCX

from layers.layer_config import LayerConfig
from layers.blocks import BlockConv



def linear_embedding(config: LayerConfig, input_channels: int, output_channels: int) -> nn.Module:
    block = BlockConv(
        config=config, 
        kernel_size=1,
        input_channels=input_channels, 
        output_channels=output_channels
    )

    assert len(block.ops) == 1
    # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
    #nn.init.constant_(block.ops[0].weight, 0)
    #nn.init.constant_(block.ops[0].bias, 0)
    return block


def identity(config: LayerConfig, input_channels: int, output_channels: int) -> nn.Module:
    assert input_channels == output_channels, 'identity: expected identical number of input/output!'
    return nn.Identity()
        

class BlockNonLocal(nn.Module):
    """
    Non local block implementation of [1]

    Defaults to dot product of each feature of each location and using
    a softmax layer to normalize the attention mask.

    [1] https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf

    Support n-d input data.
    """
    def __init__(
            self,
            config: LayerConfig,
            input_channels: int, 
            intermediate_channels: int,
            f_mapping_fn: Callable[[LayerConfig, int, int], nn.Module] = identity,
            g_mapping_fn: Callable[[LayerConfig, int, int], nn.Module] = identity,
            w_mapping_fn: Callable[[LayerConfig, int, int], nn.Module] = linear_embedding,
            normalize_output_fn: nn.Module = nn.Softmax(dim=-1)
            ):
        super().__init__()
        self.input_channels = input_channels
        self.intermediate_channels = intermediate_channels
        self.f_mapping_i = f_mapping_fn(config, input_channels, intermediate_channels)
        self.f_mapping_j = f_mapping_fn(config, input_channels, intermediate_channels)
        self.g_mapping = g_mapping_fn(config, input_channels, intermediate_channels)
        self.w_mapping = w_mapping_fn(config, intermediate_channels, input_channels)
        self.normalize_output_fn = normalize_output_fn

    def forward(self, x: TorchTensorNCX, return_non_local_map: bool = False):
        batch_size = x.shape[0]

        g_x_mapping = self.g_mapping(x)
        intermediate_channels = g_x_mapping.shape[1]
        assert intermediate_channels == self.intermediate_channels, \
            f'unexpected number of channels. Got={intermediate_channels}, expected={self.intermediate_channels}'
        g_x = g_x_mapping.view(batch_size, intermediate_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        f_mapping_i = self.f_mapping_i(x)
        assert f_mapping_i.shape == g_x_mapping.shape, \
            f'shape should match. Got={f_mapping_i.shape}, expected={g_x_mapping.shape}'
        f_mapping_i = f_mapping_i.view(batch_size, intermediate_channels, -1)
        f_mapping_i = f_mapping_i.permute(0, 2, 1)
        f_mapping_j = self.f_mapping_j(x).view(batch_size, intermediate_channels, -1)
        
        f = torch.matmul(f_mapping_i, f_mapping_j)
        # normalize relative to the window position
        f_norm = self.normalize_output_fn(f)
        assert f_norm.shape == f.shape

        y = torch.matmul(f_norm, g_x)

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, intermediate_channels, *x.size()[2:])
        W_y = self.w_mapping(y)
        z = W_y + x

        if return_non_local_map:
            return z, f
        return z
