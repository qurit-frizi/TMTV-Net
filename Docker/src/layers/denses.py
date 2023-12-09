import copy

import torch.nn as nn
from .layer_config import default_layer_config, LayerConfig, NormType
from .flatten import Flatten
from typing import Any, Sequence, List, Optional


def denses(
        sizes: Sequence[int],
        dropout_probability: float = None,
        activation: Any = nn.ReLU,
        normalization_type: Optional[NormType] = NormType.BatchNorm,
        last_layer_is_output: bool = False,
        with_flatten: bool = True,
        config: LayerConfig = default_layer_config(dimensionality=None)) -> nn.Module:
    """

    Args:
        sizes: the size of the linear layers_legacy. The format is [linear1_input, linear1_output, ..., linearN_output]
        dropout_probability: the probability of the dropout layer. If `None`, no dropout layer is added.
        activation: the activation to be used
        normalization_type: the normalization to be used between dense layers_legacy. If `None`, no normalization added
        last_layer_is_output: This must be set to `True` if the last layer of dense is actually an output.
            If the last layer is an output, we should not add batch norm, dropout or
            activation of the last `nn.Linear`
        with_flatten: if True, the input will be flattened
        config: defines the available operations
        
    Returns:
        a nn.Module
    """
    config = copy.copy(config)
    if activation is not None:
        config.activation = activation
    config.norm_type = normalization_type
    config.set_dim(1)

    ops: List[nn.Module] = []
    if with_flatten:
        ops.append(Flatten())
    
    for n in range(len(sizes) - 1):
        current = sizes[n]
        next = sizes[n + 1]

        ops.append(nn.Linear(current, next))
        if n + 2 == len(sizes) and last_layer_is_output:
            pass

        else:
            if config.norm_type is not None:
                ops.append(nn.BatchNorm1d(next, **config.norm_kwargs))

            ops.append(activation(**config.activation_kwargs))

            if dropout_probability is not None and config.dropout is not None:
                ops.append(config.dropout(p=dropout_probability, **config.dropout_kwargs))
    return nn.Sequential(*ops)
