"""
The purpose of this file is to group all functions related to pytorch graph reflection
such as finding layers of specified types in a nn.Module or using the `grad_fn`
"""
from typing import Sequence, Any, Union, List, Optional, Mapping, Tuple

import torch
import torch.nn as nn
import utilities
import collections
import logging
import traceback
import io


logger = logging.getLogger(__name__)


def find_tensor_leaves_with_grad(tensor: torch.Tensor) -> Sequence[torch.Tensor]:
    """
    Find the input leaves of a tensor.

    Input Leaves **REQUIRES** have `requires_grad=True`, else they will not be found

    Args:
        tensor: a torch.Tensor

    Returns:
        a list of torch.Tensor with attribute `requires_grad=True` that is an input of `tensor`
    """
    leaves = []
    visited = set()

    queue = [tensor.grad_fn]
    while len(queue) > 0:
        current = queue.pop()
        visited.add(current)

        if isinstance(current, torch.Tensor) and not isinstance(current, nn.Parameter):
            leaves.append(current)

        if hasattr(current, 'next_functions'):
            for next, _ in current.next_functions:
                if next not in visited:
                    if hasattr(next, 'variable'):
                        queue.append(next.variable)
                    elif hasattr(next, 'next_functions'):
                        queue.append(next)
                    elif next is None:
                        pass
                    else:
                        assert 0
    return leaves


class _CaptureLastModuleType:
    """
    Capture a specified by type and forward traversal with an optional relative index
    """
    def __init__(self, types_of_module, relative_index=0):
        """
        Args:
            types_of_module (nn.Module or tuple): the types of modules we are targeting
            relative_index (int): indicate which module to return from the last collected module
        """
        self.types_of_module = types_of_module
        self.recorded_modules = collections.deque(maxlen=relative_index + 1)
        self.relative_index = relative_index

    def __call__(self, module, module_input, module_output):
        if isinstance(module, self.types_of_module):
            self.recorded_modules.append((module, module_input, module_output))

    def get_module(self):
        if len(self.recorded_modules) == 0:
            return
        if len(self.recorded_modules) < self.relative_index:
            return None
        return self.recorded_modules[0]


def find_last_forward_types(model: nn.Module, inputs: Any, types: Union[Any, Tuple[Any]], relative_index: int = 0) \
        -> Optional[Mapping]:
    """
    Perform a forward pass of the model with given inputs and retrieve the last layer of the specified type

    Args:
        inputs: the input of the model so that we can call `model(inputs)`
        model: the model
        types: the types to be captured. Can be a single type or a tuple of types
        relative_index: indicate which module to return from the last collected module

    Returns:
        None if no layer found or a dictionary of
        (outputs, matched_module, matched_module_input, matched_module_output) if found
    """
    with utilities.CleanAddedHooks(model):
        try:
            capture_conv = _CaptureLastModuleType(types, relative_index=relative_index)
            for module in model.modules():
                if not isinstance(module, nn.Sequential):
                    module.register_forward_hook(capture_conv)
    
            outputs = model(inputs)
            
        except Exception as e:
            io_string = io.StringIO()
            traceback.print_exc(file=io_string)
            logger.error('find_last_forward_types: exception={}'.format(io_string.getvalue()))

    collected_module = capture_conv.get_module()
    if collected_module is not None:
        module, module_input, module_output = collected_module

        r = {
            'outputs': outputs,
            'matched_module': module,
            'matched_module_inputs': module_input,
            'matched_module_output': module_output,
        }

        return r

    return None


def find_last_forward_convolution(
        model: nn.Module,
        inputs: Any,
        types: Union[Any, Tuple[Any]] = (nn.Conv2d, nn.Conv3d, nn.Conv1d), relative_index=0) \
            -> Optional[Mapping]:
    """
    Perform a forward pass of the model with given inputs and retrieve the last convolutional layer

    Args:
        inputs: the input of the model so that we can call `model(inputs)`
        model: the model
        types: the types to be captured. Can be a single type or a tuple of types
        relative_index (int): indicate which module to return from the last collected module

    Returns:
        None if no layer found or a dictionary of
        (outputs, matched_module, matched_module_input, matched_module_output) if found
    """
    return find_last_forward_types(model, inputs, types=types, relative_index=relative_index)


def find_first_forward_convolution(
        model: nn.Module,
        inputs: Any = None,
        types: Union[Any, Tuple[Any]] = (nn.Conv2d, nn.Conv3d, nn.Conv1d), relative_index=0) \
            -> Optional[Mapping]:
    """
    Perform a forward pass of the model with given inputs and retrieve the last convolutional layer

    Args:
        inputs: NOT USED
        model: the model
        types: the types to be captured. Can be a single type or a tuple of types
        relative_index (int): indicate which module to return from the last collected module

    Returns:
        None if no layer found or a dictionary of
        (outputs, matched_module, matched_module_input, matched_module_output) if found
    """
    modules_of_interest = []
    for module in model.modules():
        if isinstance(module, types):
            modules_of_interest.append(module)
    if relative_index >= len(modules_of_interest):
        return None

    return {
        'outputs': None,
        'matched_module': modules_of_interest[relative_index],
        'matched_module_inputs': None,
        'matched_module_output': None
    }
