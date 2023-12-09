from basic_typing import Batch
from typing import Union, List, Dict, Mapping, Sequence, Any

import collections
import logging
import numbers
import sys

import torch
import numpy as np


logger = logging.getLogger(__name__)


def collate_tensors(
        values: Union[
            np.ndarray,
            torch.Tensor,
            Union[
                List[np.ndarray],
                List[torch.Tensor],
                List[numbers.Number],
                List[str],
                List[List[np.ndarray]],
                List[List[torch.Tensor]],
                List[List[numbers.Number]],
                List[List[str]]
            ]
        ],
        device: torch.device,
        pin_memory: bool = False,
        non_blocking: bool = False) -> Union[torch.Tensor, List]:
    """
    express `values` as a torch.Tensor


    Args:
        values: nd.array or torch.Tensor
        device: the device where to create the torch.Tensor
        pin_memory: if True, pin the memory. Required to be a `CUDA` allocated torch.Tensor
        non_blocking: if True, use non blocking memory transfer

    Returns:
        a torch.Tensor if of type numpy.ndarray else, the input type
    """
    tensor = values
    if isinstance(values, (list, np.ndarray, torch.Tensor)) and len(values) == 0:
        return []

    if isinstance(values, list) and isinstance(values[0], torch.Tensor):
        if len(values) == 1:
            # no need to concatenate if we have only a single element!
            # this scenario is quite common with SequenceAsyncReservoir
            # and taking significant time with large tensors
            tensor = values[0]
        else:
            tensor = torch.cat(values)  # type: ignore
    elif isinstance(values, list) and isinstance(values[0], np.ndarray):
        tensor = torch.as_tensor(np.concatenate(values))
    elif isinstance(values, np.ndarray):
        tensor = torch.as_tensor(values)

    elif isinstance(values, list) and isinstance(values[0], list) and isinstance(values[0][0], torch.Tensor):
        # this is from a list of dictionary
        merged = [torch.cat(value) for value in values]  # type: ignore
        if len(merged) == 1:
            tensor = merged[0].view([1] + list(merged[0].shape))
        else:
            tensor = torch.stack(merged)
    elif isinstance(values, list) and isinstance(values[0], list) and isinstance(values[0][0], np.ndarray):
        # this is from a list of dictionary
        merged = [torch.as_tensor(np.concatenate(value)) for value in values]
        if len(merged) == 1:
            tensor = merged[0].view([1] + list(merged[0].shape))
        else:
            tensor = torch.stack(merged)
    elif isinstance(values, list) and isinstance(values[0], numbers.Number):
        tensor = torch.as_tensor(np.asarray(values))
    elif isinstance(values, list) and isinstance(values[0], list) and isinstance(values[0][0], numbers.Number):
        tensor = torch.as_tensor(np.concatenate(values))
    elif isinstance(values, list) and isinstance(values[0], list) and isinstance(values[0][0], str):
        r = []  # type: ignore
        for r_append in values:
            r += r_append  # type: ignore
        tensor = r

    # on torch.Tensor only
    if isinstance(tensor, torch.Tensor):
        if pin_memory:
            tensor.pin_memory()
        if device is not None and tensor.device != device:
            tensor = tensor.to(device, non_blocking=non_blocking)
    else:
        pass
    return tensor


def collate_dicts(
        batch: Batch,
        device: torch.device,
        pin_memory: bool = False,
        non_blocking: bool = False) -> Mapping[str, Union[torch.Tensor, List]]:
    """
    Default function to collate a dictionary of samples to a dictionary of torch.Tensor

    Args:
        batch: a dictionary of features
        device: the device where to create the torch.Tensor
        pin_memory: if True, pin the memory. Required to be a `CUDA` allocated torch.Tensor
        non_blocking: if True, use non blocking memory transfer

    Returns:
        a dictionary of torch.Tensor
    """
    assert isinstance(batch, collections.Mapping), 'must be a dictionary like!'

    collated = collections.OrderedDict()
    for name, values in batch.items():
        try:
            collated[name] = collate_tensors(
                values=values,
                device=device,
                pin_memory=pin_memory,
                non_blocking=non_blocking
            )
        except Exception as e:
            # useful debugging info
            info = f'collate_dicts with name={name}. Exception={e}'
            print(info, file=sys.stderr)
            logger.error(info, exc_info=1)
            raise e

    return collated


def collate_list_of_dicts(
        batches: Sequence[Batch],
        device: torch.device,
        pin_memory: bool = False,
        non_blocking: bool = False) -> Mapping[str, Union[torch.Tensor, List]]:
    """
        Default function to collate a list of dictionary to a dictionary of `torch.Tensor`s

        Args:
            batches: a list of dictionary of features
            device: the device where to create the torch.Tensor
            pin_memory: if True, pin the memory. Required to be a `CUDA` allocated torch.Tensor
            non_blocking: if True, use non blocking memory transfer

        Returns:
            a dictionary of torch.Tensor
        """
    assert isinstance(batches, collections.Sequence), f'must be a list of dictionary! Got={type(batches)}'
    assert isinstance(batches[0], collections.Mapping), f'must be a list of dictionary! ' \
                                                        f'Got={type(batches[0])}, str={str(batches[0])}'

    d = collections.OrderedDict()
    for key in batches[0]:
        try:
            bs: Union[torch.Tensor, List] = [b[key] for b in batches]
            bs = collate_tensors(bs, device=device, pin_memory=pin_memory, non_blocking=non_blocking)
            d[key] = bs
        except Exception as e:
            # useful debugging info
            info = f'ERROR collate_list_of_dicts with name={key}. Exception={e}'
            print(info, file=sys.stderr)
            logger.error(info, exc_info=1)
            raise e

    return d


def default_collate_fn(
        batch: Union[Sequence[Any], Mapping[str, Any]],
        device: torch.device,
        pin_memory: bool = False,
        non_blocking: bool = False):
    """

    Args:
        batch: a dictionary of features or a list of dictionary of features
        device: the device where to create the torch.Tensor
        pin_memory: if True, pin the memory. Required to be a `CUDA` allocated torch.Tensor
        non_blocking: if True, use non blocking memory transfer

    Returns:
        a dictionary of torch.Tensor
    """
    if isinstance(batch, collections.Sequence):
        return collate_list_of_dicts(batch, device, pin_memory, non_blocking)

    if isinstance(batch, collections.Mapping):
        return collate_dicts(batch, device, pin_memory, non_blocking)

    raise NotImplementedError()
