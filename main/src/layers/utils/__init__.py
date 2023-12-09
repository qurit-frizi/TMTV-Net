"""
This package will contain utility function that only depends on numpy and pytorch
"""
import collections

import numpy as np
import torch

from .upsample import upsample
from .clamp_n import clamp_n
from .sub_tensor import sub_tensor
from .flatten import flatten
from .global_pooling import global_max_pooling_2d, global_average_pooling_2d, global_average_pooling_3d, \
    global_max_pooling_3d
from .batch_pad import batch_pad, batch_pad_joint, batch_pad_torch, batch_pad_numpy
from .batch_pad_minmax import batch_pad_minmax, batch_pad_minmax_joint, batch_pad_minmax_numpy, batch_pad_minmax_torch
from .safe_filename import safe_filename
from .optional_import import optional_import
from .requires import torch_requires
from .load_module import find_global_name
from .number_formatting import bytes2human, number2human


def collect_hierarchical_module_name(base_name, model, module_to_name=None):
    """
    Create a meaningful name of the module based on the module hierarchy

    Args:
        base_name: the base name
        model: the model
        module_to_name: where to store the module to name conversion

    Returns:
        a dictionary with mapping nn.Module to string
    """
    if module_to_name is None:
        module_to_name = collections.OrderedDict()

    module_to_name[model] = base_name
    for child_id, child in enumerate(model.children()):
        child_name = base_name + '/' + type(child).__name__ + f'_{child_id}'
        collect_hierarchical_module_name(child_name, child, module_to_name=module_to_name)

    return module_to_name


def collect_hierarchical_parameter_name(base_name, model, parameter_to_name=None, with_grad_only=False):
    """
        Create a meaningful name of the module's parameters based on the module hierarchy

        Args:
            base_name: the base name
            model: the model
            parameter_to_name: where to store the module to name conversion
            with_grad_only: only the parameters requiring gradient are collected

        Returns:
            a dictionary with mapping nn.Parameter to string
        """
    if parameter_to_name is None:
        parameter_to_name = collections.OrderedDict()

    for child_id, child in enumerate(model.children()):
        child_name = base_name + '/' + type(child).__name__ + f'_{child_id}'
        for name, parameter in child.named_parameters(recurse=False):
            if with_grad_only and not parameter.requires_grad:
                # discard if not gradient
                continue
            parameter_name = child_name + '/' + name
            parameter_to_name[parameter] = parameter_name

        collect_hierarchical_parameter_name(
            child_name,
            child,
            parameter_to_name=parameter_to_name,
            with_grad_only=with_grad_only)

    return parameter_to_name


def get_batch_n(split, nb_samples, indices, transforms, use_advanced_indexing):
    """
    Collect the split indices given and apply a series of transformations

    Args:
        nb_samples: the total number of samples of split
        split: a mapping of `np.ndarray` or `torch.Tensor`
        indices: a list of indices as numpy array
        transforms: a transformation or list of transformations or None
        use_advanced_indexing: if True, use the advanced indexing mechanism else
            use a simple list (original data is referenced)
            advanced indexing is typically faster for small objects, however for large objects (e.g., 3D data)
            the advanced indexing makes a copy of the data making it very slow.

    Returns:
        a split with the indices provided
    """
    data = {}
    for split_name, split_data in split.items():
        if isinstance(split_data, (torch.Tensor, np.ndarray)) and len(split_data) == nb_samples:
            # here we prefer [split_data[i] for i in indices] over split_data[indices]
            # this is because split_data[indices] will make a deep copy of the data which may be time consuming
            # for large data
            if use_advanced_indexing:
                split_data = split_data[indices]
            else:
                split_data = [[split_data[i]] for i in indices]
        if isinstance(split_data, list) and len(split_data) == nb_samples:
            split_data = [split_data[i] for i in indices]

        data[split_name] = split_data

    if transforms is None:
        # do nothing: there is no transform
        pass
    elif isinstance(transforms, collections.Sequence):
        # we have a list of transforms, apply each one of them
        for transform in transforms:
            data = transform(data)
    else:
        # anything else should be a functor
        data = transforms(data)

    return data


def to_value(v):
    """
    Convert where appropriate from tensors to numpy arrays

    Args:
        v: an object. If ``torch.Tensor``, the tensor will be converted to a numpy
            array. Else returns the original ``v``

    Returns:
        ``torch.Tensor`` as numpy arrays. Any other type will be left unchanged
    """
    if isinstance(v, torch.Tensor):
        return v.cpu().data.numpy()
    return v


def safe_lookup(dictionary, *keys, default=None):
    """
    Recursively access nested dictionaries

    Args:
        dictionary: nested dictionary
        *keys: the keys to access within the nested dictionaries
        default: the default value if dictionary is ``None`` or it doesn't contain
            the keys

    Returns:
        None if we can't access to all the keys, else dictionary[key_0][key_1][...][key_n]
    """
    if dictionary is None:
        return default

    for key in keys:
        dictionary = dictionary.get(key)
        if dictionary is None:
            return default

    return dictionary


def recursive_dict_update(dict, dict_update):
    """
    This adds any missing element from ``dict_update`` to ``dict``, while keeping any key not
        present in ``dict_update``

    Args:
        dict: the dictionary to be updated
        dict_update: the updated values
    """
    for updated_name, updated_values in dict_update.items():
        if updated_name not in dict:
            # simply add the missing name
            dict[updated_name] = updated_values
        else:
            values = dict[updated_name]
            if isinstance(values, collections.Mapping):
                # it is a dictionary. This needs to be recursively
                # updated so that we don't remove values in the existing
                # dictionary ``dict``
                recursive_dict_update(values, updated_values)
            else:
                # the value is not a dictionary, we can update directly its value
                dict[updated_name] = values


def len_batch(batch):
    """

    Args:
        batch: a data split or a `collections.Sequence`

    Returns:
        the number of elements within a data split
    """
    if isinstance(batch, (collections.Sequence, torch.Tensor)):
        return len(batch)

    assert isinstance(batch, collections.Mapping), 'Must be a dict-like structure! got={}'.format(type(batch))

    for name, values in batch.items():
        if isinstance(values, (list, tuple)):
            return len(values)
        if isinstance(values, torch.Tensor) and len(values.shape) != 0:
            return values.shape[0]
        if isinstance(values, np.ndarray) and len(values.shape) != 0:
            return values.shape[0]
    return 0


def flatten_nested_dictionaries(d, root_name='', delimiter='-'):
    """
    Recursively flatten a dictionary of arbitrary nested size into a flattened dictionary
    of nested size 1

    Args:
        d: a dictionary
        root_name: the root name to be appended of the keys of d
        delimiter: use this string as delimiter to concatenate nested dictionaries

    Returns:
        a dictionary of maximum depth 1
    """
    assert isinstance(d, collections.Mapping)
    flattened = collections.OrderedDict()
    for name, value in d.items():
        if len(root_name) == 0:
            full_name = name
        else:
            full_name = f'{root_name}{delimiter}{name}'

        if isinstance(value, collections.Mapping):
            sub_flattened = flatten_nested_dictionaries(value, root_name=full_name, delimiter=delimiter)
            flattened.update(sub_flattened)
        else:
            flattened[full_name] = value
    return flattened


class ExceptionAbortRun(BaseException):
    """
    The run has been pruned due to performance reason
    """
    def __init__(self, history, metrics=None, reason=None):
        self.reason = reason
        self.history = history
        self.metrics = metrics

    def __str__(self):
        return f'ExceptionAbortRun(reason={self.reason})'
