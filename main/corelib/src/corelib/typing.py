from numbers import Number
from typing import Sequence, Union, Dict, Any, List, Optional, Tuple, Mapping

import numpy as np
import torch

"""Generic numeric type"""
Numeric = Union[int, float]

"""Generic Shape"""
Shape = Sequence[int]

"""Shape expressed as [N, C, D, H, W, ...] components"""
ShapeNCX = Sequence[int]

"""Shape expressed as [C, D, H, W, ...] components"""
ShapeCX = Sequence[int]

"""Shape expressed as [D, H, W, ...] components"""
ShapeX = Sequence[int]

"""Shape expressed as [N, D, H, W, ...] components (the component `C` is removed)"""
ShapeNX = Sequence[int]

"""Generic Tensor as numpy or torch"""
Tensor = Union[np.ndarray, torch.Tensor]

"""Generic Tensor as numpy or torch. Must be shaped as [N, C, D, H, W, ...]"""
TensorNCX = Union[np.ndarray, torch.Tensor]

"""Generic Tensor as numpy or torch. Must be shaped as [C, D, H, W, ...]"""
TensorCX = Union[np.ndarray, torch.Tensor]

"""Generic Tensor as numpy or torch. Must be shaped as 2D array [N, X]"""
TensorNX = Union[np.ndarray, torch.Tensor]

"""Generic Tensor with th `N` and `C` components removed"""
TensorX = Union[np.ndarray, torch.Tensor]


"""Torch Tensor. Must be shaped as [N, C, D, H, W, ...]"""
TorchTensorNCX = torch.Tensor

"""Torch Tensor. Must be shaped as 2D array [N, X]"""
TorchTensorNX = torch.Tensor

"""Torch Tensor with th `N` and `C` components removed"""
TorchTensorX = torch.Tensor


"""Numpy Tensor. Must be shaped as [N, C, D, H, W, ...]"""
NumpyTensorNCX = np.ndarray

"""Numpy Tensor. Must be shaped as 2D array [N, X]"""
NumpyTensorNX = np.ndarray

"""Numpy Tensor with th `N` and `C` components removed"""
NumpyTensorX = np.ndarray

"""Represent a dictionary of (key, value)"""
Batch = Mapping[str, Any]

"""Length shaped as D, H, W, ..."""
Length = Union[Sequence[float], np.ndarray, torch.Tensor]

"""Represent a data split, a dictionary of any value"""
Split = Any

"""Represent a dataset which is composed of named data splits"""
Dataset = Dict[str, Split]
DatasetInfo = Dict[str, Any]

"""Represent a collection of datasets"""
Datasets = Dict[str, Dataset]
DatasetsInfo = Dict[str, DatasetInfo]

HistoryStep = Dict[str, Number]
History = List[HistoryStep]

Activation = Any


IntTupleList = List[Tuple[int, ...]]
IntListList = List[List[int]]

ConvKernels = Union[int, Sequence[int], IntTupleList]
ConvStrides = ConvKernels
PoolingSizes = Optional[ConvKernels]

Stride = Union[int, Tuple[int, ...]]
KernelSize = Union[int, Tuple[int, ...]]
Padding = Union[int, str, Tuple[int, ...]]
Paddings = Union[Padding, Sequence[int], Sequence[str], IntTupleList]

