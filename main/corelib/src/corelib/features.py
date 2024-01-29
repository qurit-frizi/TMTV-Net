from typing import Optional, Sequence
import torch
from basic_typing import Batch
from .features_generic import feature_3d_z, feature_2d_slices
import numpy as np


def create_3d_features(case_data: dict, index: np.ndarray, half_size: int, half_size_z: Optional[int] = None) -> Batch:
    features = {}
    assert index.shape == (3,)

    if half_size_z is None:
        half_size_z = half_size

    volumes = [name for name, value in case_data.items() if isinstance(value, torch.Tensor) and len(value.shape) >= 3]
    for v_name in volumes:
        padding_value = 0
        features[v_name] = feature_3d_z(case_data[v_name], index_np=index, half_size=half_size, half_size_z=half_size_z, padding_value=padding_value)

    features['uid'] = case_data['uid']
    features['index'] = torch.from_numpy(index).unsqueeze(0)
    return features


def create_2d_slices(case_data: dict, index: np.ndarray, num_slices: int = None) -> Batch:
    features = {}
    assert index.shape == (3,)

    volumes = [name for name, value in case_data.items() if isinstance(value, torch.Tensor) and len(value.shape) == 3]
    for v_name in volumes:
        padding_value = 0
        features[v_name] = feature_2d_slices(case_data[v_name], index_np=index, num_slices=num_slices, padding_value=padding_value)

    features['uid'] = case_data['uid']
    features['index'] = torch.from_numpy(index).unsqueeze(0)
    return features
