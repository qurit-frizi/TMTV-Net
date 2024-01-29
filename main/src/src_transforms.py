from argparse import Namespace
import time
from typing import List, Optional
import torch
from corelib import create_2d_slices, create_3d_features
from basic_typing import Batch
from corelib import sample_random_subvolumes, sample_random_subvolumes_weighted
import numpy as np
from collate import default_collate_fn


def transform_report_time(batch: Batch, transforms: Optional[List]) -> Batch:
    """
    Report the time it took to run the transforms
    """
    if transforms is None or len(transforms) == 0:
        return batch

    if not isinstance(transforms, list):
        transforms = [transforms]

    time_start = time.perf_counter()
    for transform in transforms:
        batch = transform(batch)
    time_end = time.perf_counter()

    #print(f'transform time={time_end - time_start}')
    return batch


def transform_feature_2d_slice(batch: Batch, configuration: Namespace, sample_volume_name: str, only_valid_z=True) -> Batch:
    """
    Extract 2D slices randomly from the volumes
    """
    time_start = time.perf_counter()
    with torch.no_grad():
        fov_half_size = configuration.data['fov_half_size']
        num_slices = configuration.data['num_slices']
        
        positions = sample_random_subvolumes(
            batch, 
            nb_samples=configuration.data['samples_per_patient'], 
            tile_size = fov_half_size * 2, 
            volume_name=sample_volume_name,
            only_valid_z=only_valid_z
        )
        
        features = []
        for p in positions:
            f = create_2d_slices(batch, index=p, num_slices=num_slices)
            features.append(f)

        features = default_collate_fn(features, device=torch.device('cpu'))
    time_end = time.perf_counter()

    #print(f'process (slice2d)={os.getpid()} time={time_end - time_start}')
    return features


def transform_feature_3d(batch: Batch, configuration: Namespace, sample_volume_name: str, only_valid_z=True) -> Batch:
    """
    Extract 3D sub-volumes randomly
    """
    time_start = time.perf_counter()
    with torch.no_grad():
        fov_half_size = configuration.data['fov_half_size']     
        fov_half_size_z = fov_half_size
        positions = sample_random_subvolumes(
            batch, 
            nb_samples=configuration.data['samples_per_patient'], 
            tile_size = fov_half_size * 2,
            volume_name=sample_volume_name,
            only_valid_z=only_valid_z
        )
        if isinstance(fov_half_size, np.ndarray):
            assert len(fov_half_size) == 3
            assert fov_half_size[1] == fov_half_size[2], 'XY must be the same!'
            fov_half_size_z = fov_half_size[0]
            fov_half_size = fov_half_size[1]
        
        features = []
        for p in positions:
            f = create_3d_features(batch, index=p, half_size=fov_half_size, half_size_z=fov_half_size_z)
            features.append(f)

        features = default_collate_fn(features, device=torch.device('cpu'))
    time_end = time.perf_counter()

    #print(f'process (patch3d)={os.getpid()} time={time_end - time_start}')
    return features


def transform_feature_3d_v2(batch: Batch, configuration: Namespace, sample_volume_name: str, only_valid_z=True, nb_samples: int = 1) -> Batch:
    """
    Extract 3D sub-volumes randomly
    """
    time_start = time.perf_counter()
    with torch.no_grad():
        fov_half_size = configuration.data['fov_half_size']     
        fov_half_size_z = fov_half_size
        positions = sample_random_subvolumes(
            batch, 
            nb_samples=nb_samples, 
            tile_size = fov_half_size * 2,
            volume_name=sample_volume_name,
            only_valid_z=only_valid_z,
        )
        if isinstance(fov_half_size, np.ndarray):
            assert len(fov_half_size) == 3
            assert fov_half_size[1] == fov_half_size[2], 'XY must be the same!'
            fov_half_size_z = fov_half_size[0]
            fov_half_size = fov_half_size[1]
        
        features = []
        for p in positions:
            f = create_3d_features(batch, index=p, half_size=fov_half_size, half_size_z=fov_half_size_z)
            features.append(f)

        features = default_collate_fn(features, device=torch.device('cpu'))
    time_end = time.perf_counter()

    #print(f'process (patch3d)={os.getpid()} time={time_end - time_start}')
    return features


def transform_feature_3d_resampled(
        batch: Batch, 
        configuration: Namespace, 
        sample_volume_name: str, 
        only_valid_z: bool = True, 
        nb_samples: int = 1,
        foreground_fraction=0.5) -> Batch:
    """
    Extract 3D sub-volumes randomly
    """
    time_start = time.perf_counter()
    with torch.no_grad():
        fov_half_size = configuration.data['fov_half_size']     
        fov_half_size_z = fov_half_size
        positions = sample_random_subvolumes_weighted(
            batch, 
            tile_size = fov_half_size * 2,
            volume_name=sample_volume_name,
            only_valid_z=only_valid_z,
            nb_samples=nb_samples,
            foreground_fraction=foreground_fraction
        )
        if isinstance(fov_half_size, np.ndarray):
            assert len(fov_half_size) == 3
            assert fov_half_size[1] == fov_half_size[2], 'XY must be the same!'
            fov_half_size_z = fov_half_size[0]
            fov_half_size = fov_half_size[1]
        
        features = []
        for p in positions:
            f = create_3d_features(batch, index=p, half_size=fov_half_size, half_size_z=fov_half_size_z)
            features.append(f)

        features = default_collate_fn(features, device=torch.device('cpu'))
    time_end = time.perf_counter()

    #print(f'process (patch3d)={os.getpid()} time={time_end - time_start}')
    return features


def transform_strip_features(batch: Batch, features_to_keep: List[str]) -> Batch:
    """
    Keep the minimal number of features needed, this is to accelerate data augmentation
    """
    b = {name: value for name, value in batch.items() if name in features_to_keep}
    return b
