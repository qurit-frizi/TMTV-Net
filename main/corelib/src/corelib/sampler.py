import collections
from typing import Dict
import numpy as np


def sample_tiled_volumes(
        case_data: Dict,
        *,
        tile_step=None,
        tile_size: int,
        random_offset=None,
        volume_name: str,
        tile_step_z=None) -> np.ndarray:
    """
    Tiles the whole volumes as a sub-volumes.

    Args:
        case_data: a single patient data
        random_offset: a random offset added to the tile. This is to make sure the sampling of the tile
            is truly random and not biased the patient position within the volume
        tile_size: the size of the file
        tile_step: the position offset between two tiles
    """
    samples = []
    if tile_step is None:
        tile_step = tile_size

    if tile_step_z is None:
        tile_step_z = tile_step

    shape = case_data[volume_name].shape
    assert len(shape) == 3

    if random_offset is not None and random_offset != 0:
        offset = np.random.random_integers(0, random_offset - 1, size=3)
    else:
        offset = [0, 0, 0]

    half_size = tile_size // 2
    for z in range(offset[0], shape[0], tile_step_z):
        for y in range(offset[1], shape[1], tile_step):
            for x in range(offset[2], shape[2], tile_step):
                center_voxel = np.asarray([z, y, x], dtype=int) + half_size
                samples.append(center_voxel)

    return np.asarray(samples)


def sample_random_subvolumes(
        case_data: Dict,
        *,
        tile_size: collections.Sequence,
        nb_samples: int,
        volume_name: str,
        only_valid_z: bool = True,
        only_valid_xy: bool = True,
        ) -> np.ndarray:
    """
    Randomly sample positions within a 3D volume, considering margins (i.e., tile size)

    args:
        only_valid_z: only sample from a valid portion of the image 
            (i.e., all voxels within `offset +/- tile_size / 2` are within the volume in z)
        only_valid_xy: only sample from a valid portion of the image 
            (i.e., all voxels within `offset +/- tile_size / 2` are within the volume in x, y)
    """
    shape = case_data[volume_name].shape
    assert len(shape) == 3

    if not isinstance(tile_size, (np.ndarray, collections.Sequence)):
        half = [tile_size // 2] * 3
    else:
        assert len(tile_size) == 3
        half = np.asarray(tile_size) // 2

    # bias the center to be the next voxel (+1) to be in line with
    # the feature extraction
    if only_valid_z and half[0] < (shape[0] - half[0] + 1):  # very short z-axis volume
        offset_z = np.random.randint(0 + half[0], shape[0] - half[0] + 1, size=nb_samples)
    else:
        offset_z = np.random.randint(0, shape[0] + 1, size=nb_samples)

    if only_valid_xy and half[1] < (shape[1] - half[1] + 1):
        offset_y = np.random.randint(0 + half[1], shape[1] - half[1] + 1, size=nb_samples)
    else:
        offset_y = np.random.randint(0, shape[1] + 1, size=nb_samples)

    if only_valid_xy and half[2] < (shape[2] - half[2] + 1):
        offset_x = np.random.randint(0 + half[2], shape[2] - half[2] + 1, size=nb_samples)
    else:
        offset_x = np.random.randint(0, shape[2] + 1, size=nb_samples)
    
    samples = np.asarray([offset_z, offset_y, offset_x]).transpose()
    return np.asarray(samples)


def sample_random_subvolumes_weighted(
        case_data: Dict,
        *,
        tile_size: collections.Sequence,
        nb_samples: int,
        volume_name: str,
        only_valid_z: bool = True,
        foreground_fraction: float = 0.5,
        bounding_box_name: str = 'bounding_boxes_min_max') -> np.ndarray:
    """
    Randomly sample positions within a 3D volume, considering margins (i.e., tile size) and a foreground class
    """
    shape = case_data[volume_name].shape
    assert len(shape) == 3

    if not isinstance(tile_size, (np.ndarray, collections.Sequence)):
        half = [tile_size // 2] * 3
    else:
        assert len(tile_size) == 3
        half = np.asarray(tile_size) // 2

    offsets_z = []
    offsets_y = []
    offsets_x = []
    bounding_boxes = case_data.get(bounding_box_name)
    assert bounding_boxes is not None, f'cannot find the bounding boxes={bounding_box_name}'
    bounding_boxes = bounding_boxes[0][0]  # remove the NC components
    for _ in range(nb_samples):
        sample_foreground = np.random.rand() < foreground_fraction
        if sample_foreground and bounding_boxes is not None and len(bounding_boxes) > 0:
            # we need to sample from a segmentation bounding box
            # randomly select a bounding box
            bb_min, bb_max = bounding_boxes[np.random.choice(len(bounding_boxes))]
            quarter_z = half[0] // 2
            quarter_y = half[1] // 2
            quarter_x = half[2] // 2

            offset_z = np.random.randint(bb_min[0] - quarter_z, bb_max[0] + 1 + quarter_z)
            offset_y = np.random.randint(bb_min[1] - quarter_y, bb_max[1] + 1 + quarter_y)
            offset_x = np.random.randint(bb_min[2] - quarter_x, bb_max[2] + 1 + quarter_x)

        else:
            # randomly sample inside the FoV
            # bias the center to be the next voxel (+1) to be in line with
            # the feature extraction
            if only_valid_z and half[0] < (shape[0] - half[0] + 1):
                offset_z = np.random.randint(0 + half[0], shape[0] - half[0] + 1)
            else:
                offset_z = np.random.randint(0, shape[0] + 1)
                
            offset_y = np.random.randint(0 + half[1], shape[1] - half[1] + 1)
            offset_x = np.random.randint(0 + half[2], shape[2] - half[2] + 1)

        offsets_z.append(offset_z)
        offsets_y.append(offset_y)
        offsets_x.append(offset_x)

    samples = np.asarray([offsets_z, offsets_y, offsets_x]).transpose()
    return samples

