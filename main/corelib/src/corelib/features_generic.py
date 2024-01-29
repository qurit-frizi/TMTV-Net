from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F


def feature_2d_slices(volume: torch.Tensor, index_np: Sequence[int], padding_value, num_slices: int = None) -> torch.Tensor:
    """
    Extract slices from a volume

    Returns:
        a tensor of shape (1, 1, num_slices, [Height], [Width])
    """
    z = index_np[0]
    z_half = num_slices // 2
    assert len(volume.shape) == 3

    if z < -z_half or z > volume.shape[0] + z_half:
        # completely outside FoV
        v = torch.full([1, 1, num_slices, volume.shape[1], volume.shape[2]],
                       padding_value,
                       dtype=volume.dtype,
                       device=volume.device)
        return v

    # avoid `-1` as this will loop around
    s = volume[max(0, z - z_half):z + z_half + 1].unsqueeze(0).unsqueeze(0)

    if s.shape[1] != num_slices:
        # we are partially outside the FoV, padd
        # with empty slices
        padding_z_min = max(0 - (z - z_half), 0)
        padding_z_max = max((z + z_half) - (volume.shape[0] - 1), 0)

        padding = [
            0, 0,
            0, 0,
            padding_z_min, padding_z_max
        ]
        s = F.pad(s, padding, mode='constant', value=padding_value)        

    assert s.shape[1] == num_slices, f's.shape={s.shape}, should have slice={num_slices}'
    return s


def feature_3d_z(volume: torch.Tensor, index_np: Sequence[int], half_size: int, padding_value, half_size_z: Optional[int] = None) -> torch.Tensor:
    """
    Create a 3D sub-volume (DHW) or (CDHW) format

    Returns:
        a tensor of shape (1, C, 2 * half_size_z, 2 * half_size, 2 * half_size)
    """
    volume_shape = np.asarray(volume.shape[-3:])
    if len(volume.shape) == 3:
        # reformat to CDHW format for uniformity
        volume = volume.unsqueeze(0)
    nb_c = volume.shape[0]

    full_size = 2 * half_size
    if half_size_z is None:
        half_size_z = half_size

    if half_size_z == 1:
        # special case: extract a single slice for 2D models
        full_size_z = 1
        half_min = np.asarray([0, half_size, half_size])
        half_max = np.asarray([1, half_size, half_size])
    else:
        full_size_z = 2 * half_size_z
        half_min = np.asarray([half_size_z, half_size, half_size])
        half_max = np.asarray([half_size_z, half_size, half_size])

    index_np = np.asarray(index_np)
    assert len(index_np) == 3

    if (index_np > half_max + volume_shape).any() or (index_np < -half_min).any():
        # completely outside FoV
        v = torch.full([1, nb_c, full_size_z, full_size, full_size],
                       padding_value,
                       dtype=volume.dtype,
                       device=volume.device)
        return v

    min_corner = index_np - half_min
    max_corner = index_np + half_max

    min_corner_safe = [max(0, min_corner[0]), max(0, min_corner[1]), max(0, min_corner[2])]  # deal with negative min indices
    sub_volume = volume[:, min_corner_safe[0]:max_corner[0], min_corner_safe[1]:max_corner[1], min_corner_safe[2]:max_corner[2]]

    if min(sub_volume.shape) == 0:
        # completely outside, return a background value
        v = torch.full([1, nb_c, full_size_z, full_size, full_size], padding_value, dtype=volume.dtype, device=volume.device)
        return v

    # BEWARE: padding is defined in reverse axes
    padding = [
        max(0, -min_corner[2]), max(0, max_corner[2] - volume_shape[2]),
        max(0, -min_corner[1]), max(0, max_corner[1] - volume_shape[1]),
        max(0, -min_corner[0]), max(0, max_corner[0] - volume_shape[0])
    ]

    if max(padding) > 0 or min(padding) < 0:
        # check if we need to pad a region of the data (i.e., out of FOV)
        sub_volume = F.pad(sub_volume, padding, mode='constant', value=padding_value)

    assert sub_volume.shape[-3:] == (full_size_z, full_size, full_size), f'size={sub_volume.shape[-3:]}, expected=({full_size_z}, {full_size}, {full_size}), padding={padding}'
    # add the `N` component
    return sub_volume.unsqueeze(dim=0)
