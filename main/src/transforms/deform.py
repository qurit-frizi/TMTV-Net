import math

import warnings

import torch
from filter_gaussian import FilterGaussian
from typing import Union, Sequence, Optional, List
from typing_extensions import Literal
import numpy as np

from .resize import resize
from .spatial_info import SpatialInfo
from .resample import affine_grid_fixed_to_moving
from compatibility import grid_sample
from basic_typing import TorchTensorNCX, ShapeNX
from utils import batch_pad


def random_grid_using_control_points(
        shape: ShapeNX,
        control_points: Union[int, Sequence[int]],
        max_displacement: Optional[Union[float, Sequence[float]]] = None,
        geometry_moving: Optional[SpatialInfo] = None,
        tfm: Optional[torch.Tensor] = None,
        geometry_fixed: Optional[SpatialInfo] = None,
        gaussian_filter_sigma: Optional[float] = None,
        align_corners: bool = False) -> torch.Tensor:
    """
    Generate random deformation grid (one for each sample)
    based on the number of control points and maximum displacement
    of the control points.

    This is done by decomposing the affine (grid) and deformable components.

    The gradient can be back-propagated through this transform.

    Notes:
        The deformation field's max_displacement will not rotate
        according to geometry_fixed but will be axis aligned.

    Args:
        control_points: the control points spread on the image at regularly
            spaced intervals with random `max_displacement` magnitude
        max_displacement: specify the maximum displacement of a control point. Range [-1..1]
        geometry_moving: defines the geometry of an image. In particular to handle non-isotropic spacing
        align_corners: should be False. The (0, 0) is the center of a voxel
        shape: the shape of the moving geometry. Must match the `geometry_moving` if specified
        geometry_moving: geometry of the moving object. If None, default to a geometry
            of spacing 1 and origin 0
        geometry_fixed: geometry output (dictate the final geometry). If None, use the same
            as the geometry_moving
        tfm: the transformation to be applied to the `geometry_moving`
        gaussian_filter_sigma: if not None, smooth the deformation field using a gaussian filter.
            The smoothing is done in the control point space

    Returns:
        N * X * dim displacement field
    """
    dim = len(shape) - 1
    if geometry_moving is None:
        geometry_moving = SpatialInfo(shape=shape[1:])
    if geometry_fixed is None:
        geometry_fixed = geometry_moving
    if tfm is None:
        tfm = torch.eye(dim + 1, dtype=torch.float32)
    assert (np.asarray(shape[1:]) == np.asarray(geometry_moving.shape)).all(), \
        f'shape mismatch! Got={geometry_moving.shape}, expected={shape}'

    filter_gaussian = None
    if gaussian_filter_sigma is not None:
        filter_gaussian = FilterGaussian(
            input_channels=1,
            nb_dims=dim,
            sigma=gaussian_filter_sigma,
            kernel_sizes=int(2 * math.ceil(gaussian_filter_sigma) + 1))

    dtype = torch.float32
    grid = affine_grid_fixed_to_moving(
        geometry_moving=geometry_moving,
        geometry_fixed=geometry_fixed,
        tfm=tfm,
        align_corners=align_corners
    ).type(dtype)

    if isinstance(control_points, int):
        control_points = [control_points] * dim
    else:
        assert len(control_points) == dim
    control_points = np.asarray(control_points)

    if max_displacement is None:
        max_displacement = 2.0 / (2.0 * control_points * 5)

    if isinstance(max_displacement, float):
        max_displacement = [max_displacement] * dim
    else:
        assert len(max_displacement) == dim
    max_displacement = np.asarray(max_displacement)

    if max(max_displacement) > 1 or max(max_displacement) < -1:
        warnings.warn('displacement should be in [-1..1] range. -1 left, +1 right')

    # fill the displacement map with random uniform values
    shape_def = [shape[0]] + list(control_points) + [dim]
    deformable_component = torch.zeros(shape_def, dtype=dtype)
    for i in range(dim):
        deformable_component[..., i].uniform_(-max_displacement[i], max_displacement[i])
        if filter_gaussian is not None:
            v = deformable_component[..., i].unsqueeze(1)
            v[:] = filter_gaussian(v)[:]

    # add a margin to make the displacement 0 around the edges
    # then interpolate the deformable displacement
    # conversion from nxc to ncx to nxc
    deformable_component_ncx = deformable_component.permute((0, 3, 1, 2))
    deformable_component_ncx = batch_pad(deformable_component_ncx, [0] + [1] * dim, mode='constant', constant_value=0)
    deformable_component_ncx = resize(deformable_component_ncx, size=shape[1:], mode='linear')
    resized_deformable_component = deformable_component_ncx.permute((0, 2, 3, 1))

    # finally the full transformation is defined by affine and
    # the deformable components
    #
    # IMPORTANT: the deformable component has no knowledge of the
    # geometry. For speed reason, the displacement is not rotated
    # by the geometry's rotational component
    return grid + resized_deformable_component


def deform_image_random(
        moving_volumes: List[TorchTensorNCX],
        control_points: Union[int, Sequence[int]],
        max_displacement: Optional[Union[float, Sequence[float]]] = None,
        geometry: Optional[SpatialInfo] = None,
        interpolation: Literal['linear', 'nearest'] = 'linear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
        gaussian_filter_sigma: Optional[float] = None,
        align_corners: bool = False) -> List[TorchTensorNCX]:
    """
    Non linearly deform an image based on a grid of control points.

    The grid of control points is first uniformly mapped to span the whole image, then
    the control point position will be randomized using `max_displacement`. To avoid
    artifacts at the image boundary, a control point is added with 0 max displacement all
    around the image.

    The gradient can be back-propagated through this transform.

    Notes:
        The deformation field's max_displacement will not rotate
        according to `geometry_fixed` but instead is axis aligned.

    Args:
        moving_volumes: a list of moving volumes. All volumes will be
            deformed using the same deformation field
        control_points: the control points spread on the image at regularly
            spaced intervals with random `max_displacement` magnitude
        max_displacement: specify the maximum displacement of a control point. Range [-1..1]. If None, use
            the moving volume shape and number of control points to calculate appropriate small deformation
            field
        geometry: defines the geometry of an image. In particular to handle non-isotropic spacing
        interpolation: the interpolation of the image with displacement field
        padding_mode: how to handle data outside the volume geometry
        align_corners: should be False. The (0, 0) is the center of a voxel
        gaussian_filter_sigma: if not None, smooth the deformation field using a gaussian filter.
            The smoothing is done in the control point space

    Returns:
        a deformed image
    """
    volume_ref = moving_volumes[0]
    for v in moving_volumes[1:]:
        assert v.shape == volume_ref.shape
        assert v.device == volume_ref.device
        assert v.dtype == volume_ref.dtype

    shape_nx = [volume_ref.shape[0]] + list(volume_ref.shape[2:])
    grid = random_grid_using_control_points(
        shape_nx,
        control_points=control_points,
        max_displacement=max_displacement,
        geometry_moving=geometry,
        gaussian_filter_sigma=gaussian_filter_sigma
    ).to(moving_volumes[0].device)

    if interpolation == 'linear':
        interpolation = 'bilinear'  # type: ignore

    all_resampled_torch = []
    for v in moving_volumes:
        resampled_torch = grid_sample(
            v.type(grid.dtype),
            grid,
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners)
        all_resampled_torch.append(resampled_torch)

    return all_resampled_torch
