import collections
import functools
from .transforms import CriteriaFn
from typing import Union, Callable, Dict, Sequence

from typing_extensions import Literal

from transforms import transforms
from transforms.resample import resample_3d
from transforms.spatial_info import SpatialInfo
from basic_typing import Numeric, Batch, Length, ShapeX
import numpy as np

"""Functor to retrieve spatial information from a batch and tensor name"""
get_spatial_info_type = Callable[[Batch, str], SpatialInfo]

"""Represent a background value as numeric or numeric by tensor name (i.e., tensor dependent background value)"""
constant_background_value_type = Union[Numeric, Dict[str, Numeric]]


def _transform_resample_fn(
        feature_names,
        batch,
        resampling_geometry,
        get_spatial_info_from_batch_name,
        interpolation_mode,
        padding_mode):
    geometry_by_name = {}
    for name in feature_names:
        geometry = get_spatial_info_from_batch_name(batch, name)
        geometry_by_name[name] = geometry

    if not isinstance(resampling_geometry, SpatialInfo):
        resampling_geometry = resampling_geometry(geometry_by_name)

    batch_transformed = collections.OrderedDict()
    for name in feature_names:
        geometry = geometry_by_name[name]
        v = batch[name]
        assert len(v.shape) == 5, 'Must be a NCDHW shape'
        if v.shape[0] != 1:
            raise NotImplementedError('Only single volume per batch is implemented!')
        if v.shape[1] != 1:
            raise NotImplementedError('Only single is implemented!')

        min_bb_mm = resampling_geometry.origin
        max_bb_mm = np.asarray(resampling_geometry.origin) + \
                    np.asarray(resampling_geometry.spacing) * np.asarray(resampling_geometry.shape)
        v_r = resample_3d(
            v[0, 0],  # single volume single channel tensor
            np_volume_spacing=geometry.spacing,
            np_volume_origin=geometry.origin,
            min_bb_mm=min_bb_mm,
            max_bb_mm=max_bb_mm,
            resampled_spacing=resampling_geometry.spacing,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
        )

        # put back the N and C channels
        v_r = v_r.reshape([1, 1] + list(v_r.shape))
        batch_transformed[name] = v_r

    for feature_name, feature_value in batch.items():
        if feature_name not in feature_names:
            # not in the transformed features, so copy the original value
            batch_transformed[feature_name] = feature_value
    return batch_transformed


def find_largest_geometry(geometries: Sequence[SpatialInfo]) -> SpatialInfo:
    assert geometries is not None
    assert len(geometries) > 0

    if len(geometries) == 1:
        return geometries[0]

    max_volume = -1
    max_volume_geometry = None
    for g in geometries:
        extent = np.asarray(g.spacing) * np.asarray(g.shape)
        volume = np.prod(extent)
        if volume > max_volume:
            max_volume = volume
            max_volume_geometry = g
    assert max_volume_geometry is not None
    return max_volume_geometry


def random_fixed_geometry_within_geometries(
        geometries: Dict[str, SpatialInfo],
        fixed_geometry_shape: ShapeX,
        fixed_geometry_spacing: Length,
        geometry_selector: Callable[[Sequence[SpatialInfo]], SpatialInfo] = find_largest_geometry):
    """
    Place randomly a fixed geometry within the largest available geometry.

    Args:
        geometries: a dictionary of available geometries
        fixed_geometry_shape: the shape of the returned geometry
        fixed_geometry_spacing: the spacing of the geometry
        geometry_selector: select a geometry for the random geometry calculation

    Returns:
        a geometry
    """
    reference_geometry = geometry_selector(list(geometries.values()))
    reference_origin = np.asarray(reference_geometry.origin)
    extent_available = np.asarray(reference_geometry.spacing) * np.asarray(reference_geometry.shape)
    extent = np.asarray(fixed_geometry_spacing) * np.asarray(fixed_geometry_shape)
    origin_range = extent_available - extent
    origin_range[origin_range < 0] = 1e-3
    random_origin = [np.random.uniform(0, r) for r in origin_range]

    return SpatialInfo(
        origin=reference_origin + random_origin,
        shape=fixed_geometry_shape,
        spacing=fixed_geometry_spacing
    )


class TransformResample(transforms.TransformBatchWithCriteria):
    """
    Resample a tensor with spatial information (e.g., a 3D volume with origin and spacing)
    """

    def __init__(
            self,
            resampling_geometry: Union[SpatialInfo, Callable[[Dict[str, SpatialInfo]], SpatialInfo]],
            get_spatial_info_from_batch_name: get_spatial_info_type,
            criteria_fn: CriteriaFn = transforms.criteria_is_array_4_or_above,
            interpolation_mode: Literal['linear', 'nearest'] = 'linear',
            padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros'):
        """
        Args:
            resampling_geometry: the geometric info to be used for the resampling. This can be a fixed
                geometry or a function taking as input a dict of geometries (for each selected volume)
            get_spatial_info_from_batch_name: function to calculate the spatial info of a (batch, name)
            criteria_fn: how to select the features to transform. If `None` transform all arrays with dim >= 3
            interpolation_mode:  interpolation mode
            padding_mode: indicate how to pad missing data
        """
        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=functools.partial(
                _transform_resample_fn,
                resampling_geometry=resampling_geometry,
                get_spatial_info_from_batch_name=get_spatial_info_from_batch_name,
                interpolation_mode=interpolation_mode,
                padding_mode=padding_mode))
