"""
This module is dedicated to data augmentations. In particular we strive to have a numpy and pytorch
implementation for each augmentation so that we could if perform it on GPU

Transforms are designed to work for n-dimensional data.
"""

from .crop import transform_batch_random_crop, batch_crop
from layers.utils import batch_pad_numpy, batch_pad_torch, batch_pad
from .flip import flip
from .copy import copy
from .cutout_function import cutout, cutout_random_ui8_torch, cutout_value_fn_constant, cutout_random_size
from .resize import resize
from .stack import stack
from .normalize import normalize
from .renormalize import renormalize
from .resample import resample_3d, resample_spatial_info, affine_grid_fixed_to_moving
from .deform import deform_image_random, random_grid_using_control_points
from .affine import affine_transformation_translation, affine_transformation_rotation2d, affine_transformation_scale, \
    affine_transform, to_voxel_space_transform, apply_homogeneous_affine_transform, \
    apply_homogeneous_affine_transform_zyx
from .spatial_info import SpatialInfo

from .transforms import Transform, TransformBatchWithCriteria, criteria_feature_name, criteria_is_array_4_or_above, \
    criteria_is_array_n_or_above, criteria_is_tensor
from .transforms_random_crop_pad import TransformRandomCropPad
from .transforms_random_flip import TransformRandomFlip
from .transforms_random_cutout import TransformRandomCutout
from .transforms_resize import TransformResize
from .transforms_normalize_intensity import TransformNormalizeIntensity
from .transforms_compose import TransformCompose
from .transforms_affine import TransformAffine
from .transforms_cast import TransformCast
from .transforms_random_crop_resize import TransformRandomCropResize
from .transforms_resize_modulo_pad_crop import TransformResizeModuloCropPad
from .transforms_resample import TransformResample, random_fixed_geometry_within_geometries, find_largest_geometry
from .transforms_one_of import TransformOneOf
from .transforms_random_deformation import TransformRandomDeformation
from .transforms_squeeze import TransformSqueeze
from .transforms_unsqueeze import TransformUnsqueeze
from .transform_move_to_device import TransformMoveToDevice