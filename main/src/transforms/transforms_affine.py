import collections
from numbers import Number

from typing import Sequence, Optional

import numpy as np
import torch
from .transforms import CriteriaFn
from transforms import transforms
from transforms import affine


def rand_n_2(n_min_max):
    """
    Create random values for a list of min/max pairs

    Args:
        n_min_max: a matrix of size N * 2 representing N points where `n_min_max[:, 0]` are the minimum values
            and `n_min_max[:, 1]` are the maximum values

    Returns:
        `N` random values in the defined interval

    Examples:
        To return 3 random values in interval [-1..10], [-2, 20], [-3, 30] respectively:
        >>> n_min_max = np.asarray([[-1, 10], [-2, 20], [-3, 30]])
        >>> rand_n_2(n_min_max)
    """
    assert len(n_min_max.shape) == 2
    assert n_min_max.shape[1] == 2, 'must be min/max'

    r = np.random.rand(len(n_min_max))
    scaled_r = np.multiply(r, (n_min_max[:, 1] - n_min_max[:, 0])) + n_min_max[:, 0]
    return scaled_r


def _random_affine_2d(translation_random_interval, rotation_random_interval, scaling_factors_random_interval):
    """
    Random 2D transformation matrix defined as Tfm = T * Rotation * Scaling

    Args:
        translation_random_interval: 2 x (min, max) array to specify (min, max) of x and y
        rotation_random_interval: 1 x (min, max) radian angle
        scaling_factors_random_interval: 2 x (min, max) or 1 x (min, max) scaling factors for x and y. if size
            ``1 x (min, max)``, the same random scaling will be applied

    Returns:
        a 3x3 transformation matrix
    """
    translation_offset = rand_n_2(translation_random_interval)
    rotation_angle = rand_n_2(rotation_random_interval)
    scaling_factors = rand_n_2(scaling_factors_random_interval)
    if len(scaling_factors) == 1:
        scaling_factors = np.asarray([float(scaling_factors), float(scaling_factors)])

    matrix_translation = affine.affine_transformation_translation(translation_offset)
    matrix_scaling = affine.affine_transformation_scale(scaling_factors)
    matrix_rotation = affine.affine_transformation_rotation2d(rotation_angle)

    return matrix_translation.mm(matrix_rotation.mm(matrix_scaling))


def _random_affine_3d(translation_random_interval, rotation_random_interval, scaling_factors_random_interval):
    """
    Random 3D transformation matrix defined as Tfm = T * Rotation * Scaling

    Args:
        translation_random_interval: 3 x (min, max) array to specify (min, max) of x and y
        rotation_random_interval: 1 x (min, max) radian angle
        scaling_factors_random_interval: 3 x (min, max) or 1 x (min, max) scaling factors for x and y. if size
            ``1 x (min, max)``, the same random scaling will be applied

    Returns:
        a 4x4 transformation matrix
    """
    translation_offset = rand_n_2(translation_random_interval)
    dim_data = len(translation_offset)

    rotation_angle = rand_n_2(rotation_random_interval)
    scaling_factors = rand_n_2(scaling_factors_random_interval)
    if len(scaling_factors) == 1:
        scaling_factors = np.asarray([float(scaling_factors)] * dim_data)

    matrix_translation = affine.affine_transformation_translation(translation_offset)
    matrix_scaling = affine.affine_transformation_scale(scaling_factors)
    matrix_rotation_x = affine.affine_transformation_rotation_3d_x(rotation_angle[0])
    matrix_rotation_y = affine.affine_transformation_rotation_3d_y(rotation_angle[1])
    matrix_rotation_z = affine.affine_transformation_rotation_3d_z(rotation_angle[2])

    return matrix_translation.mm(matrix_scaling.mm(matrix_rotation_z.mm(matrix_rotation_y.mm(matrix_rotation_x))))


class TransformAffine(transforms.TransformBatchWithCriteria):
    """
    Transform an image using a random affine (2D or 3D) transformation.

    Only 2D or 3D supported transformation.

    Notes:
        the scaling and rotational components of the transformation are performed
        relative to the image.
    """
    def __init__(
            self,
            translation_min_max: Sequence[Number],
            scaling_min_max: Sequence[Number],
            rotation_radian_min_max: Sequence[Number],
            isotropic: bool = True,
            criteria_fn: Optional[CriteriaFn] = None,
            padding_mode: str = 'zeros'):
        """

        Args:
            translation_min_max: Translation component. can be expressed as number, [min, max] for all axes, or
                [[min_x, max_x], [min_y, max_y], ...]
            scaling_min_max: Scaling component. can be expressed as number, [min, max] for all axes, or
                [[min_x, max_x], [min_y, max_y], ...]
            rotation_radian_min_max:
            isotropic: if ``True``, the random scaling will be the same for all axes
            criteria_fn:
            padding_mode: one of ``zero``, ``reflexion``, ``border``
        """
        if criteria_fn is None:
            criteria_fn = transforms.criteria_is_array_4_or_above

        self.padding_mode = padding_mode
        self.criteria_fn = criteria_fn
        self.rotation_radian_min_max = rotation_radian_min_max
        self.scaling_min_max = scaling_min_max
        self.translation_min_max = translation_min_max
        self.isotropic = isotropic

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=self._transform
        )

    def _transform(self, features_names, batch):
        data_shape = batch[features_names[0]].shape
        data_dim = len(data_shape) - 2  # remove `N` and `C` components
        assert data_dim == 2 or data_dim == 3, f'only 2D or 3D data handled. Got={data_dim}'
        for name in features_names[1:]:
            # make sure the data is correct: we must have the same dimensions (except `C`)
            # for all the images
            feature = batch[name]
            feature_shape = feature.shape[2:]
            assert feature_shape == data_shape[2:], f'joint features transformed must have the same dimension. ' \
                                                    f'Got={feature_shape}, expected={data_shape[2:]}'
            assert feature.shape[0] == data_shape[0]

        # normalize the transformation. We want a Dim * 2 matrix
        def normalize_input(i, dim, no_negative_range=False):
            i = np.asarray(i)
            if len(i.shape) == 0:  # single value
                if no_negative_range:
                    i = np.asarray([[i, i]] * dim)
                else:
                    i = np.asarray([[-i, i]] * dim)
            elif len(i.shape) == 1:  # min/max
                assert len(i) == 2
                i = np.repeat([[i[0], i[1]]], dim, axis=0)
            else:
                assert i.shape == (dim, 2), 'expected [(min_x, max_x), (min_y, max_y), ...]'
            return i

        translation_range = normalize_input(self.translation_min_max, data_dim)
        if data_dim == 2:
            rotation_range = normalize_input(self.rotation_radian_min_max, 1)
        elif data_dim == 3:
            rotation_range = normalize_input(self.rotation_radian_min_max, 3)
        else:
            raise NotImplementedError()

        if self.isotropic:
            scaling_range = normalize_input(self.scaling_min_max, 1, no_negative_range=True)
        else:
            scaling_range = normalize_input(self.scaling_min_max, data_dim, no_negative_range=True)

        tfms = []
        for n in range(data_shape[0]):
            if data_dim == 2:
                matrix_transform = _random_affine_2d(translation_range, rotation_range, scaling_range)
            elif data_dim == 3:
                matrix_transform = _random_affine_3d(translation_range, rotation_range, scaling_range)
            else:
                raise NotImplementedError()
            matrix_transform = affine.to_voxel_space_transform(matrix_transform, data_shape[1:])
            tfms.append(matrix_transform)

        tfms = torch.stack(tfms, dim=0)

        new_batch = collections.OrderedDict()
        for name, value in batch.items():
            if name in features_names:
                new_batch[name] = affine.affine_transform(value, tfms, padding_mode=self.padding_mode)
            else:
                new_batch[name] = value
        return new_batch
