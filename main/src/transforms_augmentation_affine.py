from typing import Tuple
import numpy as np
import torch
from transforms import affine_transform
from transforms.transforms_affine import _random_affine_3d
from basic_typing import Batch


def transform_augmentation_random_affine_transform(
        data: Batch, 
        p: float = 0.5, 
        rotation_max: float = 0.3, 
        scale_min_max: Tuple[float, float]=(0.90, 1.1)) -> Batch:
    """
    Randomly apply an affine transformation on all image like `len(shape)==3` arrays

    Args:
        data: a batch of data (expected a single case)
        p: probability of applying the transformation
        rotation_max: maximum angle (radian) of the affine transformation
        scale_min_max: a (min, max) tuple representing the scaling

    Returns:
        a batch with the affine transformation applied
    """
    if np.random.rand() > p:
        # no augmentation
        return data

    tfm = _random_affine_3d(
    np.asarray([[0, 0], [0, 0], [0, 0]]),
    np.asarray([[-rotation_max, rotation_max], [-rotation_max, rotation_max], [-rotation_max, rotation_max]]),
    np.asarray([[scale_min_max[0], scale_min_max[1]]]))

    def transform(d):
        # TODO: background by modality!
        # TODO: handle geometric space
        return affine_transform(d.unsqueeze(0).unsqueeze(0), tfm[:3].unsqueeze(0), padding_mode='zeros')[0, 0]

    new_batch = {}
    nb_transformed_volumes = 0
    for v_name, value in data.items():
        if isinstance(value, (np.ndarray, torch.Tensor)) and len(value.shape) == 3 and value.shape[0] > 1:
            v = data[v_name]
            orig_type = v.dtype
            v_tfmed = transform(v.float())
            nb_transformed_volumes += 1
            if orig_type in (torch.uint8, torch.int16, torch.int32, torch.int64):
                # must round as there will be holes in the segmentation
                # (due to the interpolation, we may end up voxel value if 0.99999)
                v_tfmed = v_tfmed.round().type(orig_type)
            new_batch[v_name] = v_tfmed

            #np.save(f'/mnt/datasets/ludovic/AutoPET/tmp2/{v_name}_before.npy', data[v_name].numpy())
            #np.save(f'/mnt/datasets/ludovic/AutoPET/tmp2/{v_name}_after.npy', new_batch[v_name].numpy())
        else:
            new_batch[v_name] = value

    #new_batch['affine_tfm'] = tfm.unsqueeze(0)
    assert nb_transformed_volumes > 0, 'no volume found in the batch! Something is not right!'
    return new_batch