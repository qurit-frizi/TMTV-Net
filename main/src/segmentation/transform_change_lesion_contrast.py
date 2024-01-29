import copy
import time
from typing import Tuple

import torch
from basic_typing import Batch
import cc3d
import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance
from scipy.ndimage.morphology import binary_dilation

class TransformChangeLesionContrast:
    """
    Change the contrast of the lesions. The model struggles at times on the
    lesions with lower FDG uptake. Simulate those by using the segmentation
    mask.

    The background is weighted by using a distance transform of the segmentation
    mask. Only voxels with high enough intensities are modified
    """
    def __init__(
            self, 
            min_num_voxels: int = 25,
            min_lesion_contrast: float = 4,
            min_transformed_contrast: float = 1.5,
            lesion_intensity_factor_range: Tuple[float, float] = (0.25, 1.5),
            lesion_mask_name: str = 'seg',
            pt_name: str = 'suv',
            connectivity: int = 18,
            bounding_box_margin: int = 5,
            probability: float = 0.5,
            std_keep_fraction: float = 1.0,
            ) -> None:
        """
        Args:
            min_num_voxels: the minimum number of voxels for a lesion to be contrast
                modified. Small lesions should not be changed, we probably won't see
                anything on the downsampled CT, only medium-large lesions should be transformed
            min_lesion_contrast: after contrast is changed, only lesions with at least
                this contrast will be updated
            lesion_intensity_factor_range: a factor to be applied on the lesion intensity
            lesion_mask_name: the name of the mask
            pt_name: the name of the PET image to be transformed
            connectivity: connectivity to find the lesions using region growing
            bounding_box_margin: add extra context around the lesion so that updated
                lesion doesn't show large boundaries
            probability: probability of transforming the image
        """

        self.min_num_voxels = min_num_voxels
        self.min_lesion_contrast = min_lesion_contrast
        self.lesion_intensity_factor_range = lesion_intensity_factor_range
        self.lesion_mask_name = lesion_mask_name
        self.pt_name = pt_name
        self.connectivity = connectivity
        self.bounding_box_margin = bounding_box_margin
        self.min_transformed_contrast = min_transformed_contrast
        self.probability = probability
        self.std_keep_fraction = std_keep_fraction

    def __call__(self, batch: Batch) -> Batch:
        r = np.random.rand()
        if r > self.probability:
            # no augmentation!
            return batch

        # soft copy of the input batch except the image
        # to avoid side effects
        batch_orig = batch
        batch = copy.copy(batch_orig)
        batch[self.pt_name] = copy.deepcopy(batch[self.pt_name])

        #time_start = time.perf_counter()
        lesion_mask = batch[self.lesion_mask_name]
        assert len(lesion_mask.shape) == 3, 'Must be DHW format!'
        pt = batch[self.pt_name]
        assert pt.shape == lesion_mask.shape, 'PT/Mask should match exactly!'
        
        # find all the connected components
        labels = cc3d.connected_components(lesion_mask.numpy(), connectivity=self.connectivity)
        nb_labels = labels.max()
        #np.save('/mnt/datasets/ludovic/AutoPET/tmp/suv_all.npy', pt.numpy())
        for i in range(1, nb_labels + 1):            
            # find the extent of the segmentation
            current_labels = labels==i
            if current_labels.sum() < self.min_num_voxels:
                continue

            indices = np.where(current_labels)
            bb_min = []
            bb_max = []
            for d in indices:
                bb_min.append(min(d))
                bb_max.append(max(d))
            bb_min = np.asarray(bb_min) - self.bounding_box_margin
            # handle the negative indices...
            bb_min = np.maximum(bb_min, np.zeros_like(bb_min))
            bb_max = np.asarray(bb_max) + self.bounding_box_margin

            # extract the segmentation/PT
            pt_bb_torch = pt[
                bb_min[0]:bb_max[0],
                bb_min[1]:bb_max[1],
                bb_min[2]:bb_max[2]
            ]
            pt_bb = pt_bb_torch.numpy()

            current_label_bb = current_labels[
                bb_min[0]:bb_max[0],
                bb_min[1]:bb_max[1],
                bb_min[2]:bb_max[2]
            ]

            dilated_diff = np.logical_xor(binary_dilation(current_label_bb, iterations=4), current_label_bb)

            contrast_before = np.average(pt_bb[current_label_bb]) / (np.average(pt_bb[dilated_diff]) + 1e-2)
            if contrast_before < self.min_lesion_contrast:
                continue

            # calculate distance transform from the segmentation

            try:
                dt_bb_not = eucl_distance(np.logical_not(current_label_bb))
                dt_bb_not = (1 - dt_bb_not / (dt_bb_not.max() - dt_bb_not.min() + 1)) ** 9
            except Exception as e:
                print('DEBUG')
            #np.save('/mnt/datasets/ludovic/AutoPET/tmp/current_label_bb.npy', current_label_bb.numpy().astype(np.float32))
            #np.save('/mnt/datasets/ludovic/AutoPET/tmp/dt_bb_not.npy', dt_bb_not.astype(np.float32))
            #np.save('/mnt/datasets/ludovic/AutoPET/tmp/suv.npy', pt_bb.astype(np.float32))
            #np.save('/mnt/datasets/ludovic/AutoPET/tmp/dilated_diff.npy', dilated_diff.astype(np.float32))

            intensity_factor = np.random.uniform(
                self.lesion_intensity_factor_range[0], 
                self.lesion_intensity_factor_range[1]
            )

            # interpolate between PT original & intensity transformed based on distance
            # transform 
            mean_mask = pt_bb[current_label_bb].mean()
            std_mask = pt_bb[current_label_bb].std()


            high_voxels = pt_bb > (mean_mask - self.std_keep_fraction * std_mask)
            dt_bb_not_masked = dt_bb_not * high_voxels

            pt_bb_interpolated = dt_bb_not_masked * intensity_factor * pt_bb + (1-dt_bb_not_masked) * pt_bb
            contrast_after = np.average(pt_bb_interpolated[current_label_bb]) / (np.average(pt_bb_interpolated[dilated_diff]) + 1e-2)
            #np.save('/mnt/datasets/ludovic/AutoPET/tmp/suv_augmented.npy', pt_bb_interpolated)
            #np.save('/mnt/datasets/ludovic/AutoPET/tmp/dt_bb_not_masked.npy', dt_bb_not_masked.astype(np.float32))
            
            if contrast_after < self.min_transformed_contrast:
                # the contrast with the background is too small
                # or the lesion was not correctly segmented
                continue
            
            # finally, update the image
            pt_bb_torch[:] = torch.from_numpy(pt_bb_interpolated)
            #np.save('/mnt/datasets/ludovic/AutoPET/tmp/suv_augmented_all.npy', pt.numpy())
        
        #time_end = time.perf_counter()
        #np.save('/mnt/datasets/ludovic/AutoPET/tmp/suv_augmented_all_final.npy', pt.numpy())
        #print('DONE=', time_end - time_start)
        return batch