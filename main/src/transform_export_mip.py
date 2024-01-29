import datetime
import os
from typing import Optional, Tuple
import numpy as np
from basic_typing import Batch
import matplotlib.pyplot as plt
from PIL import Image


def _transform(im, max_value, min_value, cmap, flip):
        # make sure we have EXACTLY the same
        # intensity range so that the images
        # are easily comparable
        im = cmap(im / max_value)
        if flip:
            im = np.flipud(im)
        im = np.clip(im, min_value, 1.0)
        im = np.uint8(im * 255)
        return im


def transform_export_maximum_intensity_projection(
        data: Batch, 
        export_path: str, 
        volume_names: Tuple[str, ...] = ('suv',),
        unique_name: bool = True,
        min_value: float = 0,
        max_value: float = 0.25,
        case_name: str = 'case_name',
        max_num_mips: Optional[int] = 50,
        cmap = plt.get_cmap('binary')) -> Batch:
    """
    Export MIP for debug purposes

    Args:
        data: the case data
        export_path: where to export the MIPs
        volume_name: what volumes to export
        unique_name: if True, each time a unique name is generated

    Returns:
        Batch: the input batch
    """
    nb_files =os.listdir(export_path)
    if len(nb_files) > max_num_mips:
        return data

    patient_name = data[case_name]
    if unique_name:
        patient_name = patient_name + '_' + datetime.datetime.now().strftime('%H%M%S') + '.png'
    for v_name in volume_names:
        v = data.get(v_name)
        if v is not None:
            assert len(v.shape) == 3, 'Must be DHW format!'
            path = os.path.join(export_path, patient_name)

            mip = v.numpy().max(axis=1)
            mip_rgb = _transform(mip, max_value, min_value, cmap, flip=True)
            
            i = Image.fromarray(mip_rgb)
            i.save(path)
    return data