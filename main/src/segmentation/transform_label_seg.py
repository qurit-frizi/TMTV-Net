import time
from typing import Dict
import cc3d
import numpy as np
import torch


def transform_label_segmentation(batch: Dict, volume_name: str = 'seg', connectivity: int = 18) -> Dict:
    """
    Write bounding boxes `bounding_boxes_min_max` of the foreground to the batch data for fast sampling of foreground
    """
    seg = batch.get(volume_name)
    assert seg is not None, f'missing volume={volume_name}'
    assert len(seg.shape) == 3, 'expected single volume with DHW format'
    
    time_start = time.perf_counter()
    if isinstance(seg, torch.Tensor):
        seg = seg.numpy()
    labels_out = cc3d.connected_components(seg, connectivity=connectivity)
    nb_labels = labels_out.max()
    bounding_boxes_min_max = []
    for i in range(1, nb_labels + 1):
        indices = np.where(labels_out==i)
        bb_min = []
        bb_max = []
        for d in indices:
            bb_min.append(min(d))
            bb_max.append(max(d))
        bounding_boxes_min_max.append((bb_min, bb_max))
    batch['bounding_boxes_min_max'] = [bounding_boxes_min_max]
    time_end = time.perf_counter()
    #print('label_seg_time=', time_end - time_start)
    return batch