import numpy as np
from typing import Tuple
from scipy.ndimage import distance_transform_edt as eucl_distance
from basic_typing import Batch, NumpyTensorNX, Length, NumpyTensorNCX, TorchTensorNCX, TorchTensorNX, TorchTensorN
from losses import one_hot
import torch


def seg2dist(
        seg: TorchTensorNX,
        nb_classes: int,
        resolution: Length = None,
        dtype=None,
        discard_background: bool=False) -> TorchTensorNCX:
    """
    Calculate the distance map of the one-hot encoded segmentation
    """
    assert len(seg.shape) == 4 or len(seg.shape) == 3, 'Must be `Batch, [Depth,] Height, Width` format'
    assert seg.shape[0] == 1

    with torch.no_grad():
        B = seg.shape[0]
        res = torch.zeros([B, nb_classes] + list(seg.shape[1:]), dtype=dtype)

        seg_one_hot = one_hot(seg, nb_classes)
        class_start = 1 if discard_background else 0
        for b in range(B): 
            for k in range(class_start, nb_classes):
                posmask = seg_one_hot[b, k].numpy().astype(np.bool)
                if posmask.any():
                    negmask = ~posmask
                    result = eucl_distance(negmask, sampling=resolution) * negmask - \
                            (eucl_distance(posmask, sampling=resolution) - 1) * posmask
                    res[b][k] = torch.from_numpy(result)
    return res


def seg2dist_v2(
        seg: TorchTensorNX,
        nb_classes: int,
        discard_background: bool = True,
        normalized: bool = False) -> TorchTensorNCX:
    """
    Calculate the distance map of the one-hot encoded segmentation
    based on https://github.com/JunMa11/SegWithDistMap/blob/master/code/train_LITS_BD.py
    """
    from skimage import segmentation as skimage_seg
    assert len(seg.shape) == 4 or len(seg.shape) == 3, 'Must be `Batch, [Depth,] Height, Width` format'
    assert seg.shape[0] == 1

    with torch.no_grad():
        img_gt = seg.numpy().astype(np.uint8)
        gt_sdf = np.zeros([1, nb_classes] + list(seg.shape[1:]))
        b = 0  # only have a single sample in the batch
        class_start = 1 if discard_background else 0
        for c in range(class_start, nb_classes):
            posmask = img_gt[b] == c
            if posmask.any():
                negmask = ~posmask
                posdis = eucl_distance(posmask)
                negdis = eucl_distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                if not normalized:
                    sdf = negdis - posdis
                else:
                    sdf = (negdis-np.min(negdis))/(np.max(negdis) - np.min(negdis) + 1e-7) - (posdis-np.min(posdis))/(np.max(posdis) - np.min(posdis) + 1e-7)
                sdf[boundary==1] = 0
                gt_sdf[b][c] = sdf

        #np.save('/mnt/datasets/ludovic/AutoPET/class0.npy', gt_sdf[0, 0])
        #np.save('/mnt/datasets/ludovic/AutoPET/class1.npy', gt_sdf[0, 1])
        return torch.from_numpy(gt_sdf)


def transform_surface_loss_preprocessing(batch: Batch, segmentation_name: str, nb_classes: int, discard_background: bool = True, normalized: bool = False) -> Batch:
    seg = batch[segmentation_name]
    assert isinstance(seg, torch.Tensor)
    assert len(seg.shape) == 3, 'must be DHW shape!'
    #dt = seg2dist(seg.unsqueeze(0), nb_classes=nb_classes, discard_background=discard_background)
    dt = seg2dist_v2(
        seg.unsqueeze(0), 
        nb_classes=nb_classes, 
        discard_background=discard_background, 
        normalized=normalized
    )
    
    # must remove the `N` component to be consistent with the rest of the pipeline
    batch['surface_loss_distance_transform'] = dt.squeeze(0)
    return batch

def loss_surface(outputs_pb: TorchTensorNCX, distance_transform: TorchTensorNCX) -> TorchTensorN:
    # mean instead of sum to avoid very large numbers
    assert outputs_pb.shape == distance_transform.shape
    n = len(outputs_pb)
    o = (outputs_pb * distance_transform).view([n, -1]).mean(dim=1)
    return o

def loss_surface_orig(outputs_pb: TorchTensorNCX, distance_transform: TorchTensorNCX) -> TorchTensorN:
    assert outputs_pb.shape == distance_transform.shape
    n = len(outputs_pb)
    o = (outputs_pb * distance_transform).view([n, -1]).sum(dim=1)
    return o


if __name__ == "__main__":
    nb_classes = 3
    shape = [1, 32, 20, 20]
    shapec = shape[:1] + [nb_classes] + shape[1:]
    seg = torch.zeros(shape, dtype=int)
    seg[0, 8:12, 10:15, 5:15] = 1
    output_pb = torch.softmax(torch.randn(size=shapec, dtype=torch.float32), dim=1)
    dt = seg2dist(seg, nb_classes)
    l = loss_surface(output_pb, dt)
    print('DONE')