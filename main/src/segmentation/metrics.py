import numpy as np
import nibabel as nib
import pathlib as plb
import cc3d

#
# Validation code
# extracted from https://github.com/lab-midas/autoPET/blob/master/val_script.py
#

def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def false_pos_pix(gt_array,pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)

    # LUDO: small modification. If too many components
    # abort! else it will take too long to compute
    if pred_conn_comp.max() > 1000:
        return None
    
    false_pos = 0
    for idx in range(1,pred_conn_comp.max()+1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask*gt_array).sum() == 0:
            false_pos = false_pos+comp_mask.sum()
    return false_pos



def false_neg_pix(gt_array,pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)

    # LUDO: small modification. If too many components
    # abort! else it will take too long to compute
    if gt_conn_comp.max() > 10000:
        return None
    
    false_neg = 0
    for idx in range(1,gt_conn_comp.max()+1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask*pred_array).sum() == 0:
            false_neg = false_neg+comp_mask.sum()
            
    return false_neg


def dice_score(mask1,mask2):
    # compute foreground Dice coefficient
    overlap = (mask1*mask2).sum()
    sum = mask1.sum()+mask2.sum()
    dice_score = 2*overlap/sum
    return dice_score


def calculate_metrics_autopet(found, truth, voxel_vol=1.0):
    found = found.numpy()
    truth = truth.numpy()

    false_neg_vol = false_neg_pix(truth, found)
    if false_neg_vol is not None:
        false_neg_vol *= voxel_vol
    
    false_pos_vol = false_pos_pix(truth, found)
    if false_pos_vol is not None:
        false_pos_vol *= voxel_vol

    if truth.max() == 0:
        # if no foreground, it doesn't make sense
        # to report them!
        sensitivity = None
    else:
        sensitivity = (found * truth).sum() / (truth.sum() + 0.1)

    dice_sc = dice_score(truth,found)
    dice_foreground = dice_sc if truth.max() > 0.5 else None
    return {
        'dice': dice_sc,
        'dice_foreground': dice_foreground,
        'false_neg_vol': false_neg_vol,
        'false_pos_vol': false_pos_vol,
        'mtv_found': float(found.sum()),
        'mtv_truth': float(truth.sum()),
        'sensitivity': sensitivity,
    }
