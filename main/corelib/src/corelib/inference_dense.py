
from collections import namedtuple
import logging
from torch import nn
from typing import Callable, Dict, Sequence, Union, List, Optional, Tuple, Any
from typing_extensions import Literal

from .typing import Batch, TensorNCX, ShapeX, TensorX
from .features_generic import feature_2d_slices
from .sub_tensor import sub_tensor

import numpy as np
import numpy as np
import scipy.ndimage
from utils import to_value

import collections
import numbers
import torch
from src.utilities import get_device, transfer_batch_to_device
from src.layers.crop_or_pad import crop_or_pad_fun


logger = logging.getLogger(__name__)


InferenceOutput = namedtuple('InferenceOutput', 'output_found output_truth output_input output_raw')



def get_volume_shape(batch):
    for name, value in batch.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            if len(value.shape) == 3:
                return value.shape

    raise ValueError('batch doesn\'t contain 3D volume!')


def inference_process_wholebody_2d(
        batch: Batch, 
        model: nn.Module, 
        num_slices: int, 
        output_truth_name: Optional[str], 
        main_input_name: Optional[str], 
        slice_step_size: int = 1) -> InferenceOutput:
    """
    Inference of a full 3D volume based on 2.5d features

    TODO:
        - batch the slices to improve GPU utilization
    """
    # during inference, we may not have the truth!
    output_truth = None
    if output_truth_name is not None:
        output_truth = batch.get(output_truth_name)
        assert output_truth is not None, f'missing truth={output_truth_name} in batch! Choices={batch.keys()}'     

    # this is the volume to be denoised
    output_input = None
    if main_input_name is not None:
        output_input = batch.get(main_input_name)

    half = num_slices // 2
    model.eval()
    device = get_device(model)
    with torch.no_grad():
        shape = get_volume_shape(batch)
        output_found = torch.zeros(shape, dtype=torch.float32)
        output_found_mask = torch.zeros(shape, dtype=torch.float32)
        for slice_n in range(half, shape[0] - half, slice_step_size):
            slices_input = feature_2d_slices(batch, index=np.asarray([slice_n, 0, 0]), num_slices=num_slices)
            slices_input = transfer_batch_to_device(slices_input, device)
            outputs = model(slices_input)
            slice_output = outputs['output_found_full'].output
            assert slice_output.shape == tuple([1, num_slices] + list(shape[1:]))
            output_found[slice_n - half:slice_n + half + 1] += to_value(slice_output.squeeze(0))
            output_found_mask[slice_n - half:slice_n + half + 1] += 1
            
        # averages all contributions. The overlap
        # smoothes the boundaries of the FoV
        output_found /= output_found_mask + 1e-6
        return InferenceOutput(output_found=output_found, output_truth=output_truth, output_input=output_input)


def zero_to_one_tfm(d):
    d = np.sqrt(d)
    max_value = d.max()
    min_value = d.min()
    return 1 - (d - min_value) / max_value


def central_weighting(
        block_shape: ShapeX,
        center_shape_fraction: float = 0.5,
        weight_from_distance_transform_fn=zero_to_one_tfm) -> np.ndarray:
    """
    Create a "weighting" where the center has maximum weight and weights further from the
    center have lower values.

    The purpose of block weighting is to avoid artifacts from reconstructing a large object from
    smaller blocks by weighting the boundaries in a way that interpolate between adjacent blocks.


    Args:
        block_shape: the shape of the blocks
        center_shape_fraction: fraction of the block to have its value set to maximum weight
        weight_from_distance_transform_fn: calculate the weighting from distance transform

    Returns:
        a weighting array
    """
    center_half_region = (np.asarray(block_shape) * center_shape_fraction).astype(int) // 2
    w = np.zeros(block_shape, dtype=np.float32)

    slices = [slice(d // 2 - c_h, d // 2 + c_h) for d, c_h in zip(block_shape, center_half_region)]
    w[tuple(slices)] = 1.0

    w = weight_from_distance_transform_fn(scipy.ndimage.morphology.distance_transform_cdt(w < 0.5))
    return w


Number = Union[float, int]


def resize(v: TensorX, multiple_of: Union[int, Sequence[int]], padding_value: Number) -> Tuple[TensorNCX, ShapeX]:
    """
    Resize the volume so that its size is a multiple of `mod`.

    Padding is added at the end of the volume.

    Returns:
        resized volume, padding
    """
    assert len(v.shape) == 3, 'Must be DHW format!'
    padding = np.asarray(v.shape) % multiple_of
    for n in range(len(padding)):
        if padding[n] != 0:
            if isinstance(multiple_of, int):
                padding[n] = multiple_of - padding[n]
            else:
                assert len(padding) == len(multiple_of)
                padding[n] = multiple_of[n] - padding[n]

    v_sub = crop_or_pad_fun(
        v.unsqueeze(0).unsqueeze(0),  # needs NC channels!
        v.shape + padding,
        padding_default_value=padding_value
    )
    return v_sub.squeeze(0).squeeze(0), padding


def uncrop(v, orig_shape):
    """
    Remove the left and right padding
    """
    assert len(orig_shape) == 3, 'Must be DHW format!'
    assert len(v.shape) == 5, 'Must be NCDHW'
    shape_difference = np.asarray(v.shape[2:]) - np.asarray(orig_shape)
    left_padding = shape_difference // 2
    right_padding = shape_difference - left_padding

    cropping_right = np.asarray(v.shape[2:]) - right_padding
    o = sub_tensor(
        v,
        min_indices=[0, 0] + list(left_padding),
        max_indices_exclusive=list(v.shape[:2]) + list(cropping_right)
    )
    return o


def extract_sub_volume(v: TensorNCX, shape: ShapeX, p: ShapeX, margin: ShapeX):
    p = np.asarray(p)
    margin = np.asarray(margin)
    shape = np.asarray(shape)
    v_shape = np.asarray(v.shape)[2:]
    dim = len(shape)

    margin_min = p - np.stack([np.asarray([0] * dim), p - margin]).max(axis=0)
    margin_max = np.stack([v_shape, p + shape + margin]).min(axis=0) - (p + shape)

    min_bb = p - margin_min
    max_bb = p + shape + margin_max
    sub_v = sub_tensor(v, [0, 0] + list(min_bb), list(v.shape[:2]) + list(max_bb))
    return sub_v, margin_min, margin_max


def create_umap_3d_tiled(
        model,
        batch,
        tile_shape,
        tile_step,
        tile_margin,
        get_output,
        feature_names,
        nb_outputs=2,
        tile_weight=None,
        internal_type=torch.float32,
        invalid_indices_value=1):
    """
    Create a UMap for a 3D model
    """
    assert len(feature_names) > 0
    # add the N, C dimensions
    batch = {name: value.unsqueeze(0).unsqueeze(0) if name in feature_names else value for name, value in batch.items()}

    shape = batch[feature_names[0]].shape
    dim = len(shape) - 2
    assert dim == 3, 'expecting NCDHW shaped data!'
    if isinstance(tile_shape, numbers.Number):
        tile_shape = [tile_shape] * dim
    if isinstance(tile_margin, numbers.Number):
        tile_margin = [tile_margin] * dim
    if isinstance(tile_step, numbers.Number):
        tile_step = [tile_step] * dim

    shape_binary = [shape[0], nb_outputs] + list(shape[2:])

    final = torch.zeros(shape_binary, dtype=internal_type)
    final_weight = torch.zeros(shape, dtype=internal_type)
    shape_zyx = shape[2:]

    z = 0
    while z + tile_shape[0] <= shape_zyx[0]:
        y = 0
        while y + tile_shape[1] <= shape_zyx[1]:
            x = 0
            while x + tile_shape[2] <= shape_zyx[2]:
                features = {}
                for name in feature_names:
                    tiled, margin_min, margin_max = extract_sub_volume(
                        batch[name],
                        tile_shape,
                        (z, y, x),
                        margin=tile_margin
                    )
                    features[name] = tiled

                # non image features
                other_feature_names = set(batch.keys()) - set(feature_names)
                for name in other_feature_names:
                    features[name] = batch[name]

                model_device = get_device(model)
                features = transfer_batch_to_device(features, model_device)
                with torch.no_grad():
                    umap_slice_output = model(features)
                    if get_output is not None:
                        umap_slice_output = get_output(umap_slice_output)
                    sub_umap = umap_slice_output.cpu().detach()
                    assert sub_umap.shape[1] == nb_outputs, f'expecting nb_outputs={nb_outputs}, got={sub_umap.shape[1]}'

                # crop the added margin
                sub_umap = sub_tensor(sub_umap, [0, 0] + list(margin_min), list(sub_umap.shape[:2]) + list(margin_min + tile_shape))

                # copy the results to `final` tensor
                sub_final = sub_tensor(final, (0, 0, z, y, x), (final.shape[0], final.shape[1], z + tile_shape[0], y + tile_shape[1], x + tile_shape[2]))
                assert sub_final.shape == sub_umap.shape
                sub_final_weight = sub_tensor(final_weight, (0, 0, z, y, x), (final.shape[0], 1, z + tile_shape[0], y + tile_shape[1], x + tile_shape[2]))
                if tile_weight is None:
                    sub_final_weight += 1.0
                    sub_final += sub_umap
                else:
                    assert tile_weight.shape == sub_final_weight.shape
                    sub_final_weight += tile_weight
                    sub_final += sub_umap * tile_weight

                x += tile_step[2]
            y += tile_step[1]
        z += tile_step[0]

    #if final_weight.min() == 0:
    #    print('WARNING!!! The full FoV was not performed!')

    invalid_indices = torch.where(final_weight <= 0)
    final_weight[invalid_indices] = 1.0
    final[invalid_indices] = invalid_indices_value # make sure invalid indices are marked with background class!
    final /= final_weight
    return final


def inference_process_wholebody_3d(
        batch: Batch,
        model: nn.Module,
        feature_names: List[str],
        get_output: Any = None,
        output_truth_name='spect_full',
        main_input_name='spect_low3',
        tile_shape=(64, 64, 64),
        tile_step=(64, 64, 64),
        tile_margin=(32, 32, 32),
        tile_weight: Literal['none', 'weighted_central'] = 'none',
        nb_outputs=1,
        multiple_of: Optional[Union[int, Sequence[int]]] = 32,
        padding_value: Union[Number, Dict[str, Number]] = 0,
        other_feature_names: Optional[List[str]] = None,
        postprocessing_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        invalid_indices_value: float = 1,
        no_output_ref_collection=False,
        internal_type=torch.float32,
        tiling_strategy: Literal['none', 'tiled_3d'] = 'none') -> InferenceOutput:
    """
    Process fully convolutional network outputs with inputs that hare larger than
    the model field of view by tiling the input and aggregating the results.

    Args:
        invalid_indices_value: this is the value to be set for the voxels
            that have not been processed. Default to `1` to show any
            missing regions. Should not happen when all the parameters
            are set appropriately 
        no_output_ref_collection: if True, no reference data is collected
            for the output. This is mostly to save RAM...
    """

    normalized_batch = {}
    padded = None
    original_shape = batch[feature_names[0]].shape

    if isinstance(tile_shape, numbers.Number):
        tile_shape = [tile_shape] * 3

    feature_name_present = list(set(batch.keys()).intersection(set(feature_names)))

    # in certain cases (e.g., UNet), the model expects sizes
    # to be a multiple of a size
    if multiple_of is not None:
        for name in feature_name_present:
            if isinstance(padding_value, collections.Mapping):
                p = padding_value.get(name)
                assert p is not None, f'missing padding value for volume={name}'
            else:
                p = padding_value

            normalized_batch[name], padded = resize(batch[name], multiple_of, p)
    else:
        normalized_batch = batch

    if other_feature_names is not None:
        for name in other_feature_names:
            normalized_batch[name] = batch[name]

    if tile_weight == 'none':
        tile_weight = None
    elif tile_weight == 'weighted_central':
        tile_weight = central_weighting(block_shape=tile_shape)
        # must have NCX format
        tile_weight = torch.from_numpy(tile_weight).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f'value not supported={tile_weight}')

    # now do the tiling according to the strategy
    if tiling_strategy == 'none':
        model_device = get_device(model)
        normalized_batch = transfer_batch_to_device(normalized_batch, model_device)
        with torch.no_grad():
            output = model(normalized_batch)

        if get_output is not None:
            output = get_output(output)
    elif tiling_strategy == 'tiled_3d':
        output = create_umap_3d_tiled(
            model=model,
            batch=normalized_batch,
            tile_shape=tile_shape,
            tile_step=tile_step,
            tile_margin=tile_margin,
            tile_weight=tile_weight,
            get_output=get_output,
            feature_names=feature_name_present,
            nb_outputs=nb_outputs,
            invalid_indices_value=invalid_indices_value,
            internal_type=internal_type
        )
    else:
        raise ValueError(f'value={tiling_strategy} is not handled!')

    # un-crop the output
    if padded is not None and max(padded) != 0:
        output = uncrop(output, original_shape)

    # prepare the output result
    assert len(output.shape) == 5
    output_truth = None
    if output_truth_name is not None:
        output_truth = batch.get(output_truth_name)
    
    output_input = None
    if main_input_name is not None:
        output_input = batch.get(main_input_name)

    if postprocessing_fn is not None:
        output_final = postprocessing_fn(output)
    else:
        output_final = output

    # remove the `N` dimension
    output = output.squeeze(0)
    output_final = output_final.squeeze(0)
    assert output_final.shape == original_shape, f'expected shape={original_shape}, got={output.shape}'
    if no_output_ref_collection:
        output_truth=None
        output_input=None
        output_found=None
        #output=None
    return InferenceOutput(output_found=output_final, output_truth=output_truth, output_input=output_input, output_raw=output)
