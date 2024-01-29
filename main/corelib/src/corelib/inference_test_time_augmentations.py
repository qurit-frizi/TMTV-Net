from itertools import chain, combinations
from torch import nn
from typing import Callable, List, Optional, Sequence
from basic_typing import Batch, TensorNCX, ShapeX, TensorX
from corelib.inference_dense import InferenceOutput
import numpy as np
import copy
import torch


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def postprocess_c_max(r: InferenceOutput):
    """
    Create the output found from the raw probabilities
    """
    r = r._asdict()
    assert len(r['output_raw'].shape) == 4
    assert r['output_raw'].shape[0] == 2
    r['output_found'] = r['output_raw'].argmax(dim=0)
    return InferenceOutput(**r)


def test_time_inference(
        batch: Batch,
        model: nn.Module,
        inference_fn: Callable[[Batch, nn.Module], InferenceOutput],
        transforms: List[Callable[[Batch], Batch]],
        transforms_inv: List[Callable[[Batch], Batch]],
        combination_fn: Callable[[Sequence], Sequence] = powerset,
        postprocessing_fn: Optional[Callable[[InferenceOutput], InferenceOutput]] = postprocess_c_max
        ) -> InferenceOutput:
    """
    Apply test-time inference (i.e., augment the input and average the output)
    to get more reliable segmentation.

    This is done by applying a series for forward to modify the input (e.g., random affine transform)
    and inverse transform to apply on the inference output (e.g., inverse random affine transform
    to move the output to the same geometry as input).

    Args:
        batch: the data to be used for the inference
        model: the model to be used
        inference_fn: how the inference is performed
        transforms: a series of transformation
        transforms_inv: a series of transformation that inverses the `transforms` spatial transform
        combination_fn: define how the transforms are applied
        postprocessing_fn: if not `None`,  the `InferenceOutput` will be post-processed by this function 

    Returns:
        InferenceOutput: the averaged inference
    """
    assert len(transforms_inv) == len(transforms)
    list_tfm_tfmi = combination_fn(zip(transforms, transforms_inv))

    with torch.no_grad():
        batch_orig = copy.copy(batch)
        inference_output = inference_fn(batch_orig, model)._asdict()
        nb_transforms = 1
        for transform_n, transform_set in enumerate(list_tfm_tfmi):
            if len(transform_set) == 0:
                # we did that one already
                continue

            forward_tfm = [t for t, _ in transform_set]
            forward_inv_tfm = [t_inv for _, t_inv in transform_set][::-1]

            # apply the transforms
            batch = batch_orig
            for t in forward_tfm:
                batch = t(batch)
            
            # run the inference
            new_inference_output = inference_fn(batch, model)
            new_inference_output = new_inference_output._asdict()
            
            #np.save('/mnt/datasets/ludovic/AutoPET/tmp/suv_orig.npy', batch_orig['suv'])
            #np.save(f'/mnt/datasets/ludovic/AutoPET/tmp/suv_tfm_{transform_n}.npy', batch['suv'])
            #np.save(f'/mnt/datasets/ludovic/AutoPET/tmp/found_tfm_{transform_n}.npy', new_inference_output['output_found'])
            #np.save(f'/mnt/datasets/ludovic/AutoPET/tmp/found_tfm_{transform_n}_prob1.npy', new_inference_output['output_raw'][1])

            # apply the inverse transforms to get the
            # inference in the original space. This needs
            # to be done in reverse order
            for t_inv in forward_inv_tfm:
                new_inference_output = t_inv(new_inference_output)

            #np.save(f'/mnt/datasets/ludovic/AutoPET/tmp/suv_tfm_inv_{transform_n}.npy', new_inference_output['output_input'])
            #np.save(f'/mnt/datasets/ludovic/AutoPET/tmp/found_tfm_inv_{transform_n}.npy', new_inference_output['output_found'])
            #np.save(f'/mnt/datasets/ludovic/AutoPET/tmp/found_tfm_inv_{transform_n}_prob1.npy', new_inference_output['output_raw'][1])
            
            # aggregate the output
            for name, value in new_inference_output.items():
                if value is not None:
                    inference_output[name] = inference_output[name] + value

            nb_transforms += 1

        # averages the transforms
        #np.save(f'/mnt/datasets/ludovic/AutoPET/tmp/found_all.npy', inference_output['output_found'])

        for name, value in new_inference_output.items():
            if value is not None:
                if value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                    #inference_output[name] = inference_output[name] >= 4 #(nb_transforms // 2)
                    inference_output[name] = inference_output[name] >= (nb_transforms // 2)
                else:
                    inference_output[name] = inference_output[name] / nb_transforms

        #np.save(f'/mnt/datasets/ludovic/AutoPET/tmp/found_all_binary.npy', inference_output['output_found'])
        #np.save(f'/mnt/datasets/ludovic/AutoPET/tmp/found_all_prob.npy', inference_output['output_raw'][1])

        r = InferenceOutput(**inference_output)
        if postprocessing_fn is not None:
            r = postprocessing_fn(r)
        return r