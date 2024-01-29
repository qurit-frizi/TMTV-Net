from functools import partial
import unittest
import tempfile
import numpy as np
from src.preprocess_hdf5 import case_image_sampler_random, read_case_hdf5, write_case_hdf5

from itertools import chain, combinations
from torch import nn
from typing import Callable, List, Sequence
from basic_typing import Batch, TensorNCX, ShapeX, TensorX
from corelib.inference_dense import InferenceOutput
import numpy as np
import copy
import torch


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def test_time_inference(
        batch: Batch,
        model: nn.Module,
        inference_fn: Callable[[Batch, nn.Module], InferenceOutput],
        transforms: List[Callable[[Batch], Batch]],
        transforms_inv: List[Callable[[Batch], Batch]],
        combination_fn: Callable[[Sequence], Sequence] = powerset
        ) -> InferenceOutput:
    """
    Apply test-time inference (i.e., augment the input and average the output)
    to get more reliable segmentation.

    This is done by applying a series for forward to modify the input (e.g., random affine transform)
    and inverse transform to apply on the inference output (e.g., inverserandom affine transform
    to move the output to the same geometry as input).

    Args:
        batch: the data to be used for the inference
        model: the model to be used
        inference_fn: how the inference is performed
        transforms: a series of transformation
        transforms_inv: a series of transformation that inverses the `transforms` spatial transform
        combination_fn: define how the transforms are applied

    Returns:
        InferenceOutput: the averaged inference
    """
    assert len(transforms_inv) == len(transforms)
    list_tfm_tfmi = combination_fn(zip(transforms, transforms_inv))

    with torch.no_grad():
        batch_orig = copy.copy(batch)
        inference_output = inference_fn(batch_orig, model)._asdict()
        nb_transforms = 1
        for transform_set in list_tfm_tfmi:
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
            new_inference_output = inference_fn(batch, model)._asdict()
            
            # apply the inverse transforms to get the
            # inference in the original space. This needs
            # to be done in reverse order
            for t_inv in forward_inv_tfm:
                new_inference_output = t_inv(new_inference_output)
            
            # aggregate the output
            for name, value in new_inference_output.items():
                if value is not None:
                    inference_output[name] += value

            nb_transforms += 1

        # averages the transforms
        for name, value in new_inference_output.items():
            if value is not None:
                inference_output[name] /= nb_transforms

        return InferenceOutput(**inference_output)

class TestHDF5(unittest.TestCase):
    def test_a_inference(self):
        def inference_fn(batch, model):
            # static inference to simplify the test
            return InferenceOutput(
                output_found=(batch['main_image'] > 42).astype(np.float32), 
                output_input=None, 
                output_raw=None,
                output_truth=None
            )

        batch = {
            'main_image': np.random.randint(0, 50, size=(20, 30, 35))
        }

        def flip_batch(batch, axis):
            new_batch = {}
            for name, value in batch.items():
                if isinstance(value, np.ndarray) and len(value.shape) == 3:
                   new_batch[name] = np.flip(value, axis)
                else:
                    new_batch[name] = value
            return new_batch

        transforms = [
            partial(flip_batch, axis=0),
            partial(flip_batch, axis=1),
            partial(flip_batch, axis=2),
        ]

        # the inverse of axis flip is the same axis flip
        transforms_inv = [
            partial(flip_batch, axis=0),
            partial(flip_batch, axis=1),
            partial(flip_batch, axis=2),
        ]

        r = test_time_inference(
            batch=batch, 
            model=None, 
            inference_fn=inference_fn,
            transforms=transforms, 
            transforms_inv=transforms_inv
        )

        assert (r.output_found == inference_fn(batch, None).output_found).all()


if __name__ == '__main__':
    unittest.main()