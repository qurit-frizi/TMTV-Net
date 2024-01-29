from typing import Sequence, List, Union, Callable

import numpy as np
import torch
from basic_typing import Tensor, Numeric, TensorNCX, ShapeCX
from typing_extensions import Protocol


def cutout_value_fn_constant(image: Tensor, value: Numeric) -> None:
    """
    Replace all image as a constant value
    """
    image[:] = value


def cutout_random_ui8_torch(image: torch.Tensor, min_value: int = 0, max_value: int = 255) -> None:
    """
    Replace the image content as a constant value
    """
    assert isinstance(image, torch.Tensor), f'must be a tensor. Got={type(image)}'
    size = [image.shape[0]] + [1] * (len(image.shape) - 1)
    color = torch.randint(
        min_value,
        max_value,
        dtype=image.dtype,
        device=image.device,
        size=size)
    image[:] = color


def cutout_random_size(min_size: Sequence[int], max_size: Sequence[int]) -> List[int]:
    """
    Return a random size within the specified bounds.

    Args:
        min_size: a sequence representing the min size to be generated
        max_size: a sequence representing the max size (inclusive) to be generated

    Returns:
        a tuple representing the size
    """
    assert len(min_size) == len(max_size)
    return [np.random.randint(low=min_value, high=max_value + 1) for min_value, max_value in zip(min_size, max_size)]


class CutOutType(Protocol):
    def __call__(self, image: TensorNCX) -> None:
        ...


def cutout(image: TensorNCX, cutout_size: Union[ShapeCX, Callable[[], ShapeCX]], cutout_value_fn: CutOutType) -> None:
    """
    Remove a part of the image randomly

    Args:
        image: a :class:`numpy.ndarray` or :class:`torch.Tensor` n-dimensional array. Samples are stored on axis 0
        cutout_size: the cutout_size of the regions to be occluded or a callable function taking no argument
            and returning a tuple representing the shape of the region to be occluded (without the ``N`` component)
        cutout_value_fn: the function value used for occlusion. Must take as argument `image` and modify
            directly the image

    Returns:
        None
    """
    if callable(cutout_size):
        cutout_size = cutout_size()

    nb_dims = len(cutout_size)
    assert len(image.shape) == nb_dims
    offsets = [np.random.randint(0, image.shape[n] - cutout_size[n] + 1) for n in range(len(cutout_size))]

    if nb_dims == 1:
        cutout_value_fn(
            image[offsets[0]:offsets[0] + cutout_size[0]])

    elif nb_dims == 2:
        cutout_value_fn(
            image[
            offsets[0]:offsets[0] + cutout_size[0],
            offsets[1]:offsets[1] + cutout_size[1]])

    elif nb_dims == 3:
        cutout_value_fn(
            image[
            offsets[0]:offsets[0] + cutout_size[0],
            offsets[1]:offsets[1] + cutout_size[1],
            offsets[2]:offsets[2] + cutout_size[2]])

    elif nb_dims == 4:
        cutout_value_fn(
            image[
            offsets[0]:offsets[0] + cutout_size[0],
            offsets[1]:offsets[1] + cutout_size[1],
            offsets[2]:offsets[2] + cutout_size[2],
            offsets[3]:offsets[3] + cutout_size[3]])
    else:
        raise NotImplementedError()
