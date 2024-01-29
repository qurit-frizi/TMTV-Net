from typing import List, Optional

from transforms.transforms import Transform
import collections

from basic_typing import Batch
import numpy as np


class TransformOneOf(Transform):
    """
    Randomly select a transform among a set of transforms and apply it
    """
    def __init__(self, transforms: List[Optional[Transform]]):
        """
        Args:
            transforms: a list of :class:`Transform`. If one of the transform
            is `None` and selected, no transform is applied
        """
        assert isinstance(transforms, collections.Sequence), '`transforms` must be a sequence!'
        self.transforms = transforms

    def __call__(self, batch: Batch) -> Batch:
        tfm = np.random.choice(self.transforms)
        if tfm is not None:
            return tfm(batch)
        return batch
