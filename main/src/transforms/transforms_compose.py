from typing import Sequence

from transforms.transforms import Transform
import collections

from basic_typing import Batch


class TransformCompose(Transform):
    """
    Sequentially apply a list of transformations
    """
    def __init__(self, transforms: Sequence[Transform]):
        """

        Args:
            transforms: a list of :class:`Transform`
        """
        assert isinstance(transforms, collections.Sequence), '`transforms` must be a sequence!'
        self.transforms = transforms

    def __call__(self, batch: Batch) -> Batch:
        for transform in self.transforms:
            batch = transform(batch)
        return batch
