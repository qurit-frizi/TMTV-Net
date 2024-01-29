from basic_typing import Batch
from typing import Sequence


class TransformCleanFeatures:
    """
    Remove features (e.g., too large or incompatible types)
    """
    def __init__(
            self,
            features_to_remove: Sequence[str]) -> None:
        self.features_to_remove = features_to_remove

    def __call__(self, batch: Batch) -> Batch:
        for f in self.features_to_remove:
            if f in batch:
                del batch[f]
        return batch