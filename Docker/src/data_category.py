from enum import Enum
import numpy as np


class DataCategory(Enum):
    Continuous = 'continuous'
    DiscreteOrdered = 'discrete_ordered'
    DiscreteUnordered = 'discrete_unordered'
    Other = 'other'

    @staticmethod
    def from_numpy_array(array):
        if len(array.shape) == 1:
            if np.issubdtype(array.dtype, np.integer):
                return DataCategory.DiscreteOrdered
            elif np.issubdtype(array.dtype, np.floating):
                return DataCategory.Continuous
            elif np.issubdtype(array.dtype, np.str):
                return DataCategory.DiscreteUnordered

        return DataCategory.Other
