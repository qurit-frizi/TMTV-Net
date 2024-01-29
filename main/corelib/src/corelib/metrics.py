import numpy as np
from typing import Tuple
import math


def psnr(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    mse = np.mean((gt - pred) ** 2)
    if(mse == 0):
        # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100

    max_value = gt.max()
    p = 0
    if max_value > 0:
        p = 20 * math.log10(gt.max() / math.sqrt(mse))
    return p, mse 
