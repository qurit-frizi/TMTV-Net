import torch
import numpy as np


def renormalize_torch(data, desired_mean, desired_std, current_mean=None, current_std=None):
    """
    Transform the data so that it has desired mean and standard deviation element wise

    Args:
        data: a torch.Tensor
        desired_mean: the mean to transform data to
        desired_std: the std to transform data to
        current_mean: if the mean if known, do not recalculate it (e.g., training mean to be used in
            validation split)
        current_std: if the std if known, do not recalculate it (e.g., training std to be used in
            validation split)

    Returns:
        a torch.Tensor data with mean desired_mean and std desired_std
    """
    # see https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation
    if current_mean is None and current_std is None:
        current_std = torch.std(data)
        current_mean = torch.mean(data)

    if current_mean is None:
        assert current_std is None, 'std amd mean must be ether be None or defined!'
    if current_std is None:
        assert current_mean is None, 'std amd mean must be ether be None or defined!'

    assert current_std > 0
    assert desired_std > 0

    transformed_data = desired_mean + (data - current_mean) * desired_std / current_std
    return transformed_data


def renormalize_numpy(data, desired_mean, desired_std, current_mean=None, current_std=None):
    if current_mean is None and current_std is None:
        current_std, current_mean = np.std(data), np.mean(data)

    if current_mean is None:
        assert current_std is None, 'std amd mean must be ether be None or defined!'
    if current_std is None:
        assert current_mean is None, 'std amd mean must be ether be None or defined!'

    assert current_std > 0
    assert desired_std > 0

    transformed_data = desired_mean + (data - current_mean) * desired_std / current_std
    return transformed_data


def renormalize(data, desired_mean, desired_std, current_mean=None, current_std=None):
    """
    Transform the data so that it has desired mean and standard deviation element wise

    Args:
        data: a torch or numpy array
        desired_mean: the mean to transform data to
        desired_std: the std to transform data to
        current_mean: if the mean if known, do not recalculate it (e.g., training mean to be used in
            validation split)
        current_std: if the std if known, do not recalculate it (e.g., training std to be used in
            validation split)

    Returns:
        a data with mean desired_mean and std desired_std
    """
    if isinstance(data, torch.Tensor):
        return renormalize_torch(data, desired_mean, desired_std, current_mean=current_mean, current_std=current_std)
    elif isinstance(data, np.ndarray):
        return renormalize_numpy(data, desired_mean, desired_std, current_mean=current_mean, current_std=current_std)
    else:
        raise NotImplemented(f'not implemented for type={type(data)}')
