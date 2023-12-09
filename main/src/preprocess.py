import pickle
from typing import Dict
import lz4.frame
import os
import numpy as np
import torch


class PreprocessDataV1:
    """
    AutoPET data preprocessing

    Only use soft-tissue/bone CT windowing and SUV clipped to a high value
    """
    def __init__(
            self,
            ct_min_value: float = -150, 
            ct_max_value: float = 300,
            suv_min_value: float = 0, 
            suv_max_value: float = 30,
            ) -> None:
        self.ct_min_value = ct_min_value
        self.ct_max_value = ct_max_value

        self.suv_min_value = suv_min_value
        self.suv_max_value = suv_max_value

    def __call__(self, case_data: Dict) -> Dict:
        # normalize CT values to be [0..1]
        ct_range = (self.ct_max_value - self.ct_min_value)
        case_data['ct'] = (-self.ct_min_value + torch.clip(case_data['ct'], self.ct_min_value, self.ct_max_value)) / ct_range

        suv_range = (self.suv_max_value - self.suv_min_value)
        case_data['suv'] = (-self.suv_min_value + torch.clip(case_data['suv'], self.suv_min_value, self.suv_max_value)) / suv_range

        case_data['seg'] = case_data['seg'].type(torch.long)
        return case_data


class PreprocessDataV2_lung:
    """
    AutoPET data preprocessing

    Only use soft-tissue/bone CT windowing and SUV clipped to a high value
    and lung CT window
    """
    def __init__(
            self,
            ct_min_value: float = -150, 
            ct_max_value: float = 300,
            ct_lung_min_value: float = -1000, 
            ct_lung_max_value: float = -200,

            suv_min_value: float = 0, 
            suv_max_value: float = 30,
            ) -> None:
        self.ct_min_value = ct_min_value
        self.ct_max_value = ct_max_value

        self.ct_lung_min_value = ct_lung_min_value
        self.ct_lung_max_value = ct_lung_max_value

        self.suv_min_value = suv_min_value
        self.suv_max_value = suv_max_value

    def __call__(self, case_data: Dict) -> Dict:
        # maybe useful to detect CT lesions. BEWARE ordering
        ct_lung_range = (self.ct_lung_max_value - self.ct_lung_min_value)
        case_data['ct_lung'] = (-self.ct_lung_min_value + torch.clip(case_data['ct'], self.ct_lung_min_value, self.ct_lung_max_value)) / ct_lung_range

        # normalize CT values to be [0..1]
        ct_range = (self.ct_max_value - self.ct_min_value)
        case_data['ct'] = (-self.ct_min_value + torch.clip(case_data['ct'], self.ct_min_value, self.ct_max_value)) / ct_range

        suv_range = (self.suv_max_value - self.suv_min_value)
        case_data['suv'] = (-self.suv_min_value + torch.clip(case_data['suv'], self.suv_min_value, self.suv_max_value)) / suv_range

        case_data['seg'] = case_data['seg'].type(torch.long)
        return case_data


class PreprocessDataV3:
    """
    AutoPET data preprocessing

    Only use soft-tissue/bone CT windowing and SUV clipped to a high value
    and lung CT window.

    Additionally, normalize by mean/std.
    """
    def __init__(
            self,
            ct_min_value: float = -813, 
            ct_max_value: float = 624,
            ct_lung_min_value: float = -1000, 
            ct_lung_max_value: float = -200,
            ) -> None:

        self.ct_min_value = ct_min_value
        self.ct_max_value = ct_max_value

        self.ct_lung_min_value = ct_lung_min_value
        self.ct_lung_max_value = ct_lung_max_value

    def __call__(self, case_data: Dict) -> Dict:
        with torch.no_grad():
            # maybe useful to detect CT lesions. BEWARE ordering
            case_data['ct_lung'] = (-self.ct_lung_min_value + torch.clip(case_data['ct'], self.ct_lung_min_value, self.ct_lung_max_value))
            std, mean = torch.std_mean(case_data['ct_lung'])
            case_data['ct_lung'] = (case_data['ct_lung'] - mean) / (std + 1e-8)
            case_data['ct_lung_mean'] = float(mean)
            case_data['ct_lung_std'] = float(std)

            # normalize CT values
            case_data['ct'] = (-self.ct_min_value + torch.clip(case_data['ct'], self.ct_min_value, self.ct_max_value))
            std, mean = torch.std_mean(case_data['ct'])
            case_data['ct'] = (case_data['ct'] - mean) / (std + 1e-8)
            case_data['ct_mean'] = float(mean)
            case_data['ct_std'] = float(std)

            std, mean = torch.std_mean(case_data['suv'])
            case_data['suv'] = (case_data['suv'] - mean) / (std + 1e-8)
            case_data['suv_mean'] = float(mean)
            case_data['suv_std'] = float(std)

            # this is used to scale the SUV for display purposes
            # (e.g., during the wholebody inference MIP)
            case_data['suv_display_target'] = float((7.0 - mean) / (std + 1e-8))

            case_data['seg'] = case_data['seg'].type(torch.long)
            return case_data


class PreprocessDataV4_lung_soft_tissues_hot:
    """
    AutoPET data preprocessing

    Only use soft-tissue/bone CT windowing and SUV clipped to a high value
    and lung CT window & soft tissues & hot pet regions
    """
    def __init__(
            self,
            ct_min_value: float = -150, 
            ct_max_value: float = 300,
            ct_lung_min_value: float = -1000, 
            ct_lung_max_value: float = -200,
            # more like liver here
            #ct_soft_tissues_min_value: float = 0, 
            #ct_soft_tissues_max_value: float = 200,
            ct_soft_tissues_min_value: float = -100, 
            ct_soft_tissues_max_value: float = 100,
            suv_min_value: float = 0, 
            suv_max_value: float = 30,
            suv_lesion_min_value: float = 2, 
            suv_lesion_max_value: float = 10,
            post_processing_fn=None,
            internal_type=torch.float32,
            ) -> None:
        self.ct_min_value = ct_min_value
        self.ct_max_value = ct_max_value

        self.ct_lung_min_value = ct_lung_min_value
        self.ct_lung_max_value = ct_lung_max_value

        self.ct_soft_tissues_min_value = ct_soft_tissues_min_value
        self.ct_soft_tissues_max_value = ct_soft_tissues_max_value

        self.suv_min_value = suv_min_value
        self.suv_max_value = suv_max_value

        self.suv_lesion_min_value = suv_lesion_min_value
        self.suv_lesion_max_value = suv_lesion_max_value

        self.post_processing_fn = post_processing_fn
        self.internal_type = internal_type

    def __call__(self, case_data: Dict) -> Dict:
        # focus on the soft intensity range
        ct_soft_range = (self.ct_soft_tissues_max_value - self.ct_soft_tissues_min_value)
        case_data['ct_soft'] = (-self.ct_soft_tissues_min_value + torch.clip(case_data['ct'], self.ct_soft_tissues_min_value, self.ct_soft_tissues_max_value)) / ct_soft_range
        case_data['ct_soft'] = case_data['ct_soft'].type(self.internal_type)
        # maybe useful to detect CT lesions. BEWARE ordering
        ct_lung_range = (self.ct_lung_max_value - self.ct_lung_min_value)
        case_data['ct_lung'] = (-self.ct_lung_min_value + torch.clip(case_data['ct'], self.ct_lung_min_value, self.ct_lung_max_value)) / ct_lung_range
        case_data['ct_lung'] = case_data['ct_lung'].type(self.internal_type)

        # normalize CT values to be [0..1]
        ct_range = (self.ct_max_value - self.ct_min_value)
        case_data['ct'] = (-self.ct_min_value + torch.clip(case_data['ct'], self.ct_min_value, self.ct_max_value)) / ct_range
        case_data['ct'] = case_data['ct'].type(self.internal_type)

        suv_range = (self.suv_lesion_max_value - self.suv_lesion_min_value)
        case_data['suv_hot'] = (-self.suv_lesion_min_value + torch.clip(case_data['suv'], self.suv_lesion_min_value, self.suv_lesion_max_value)) / suv_range
        case_data['suv_hot'] = case_data['suv_hot'].type(self.internal_type)

        suv_range = (self.suv_max_value - self.suv_min_value)
        case_data['suv'] = (-self.suv_min_value + torch.clip(case_data['suv'], self.suv_min_value, self.suv_max_value)) / suv_range
        case_data['suv'] = case_data['suv'].type(self.internal_type)

        if 'seg' in case_data:
            case_data['seg'] = case_data['seg'].type(torch.long)

        if self.post_processing_fn is not None:
            case_data = self.post_processing_fn(case_data)

        #print('TODO REMOVE')
        #case_npy = {}
        #for name, value in case_data.items():
        #    if isinstance(value, torch.Tensor) and len(value.shape) > 1:
        #        case_npy[name] = value.numpy()
        #write_case('/mnt/datasets/ludovic/AutoPET/dataset/tmp.pkl.lz4', case_npy)
        return case_data


def read_case(path: str) -> Dict:
    with lz4.frame.open(path, mode='rb') as f:
        case_data = pickle.load(f)

    for name, value in case_data.items():
        if isinstance(value, np.ndarray) and len(value.shape) >= 3:
            case_data[name] = torch.from_numpy(value)

    case_data['case_name'] = os.path.basename(path)
    return case_data


def write_case(path: str, case_data) -> None:
    with lz4.frame.open(path, mode='wb') as f:
        pickle.dump(case_data, f)
