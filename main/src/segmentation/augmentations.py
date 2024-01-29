from functools import partial
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.abstract_transforms import Compose
import torch
from transforms import TransformBatchWithCriteria, criteria_feature_name


def train_suv_augmentations():
    transforms = [
        #GaussianNoiseTransform(p_per_sample=0.1),
        GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=False, p_per_sample=0.1, p_per_channel=1.0),
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.1),
        ContrastAugmentationTransform(p_per_sample=0.1),
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, order_downsample=0, order_upsample=3, p_per_sample=0.1, ignore_axes=None),
        GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1),  # inverted gamma
        GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.1),
    ]
    tfm = Compose(transforms)
    return tfm


def train_suv_augmentations_v2():
    transforms = [
        GaussianBlurTransform((0.3, 0.7), different_sigma_per_channel=False, p_per_sample=0.1, p_per_channel=1.0, different_sigma_per_axis=True),
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.1),
        ContrastAugmentationTransform(p_per_sample=0.1),
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, order_downsample=0, order_upsample=3, p_per_sample=0.1, ignore_axes=None),
        GammaTransform((0.85, 1.25), False, True, retain_stats=True, p_per_sample=0.1),
    ]
    tfm = Compose(transforms)
    return tfm


def train_ct_augmentations():
    transforms = [
        #GaussianNoiseTransform(p_per_sample=0.1),
        #GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=False, p_per_sample=0.1, p_per_channel=1.0),
        #BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.1),
        ContrastAugmentationTransform(p_per_sample=0.1),
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, order_downsample=0, order_upsample=3, p_per_sample=0.1, ignore_axes=None),
        GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1),  # inverted gamma
        GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.1),
    ]
    tfm = Compose(transforms)
    return tfm


def transform_generic(feature_names, batch, transform_fn):
    assert len(feature_names) == 1
    feature_name = feature_names[0]
    v = batch[feature_name]

    #
    # DEBUG ONLY
    #
    #transform_fn = GaussianNoiseTransform(p_per_sample=0.999, noise_variance=(0, 1.0))  # BAD
    #transform_fn = GaussianBlurTransform((0.3, 0.7), different_sigma_per_channel=False, p_per_sample=1.0, p_per_channel=0.5, different_sigma_per_axis=True)
    #transform_fn = BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=1.0)
    #transform_fn = ContrastAugmentationTransform(p_per_sample=1.0)
    #transform_fn = SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=1.0, order_downsample=0, order_upsample=3, p_per_sample=1.0, ignore_axes=None)
    #transform_fn = GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=1.0)  # not useful?
    #transform_fn = GammaTransform((0.85, 1.25), False, True, retain_stats=True, p_per_sample=1.0)
    
    """
    volumes = [v.numpy()]
    for i in range(5):
        print('Processing:', i)
        # data is in-place, so need to make copy
        # must have NC components!!!
        v_tfmed = transform_fn(data=v.clone().unsqueeze(0).unsqueeze(0).numpy())['data'].squeeze(0).squeeze(0)
        volumes.append(v_tfmed)
    from corelib import compare_volumes_mips
    
    suv_display_target = batch.get('suv_display_target')
    if suv_display_target is None:
        suv_display_target = 7.5

    fig = compare_volumes_mips([volumes], case_names=['case0'], category_names=['n'] * len(volumes), max_value=suv_display_target * 1.5, flip=True, with_xy=False, with_yz=False)
    fig.tight_layout()
    fig.savefig('/mnt/datasets/ludovic/AutoPET/logging/aug.png')
    """
    
    #
    # DEBUG END
    #
    
    # unsqueeze to consider the whole case as one data point (not slices)
    # must have the NC components!!!
    v_tfmed = transform_fn(data=v.clone().unsqueeze(0).unsqueeze(0).numpy())['data'].squeeze(0).squeeze(0)
    batch[feature_name] = torch.from_numpy(v_tfmed)
    return batch


class TransformGeneric(TransformBatchWithCriteria):
    def __init__(self, transform_fn, feature_name):
        criteria_fn = partial(criteria_feature_name, feature_names=[feature_name])

        super().__init__(
            criteria_fn=criteria_fn,
            transform_fn=partial(transform_generic, transform_fn=transform_fn))