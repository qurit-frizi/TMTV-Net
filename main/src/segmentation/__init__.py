from .trainer import run_trainer, load_model
from preprocess import PreprocessDataV1, PreprocessDataV2_lung, PreprocessDataV3, PreprocessDataV4_lung_soft_tissues_hot

from datasets import load_case as _load_case
from functools import partial 
from preprocess import read_case as _read_case
from preprocess_hdf5 import read_case_hdf5, case_image_sampler_random
from .augmentations import TransformGeneric, train_suv_augmentations, train_ct_augmentations, train_suv_augmentations_v2

load_case = partial(_load_case, read_case=_read_case)

_read_case_hdf5_128 = partial(read_case_hdf5, case_image_sampler_fn=partial(case_image_sampler_random, block_shape=(128, 128, 128), image_names=('ct', 'suv', 'seg')))
load_case_hdf5_random128 = partial(_load_case, read_case=_read_case_hdf5_128)

_read_case_hdf5_128_m64 = partial(read_case_hdf5, case_image_sampler_fn=partial(case_image_sampler_random, block_shape=(128, 128, 128), image_names=('ct', 'suv', 'seg'), margin=(64, 64, 64)))
load_case_hdf5_random128_m64 = partial(_load_case, read_case=_read_case_hdf5_128_m64)


_read_case_hdf5_128_m3200 = partial(read_case_hdf5, case_image_sampler_fn=partial(case_image_sampler_random, block_shape=(128, 128, 128), image_names=('ct', 'suv', 'seg'), margin=(32, 0, 0)))
load_case_hdf5_random128_m3200 = partial(_load_case, read_case=_read_case_hdf5_128_m3200)


_read_case_hdf5_96 = partial(read_case_hdf5, case_image_sampler_fn=partial(case_image_sampler_random, block_shape=(96, 96, 96), image_names=('ct', 'suv', 'seg')))
load_case_hdf5_random96 = partial(_load_case, read_case=_read_case_hdf5_96)

_read_case_hdf5_64 = partial(read_case_hdf5, case_image_sampler_fn=partial(case_image_sampler_random, block_shape=(64, 64, 64), image_names=('ct', 'suv', 'seg')))
load_case_hdf5_random64 = partial(_load_case, read_case=_read_case_hdf5_64)

load_case_hdf5 = partial(_load_case, read_case=read_case_hdf5)
