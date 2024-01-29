from argparse import Namespace
import collections
import functools
import os
import time
import json
from typing import Callable, Dict, List, Optional
import warnings
import numpy as np
from basic_typing import Batch
from sampler import SamplerRandom, SamplerSequential, SamplerSubsetRandomByListInterleaved
from sequence_array import SequenceArray

from src_transforms import transform_report_time
from sequence import Sequence


def load_case(
        batch: Batch, 
        configuration: Namespace, 
        read_case: Callable[[str], Dict], 
        transform: Optional[Callable[[Dict], Dict]]=None) -> Batch:
    """
    Load a preprocessed case

    Args:
        batch: a dict like containing a list of 'path' of size 1
        transform: a transform to be applied just after data loading

    Returns:
        a batch of data representing a single case
    """
    path = batch['path']
    assert len(path) == 1, 'should only read a single case at any time'
    data_root = configuration.datasets['data_root']
    data_version = configuration.datasets['datasets_version']
    full_path = os.path.join(data_root, path[0])
    
    time_start = time.perf_counter()
    case_data = read_case(full_path)
    time_loaded = time.perf_counter()
    
    if transform is not None:
        if not isinstance(transform, collections.Sequence):
            transform = [transform]
        
        for t in transform:
            if t is not None:
                case_data = t(case_data)

    time_processed = time.perf_counter()
    #print(f'Loaded={case_data["uid"]}, time_loading={time_loaded - time_start} time_processing={time_processed - time_loaded}')
    return case_data


def collate_cases(cases: List[Batch]) -> Batch:
    """
    Combine a list of batches as a single batch
    """
    batch = {}
    assert len(cases) > 0

    for k in cases[0].keys():
        batch[k] = [c[k] for c in cases]
    return batch


def flatten_lists(batch: Batch) -> Batch:
    """
    Undo `collate_cases`
    """
    batch_flat = {}
    for k, v in batch.items():
        if k == 'sample_uid':
            batch_flat[k] = v
        else:    
            assert isinstance(v, list) and len(v) == 1
            batch_flat[k] = v[0]  
    return batch_flat


def create_datasets_inmemory_map(
        datasets_json: Dict, 
        configuration: Namespace, 
        load_case_fn: Callable[[Dict, Namespace, List], Batch],
        preprocess_data_fn=None, 
        batch_clean=None, 
        transform_train=None, 
        transform_test=None,
        max_jobs_at_once_factor=1) -> Dict[str, Sequence]:
    """
    The data is fully loaded in memory, then augmented using a map
    """
    data_loading_workers = configuration.training['data_loading_workers']
    batch_size = configuration.training['batch_size']
    datasets = {}
    for dataset_name, dataset in datasets_json['datasets'].items():
        splits = {}
        for split_name, paths in dataset.items():
            # load the data first
            vs = []
            for path in paths:
                time_start = time.perf_counter()
                v = load_case_fn({'path': [path]}, configuration=configuration, transform=[preprocess_data_fn, batch_clean])
                time_end = time.perf_counter()
                print(f'loading={path}, time={time_end - time_start}')
                vs.append(v)
            vs = collate_cases(vs)
            
            # then apply the rest of the pipelines using a map
            if split_name == 'train':
                # training pipeline: do all the augmentations
                sampler = SamplerRandom(batch_size=1)
                sequence = SequenceArray(vs, sampler)
                sequence = sequence.map(functools.partial(transform_report_time, transforms=transform_train), nb_workers=data_loading_workers, max_jobs_at_once=max_jobs_at_once_factor * data_loading_workers)
                sequence = sequence.rebatch(batch_size)
            else:
                # evaluation pipeline: use a different transform as we
                # probably don't want to do augmentations in the eval pipeline
                sampler = SamplerSequential(batch_size=1)
                sequence = SequenceArray(vs, sampler)

                data_loading_workers_test = data_loading_workers
                if transform_test is None:
                    data_loading_workers_test = 2
                sequence = sequence.map(functools.partial(transform_report_time, transforms=transform_test), nb_workers=data_loading_workers_test, max_jobs_at_once=max_jobs_at_once_factor * data_loading_workers_test)
                sequence = sequence.rebatch(batch_size)

            splits[split_name] = sequence
        datasets[dataset_name] = splits
    return datasets


def create_datasets_reservoir_map(
        datasets_json: Dict, 
        configuration: Namespace, 
        load_case_fn: Optional[Callable[[Dict, Namespace, List], Batch]],
        preprocess_data_fn=None,
        preprocess_data_train_fn=None,
        preprocess_data_test_fn=None,
        batch_clean=None, 
        transform_train=None, 
        transform_test=None,
        max_reservoir_samples=100,
        min_reservoir_samples=100,
        nb_reservoir_workers=2,
        nb_map_workers=None,
        max_reservoir_jobs_at_once_factor=10,
        max_jobs_at_once_factor=2,
        load_case_train_fn: Optional[Callable[[Dict, Namespace, List], Batch]] = None,
        load_case_valid_fn: Optional[Callable[[Dict, Namespace, List], Batch]] = None
        ) -> Dict[str, Sequence]:
    """
    Use a reservoir for the training where the data will be accumulated, then
    features are extracted using a map.

    The test/valid are directly processed using a map and will be less efficient.
    So they should not contains long data augmentations (or any!).

    `load_case_fn` and `preprocess_data_fn` can be specialized for training/validation using
    the `_train` and `_valid` versions.
    """
    if preprocess_data_fn is not None:
        assert preprocess_data_train_fn is None
        assert preprocess_data_test_fn is None
        preprocess_data_train_fn = preprocess_data_fn
        preprocess_data_test_fn = preprocess_data_fn
    else:
        assert preprocess_data_fn is None

    if isinstance(preprocess_data_train_fn, collections.Sequence):
        transforms_train = preprocess_data_train_fn + [batch_clean]
    else:
        transforms_train = [preprocess_data_train_fn, batch_clean]

    if isinstance(preprocess_data_test_fn, collections.Sequence):
        transforms_test = preprocess_data_test_fn + [batch_clean]
    else:
        transforms_test = [preprocess_data_test_fn, batch_clean]

    if nb_map_workers is None:
        nb_map_workers = nb_reservoir_workers

    if load_case_fn is not None:
        assert load_case_train_fn is None
        assert load_case_valid_fn is None
        load_case_train_fn = load_case_fn
        load_case_valid_fn = load_case_fn
    else:
        assert load_case_train_fn is not None and load_case_valid_fn is not None

    _load_case_train_fn = functools.partial(load_case_train_fn, configuration=configuration, transform=transforms_train)
    _load_case_test_fn = functools.partial(load_case_valid_fn, configuration=configuration, transform=transforms_test)

    batch_size = configuration.training['batch_size']
    datasets = {}
    for dataset_name, dataset in datasets_json['datasets'].items():
        splits = {}
        for split_name, paths in dataset.items():            
            # then apply the rest of the pipelines using a map
            if split_name == 'train':
                # training pipeline: load cases in the background using reservoir
                # then use a map to do the augmentation
                sampler = SamplerRandom(batch_size=1)
                sequence = SequenceArray({'path': paths}, sampler)
                sequence = sequence.async_reservoir(
                    function_to_run=_load_case_train_fn, 
                    max_reservoir_samples=max_reservoir_samples, 
                    min_reservoir_samples=min_reservoir_samples, 
                    max_jobs_at_once=max_reservoir_jobs_at_once_factor,
                    nb_workers=nb_reservoir_workers
                ).collate()
                sequence = sequence.map(functools.partial(transform_report_time, transforms=transform_train), nb_workers=nb_map_workers, max_jobs_at_once=max_jobs_at_once_factor * nb_map_workers)
                sequence = sequence.rebatch(batch_size)
            else:
                # evaluation pipeline: use a different transform as we
                # probably don't want to do augmentations in the eval pipeline
                sampler = SamplerSequential(batch_size=1)
                sequence = SequenceArray({'path': paths}, sampler)

                nb_map_workers_test = nb_map_workers
                if transform_test is None:
                    nb_map_workers_test = 2
                sequence = sequence.map(functools.partial(transform_report_time, transforms=[_load_case_test_fn] + transform_test), nb_workers=nb_map_workers_test, max_jobs_at_once=max_jobs_at_once_factor * nb_map_workers_test)
                sequence = sequence.rebatch(batch_size)

            splits[split_name] = sequence
        datasets[dataset_name] = splits
    return datasets


def create_datasets_reservoir_map_weighted(
        datasets_json: Dict, 
        path_classes: str,
        configuration: Namespace, 
        load_case_fn: Callable[[Dict, Namespace, List], Batch],
        preprocess_data_fn=None,
        preprocess_data_train_fn=None,
        preprocess_data_test_fn=None,
        batch_clean=None, 
        transform_train=None, 
        transform_test=None,
        max_reservoir_samples=100,
        min_reservoir_samples=100,
        nb_reservoir_workers=2,
        nb_map_workers=None,
        max_reservoir_jobs_at_once_factor=10,
        max_jobs_at_once_factor=2,
        load_case_train_fn: Optional[Callable[[Dict, Namespace, List], Batch]] = None,
        load_case_valid_fn: Optional[Callable[[Dict, Namespace, List], Batch]] = None) -> Dict[str, Sequence]:
    """
    Use a reservoir for the training where the data will be accumulated, then
    features are extracted using a map.

    The test/valid are directly processed using a map and will be less efficient.
    """
    if preprocess_data_fn is not None:
        assert preprocess_data_train_fn is None
        assert preprocess_data_test_fn is None
        preprocess_data_train_fn = preprocess_data_fn
        preprocess_data_test_fn = preprocess_data_fn
    else:
        assert preprocess_data_fn is None

    if isinstance(preprocess_data_train_fn, collections.Sequence):
        transforms_train = preprocess_data_train_fn + [batch_clean]
    else:
        transforms_train = [preprocess_data_train_fn, batch_clean]

    if isinstance(preprocess_data_test_fn, collections.Sequence):
        transforms_test = preprocess_data_test_fn + [batch_clean]
    else:
        transforms_test = [preprocess_data_test_fn, batch_clean]

    if nb_map_workers is None:
        nb_map_workers = nb_reservoir_workers


    if load_case_fn is not None:
        assert load_case_train_fn is None
        assert load_case_valid_fn is None
        load_case_train_fn = load_case_fn
        load_case_valid_fn = load_case_fn
    else:
        assert load_case_train_fn is not None and load_case_valid_fn is not None

    _load_case_train_fn = functools.partial(load_case_train_fn, configuration=configuration, transform=transforms_train)
    _load_case_test_fn = functools.partial(load_case_valid_fn, configuration=configuration, transform=transforms_test)


    with open(path_classes, 'r') as f:
        classes_dict = json.load(f)

    batch_size = configuration.training['batch_size']
    datasets = {}
    for dataset_name, dataset in datasets_json['datasets'].items():
        splits = {}
        for split_name, paths in dataset.items():            
            # then apply the rest of the pipelines using a map
            if split_name == 'train':
                # training pipeline: load cases in the background using reservoir
                # then use a map to do the augmentation
                sample_index_by_class = [[] for _ in classes_dict.keys()]
                sample_class = []
                for p_n, p in enumerate(paths):
                    filename = os.path.basename(p)
                    found = False
                    for c_id, (c_n, c_names) in enumerate(classes_dict.items()):
                        if filename in c_names:
                            sample_index_by_class[c_id].append(p_n)
                            sample_class.append(c_id)
                            found = True
                            break
                        
                    if not found:
                        warnings.warn(f'case={filename} could not be found in the class splits!!')


                # have the samples of each class interleaved so that we have a good
                # balance of cases with foreground and background
                sampler = SamplerSubsetRandomByListInterleaved(indices=sample_index_by_class)
                sequence = SequenceArray({'path': paths, 'class_id': np.asarray(sample_class)}, sampler)
                sequence = sequence.async_reservoir(
                    function_to_run=_load_case_train_fn, 
                    max_reservoir_samples=max_reservoir_samples, 
                    min_reservoir_samples=min_reservoir_samples, 
                    max_jobs_at_once=max_reservoir_jobs_at_once_factor,
                    nb_workers=nb_reservoir_workers
                ).collate()
                sequence = sequence.map(functools.partial(transform_report_time, transforms=transform_train), nb_workers=nb_map_workers, max_jobs_at_once=max_jobs_at_once_factor * nb_map_workers)
                sequence = sequence.rebatch(batch_size)
            else:
                # evaluation pipeline: use a different transform as we
                # probably don't want to do augmentations in the eval pipeline
                sampler = SamplerSequential(batch_size=1)
                sequence = SequenceArray({'path': paths}, sampler)

                nb_map_workers_test = nb_map_workers
                if transform_test is None:
                    nb_map_workers_test = 2
                sequence = sequence.map(functools.partial(transform_report_time, transforms=[_load_case_test_fn] + transform_test), nb_workers=nb_map_workers_test, max_jobs_at_once=max_jobs_at_once_factor * nb_map_workers_test)
                sequence = sequence.rebatch(batch_size)

            splits[split_name] = sequence
        datasets[dataset_name] = splits
    return datasets