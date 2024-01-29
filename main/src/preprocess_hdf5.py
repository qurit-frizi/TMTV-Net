import os
import time
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import h5py
import torch
import copy
from basic_typing import Batch


def write_case_hdf5(path: str, case_data: Dict, chunk_shape: Tuple[int, int, int] = (64, 64, 64)) -> None:
    """
    Write a dataset as a HDF5 file.

    In particular, for the chunking & compression features.

    Args:
        path: where to write the HDF5 file. The extension will be appended
        case_data: the content of the case
        chunk_shape: Internally, chunk the data to allow partial loading.
            should not be too large (we will have to read large mostly unused data)
            and not too small (overhead to locate the chunks)
    """
    with h5py.File(path.replace('.pkl.lz4', '') + '.hdf5', 'w') as f:
        for name, value in case_data.items():
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            if name in ('ct', 'suv', 'seg'):
                f.create_dataset(name, data=value, chunks=chunk_shape, compression='lzf')
            else:
                f.create_dataset(name, data=value)


def read_case_hdf5(
        path: str, 
        image_names: Tuple[str, ...] = ('ct', 'suv', 'seg'), 
        case_image_sampler_fn: Optional[Callable[[Batch, h5py.File, Tuple[str, ...]], Dict]] = None) -> Dict:
    """
    Read a HDF5 dataset.

    Args:
        path: where to read the HDF5 file. The extension will be appended
        image_names: the names of the data to be partially loaded
        case_image_sampler_fn: if not None, load the partial data from the HDF5 object
    """
    time_start = time.perf_counter()
    case_data = {}
    if '.pkl.lz4' in path:
        path = path.replace('.pkl.lz4', '') + '.hdf5'
    with h5py.File(path, 'r') as f:
        for name in f.keys():
            if case_image_sampler_fn is not None and name in image_names:
                # all images are processed at the same time
                # by the sampler!
                continue
            else:
                case_data[name] = copy.deepcopy(f[name][()])
        
        # TODO: this should be in the data preprocessing not here...
        if 'target_spacing' in case_data:
            case_data['current_spacing'] = case_data['target_spacing']
            del case_data['target_spacing']
            case_data['current_origin'] = case_data['target_origin']
            del case_data['target_origin']
            case_data['current_shape'] = case_data['target_shape']
            del case_data['target_shape']
        else:
            case_data['current_spacing'] = case_data['original_spacing']
            case_data['current_origin'] = case_data['original_origin']
            case_data['current_shape'] = case_data['original_shape']
            

        if case_image_sampler_fn is not None:
            images = case_image_sampler_fn(case_data, f, image_names=image_names)
            case_data.update(images)

            #np.save('/mnt/datasets/ludovic/AutoPET/tmp2/ct.npy', f['ct'][()])
            #np.save('/mnt/datasets/ludovic/AutoPET/tmp2/ct_chunk.npy', images['ct'])
            #print('original_origin', case_data['original_origin'])
            #print('original_spacing', case_data['original_spacing'])
            #print('chunk_origin', case_data['original_origin'] + case_data['original_spacing'] * (images['chunking_offset_index_zyx'])[::-1])
    time_end = time.perf_counter()
    #print('done=', time_end - time_start)

    for name, value in case_data.items():
        if isinstance(value, np.ndarray) and len(value.shape) >= 3:
            case_data[name] = torch.from_numpy(value)

    case_data['case_name'] = os.path.basename(path)
    return case_data


def case_image_sampler_random(case_data: Batch, case_data_h5: h5py.File, block_shape: Tuple[int, int, int], image_names: Tuple[str, ...], margin: Tuple[int, int, int]=(0, 0, 0)) -> Dict:
    """
    At loading time, load only partially the dataset. Here, a random part of the images will be loaded.

    The idea is to load many (but small parts) cases to large variety in
    the data, most likely this will stabilize the training

    Args:
        case_data_h5: the HDF5 data
        block_shape: the size of the data to be loaded
        image_names: specify the names of the data to be partially loaded
        margin: special care for the end of each axis to account for boundary effect During inference.
            due to sub-windowing, we may have a few valid slices only! This needs to
            be approximated during the training too

    Returns:
        The case data
    """
    data_shape = case_data_h5[image_names[0]].shape
    assert len(data_shape) == 3
    assert len(block_shape) == 3
    min_corner = []
    for i in range(3):
        if data_shape[i] <= block_shape[i]:
            o = data_shape[i] // 2
        else:
            o = np.random.randint(-margin[i], data_shape[i] - block_shape[i] + margin[i])
        min_corner.append(o)
    min_corner = np.asarray(min_corner)
    max_corner = min_corner + np.asarray(block_shape)
    min_corner_safe = [max(0, min_corner[0]), max(0, min_corner[1]), max(0, min_corner[2])]  # deal with negative min indices
    
    sub_images = {}
    for image_name in image_names:
        assert case_data_h5[image_name].shape == data_shape
        sub_image = copy.deepcopy(case_data_h5[image_name][
            min_corner_safe[0]:max_corner[0],
            min_corner_safe[1]:max_corner[1],
            min_corner_safe[2]:max_corner[2]
        ])

        padding = np.asarray([
            [max(0, -min_corner[0]), max(0, max_corner[0] - data_shape[0])],
            [max(0, -min_corner[1]), max(0, max_corner[1] - data_shape[1])],
            [max(0, -min_corner[2]), max(0, max_corner[2] - data_shape[2])],
        ])

        if padding.max() > 0 or padding.min() < 0:
            # check if we need to pad a region of the data (i.e., out of FOV)
            min_value = sub_image.min()
            sub_image = np.pad(sub_image, padding, mode='constant', constant_values=min_value)
            
        assert sub_image.shape == block_shape
        sub_images[image_name] = sub_image

    # record the origin of the chunking, may be useful
    # for other transforms
    sub_images['chunking_offset_index_zyx'] = min_corner
    sub_images['current_origin'] = case_data['current_origin'] + case_data['current_spacing'] * sub_images['chunking_offset_index_zyx'][::-1]
    return sub_images