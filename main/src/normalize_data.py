import collections
import os

import numpy as np
import skimage.transform




def len_batch(batch):
    """

    Args:
        batch: a data split or a `collections.Sequence`

    Returns:
        the number of elements within a data split
    """
    if isinstance(batch, (collections.Sequence, torch.Tensor)):
        return len(batch)

    assert isinstance(batch, collections.Mapping), 'Must be a dict-like structure! got={}'.format(type(batch))

    for name, values in batch.items():
        if isinstance(values, (list, tuple)):
            return len(values)
        if isinstance(values, torch.Tensor) and len(values.shape) != 0:
            return values.shape[0]
        if isinstance(values, np.ndarray) and len(values.shape) != 0:
            return values.shape[0]
    return 0



def safe_lookup(dictionary, *keys, default=None):
    """
    Recursively access nested dictionaries
    Args:
        dictionary: nested dictionary
        *keys: the keys to access within the nested dictionaries
        default: the default value if dictionary is ``None`` or it doesn't contain
            the keys
    Returns:
        None if we can't access to all the keys, else dictionary[key_0][key_1][...][key_n]
    """
    if dictionary is None:
        return default

    for key in keys:
        dictionary = dictionary.get(key)
        if dictionary is None:
            return default

    return dictionary



from data_category import DataCategory
from table_sqlite import get_data_types_and_clean_data
from typing import List


def resize_arrays(arrays: List[np.ndarray], shape=None) -> np.ndarray:
    if shape is None:
        shapes = [a.shape for a in arrays]
        shape = np.asarray(shapes).mean(axis=0).round().astype(int)

    resized_arrays = []
    for array in arrays:
        resized_array = skimage.transform.resize(
            array,
            shape,
            order=1,
            mode='constant',
            anti_aliasing=False,
            preserve_range=True)
        resized_arrays.append(resized_array)
    return np.asarray(resized_arrays)


def normalize_data(options, data, table_name):
    """
    Normalize, subsample and categorize the data

    The following operations are performed:
    - convert to numpy arrays
    - removed type column and return a specific `type` dictionary
    - normalize the path according to deployment: if static html, nothing to do
        if deployed, we MUST add the `app name` (the top folder name containing the SQL DB)
        as root
    - categorize the data as continuous, discrete unordered, discrete ordered, other
    - recover the DB type (string) from data values (e.g., float or int)
    """
    d = collections.OrderedDict()
    for name, values in data.items():
        d[name] = np.asarray(values)

    for name in list(d.keys()):
        dtype = d[name].dtype
        if dtype in options.data.types_to_discard:
            del d[name]

    types = get_data_types_and_clean_data(d)
    type_categories = {}

    # handle the column removal: maybe they are not useful or maybe the data
    # can't be parsed
    remove_columns = safe_lookup(options.config, table_name, 'data', 'remove_columns', default=[])
    if remove_columns is not None:
        for n in remove_columns:
            if n in data:
                del d[n]
            if n in types:
                del types[n]

    subsampling_factor = safe_lookup(
        options.config,
        table_name,
        'data',
        'subsampling_factor',
        default='1.0')
    subsampling_factor = float(subsampling_factor)
    assert subsampling_factor <= 1.0, 'sub-sampling factor must be <= 1.0'
    if subsampling_factor != 1.0:
        nb_samples = int(len_batch(d) * subsampling_factor)
        d = {
            name: values[:nb_samples] for name, values in d.items()
        }

    if options.embedded:
        appname = os.path.basename(os.path.dirname(options.db_root))
        for n, t in types.items():
            if 'BLOB_' in t:
                # it is a blob, meaning that it was saved on the local HD
                column = []
                for i in d[n]:
                    if i is not None:
                        column.append(f'{appname}/' + i)
                    else:
                        # the data was not present...
                        column.append(None)
                d[n] = column
                #d[n] = list(np.core.defchararray.add(np.asarray([f'{appname}/']), d[n]))

            if 'BLOB_IMAGE' in t:
                type_categories[n] = DataCategory.Other

    # load the numpy arrays
    for name, t in list(types.items()):
        if 'BLOB_NUMPY' in t:
            loaded_np = []
            root = os.path.join(os.path.dirname(options.db_root), '..')
            for relative_path in d[name]:
                try:
                    path = os.path.join(root, relative_path)
                    loaded_np.append(np.load(path))
                except:
                    # loading failed
                    loaded_np.append(None)

            # expand the array if satisfying the criteria
            try:
                array = np.asarray(loaded_np)
            except ValueError as e:
                array = None

            if array is not None and \
                    len(array.shape) == 2 and \
                    array.shape[1] <= options.data.unpack_numpy_arrays_with_less_than_x_columns:
                for n in range(array.shape[1]):
                    name_expanded = name + f'_{n}'
                    value_expanded = array[:, n]
                    d[name_expanded] = value_expanded
                    type_categories[name_expanded] = DataCategory.from_numpy_array(value_expanded)
                del d[name]
                del types[name]
            else:
                d[name] = loaded_np
                type_categories[name] = DataCategory.Other  # dimension too high

    # the DB converted all values to string and have lost the original type. Try to
    # revert back the type from the column values
    for name in list(d.keys()):
        t = d[name]

        if name not in type_categories:
            # do not take into account the `None`
            # so create a list of indices with non `None` row
            t = np.asarray(t)
            indices_not_none = np.asarray([i for i, v in enumerate(t) if v is not None])  # consider only full row

            try:
                t_np = np.asarray(t[indices_not_none], dtype=np.int)  # if e have a float, it will raise exception
                type_categories[name] = DataCategory.DiscreteOrdered
                full = np.ndarray(len(t), dtype=object)  # handle the ``None`` values
                full[indices_not_none] = t_np
                d[name] = list(full)
                continue  # success, go to next item
            except ValueError:
                pass

            try:
                t_np = np.asarray(t[indices_not_none], dtype=np.float32)
                type_categories[name] = DataCategory.Continuous

                full = np.ndarray(len(t), dtype=object)  # handle the ``None`` values
                full[indices_not_none] = t_np
                d[name] = list(full)
                continue  # success, go to next item
            except ValueError:
                type_categories[name] = DataCategory.Other

            try:
                _ = np.asarray(t[indices_not_none], dtype=np.str)  # test data conversion
                type_categories[name] = DataCategory.DiscreteUnordered
                continue  # success, go to next item
            except ValueError:
                type_categories[name] = DataCategory.Other

    return d, types, type_categories
