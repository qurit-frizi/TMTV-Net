import torch
from export import export_sample
from collate import collate_list_of_dicts


def to_value(v):
    """
    Convert where appropriate from tensors to numpy arrays

    Args:
        v: an object. If ``torch.Tensor``, the tensor will be converted to a numpy
            array. Else returns the original ``v``

    Returns:
        ``torch.Tensor`` as numpy arrays. Any other type will be left unchanged
    """
    if isinstance(v, torch.Tensor):
        return v.cpu().data.numpy()
    return v

from table_sqlite import table_truncate, TableStream
from utilities import create_or_recreate_folder, update_json_config
from .callback import Callback

def get_batch_n(split, nb_samples, indices, transforms, use_advanced_indexing):
    """
    Collect the split indices given and apply a series of transformations
    Args:
        nb_samples: the total number of samples of split
        split: a mapping of `np.ndarray` or `torch.Tensor`
        indices: a list of indices as numpy array
        transforms: a transformation or list of transformations or None
        use_advanced_indexing: if True, use the advanced indexing mechanism else
            use a simple list (original data is referenced)
            advanced indexing is typically faster for small objects, however for large objects (e.g., 3D data)
            the advanced indexing makes a copy of the data making it very slow.
    Returns:
        a split with the indices provided
    """
    data = {}
    for split_name, split_data in split.items():
        if isinstance(split_data, (torch.Tensor, np.ndarray)) and len(split_data) == nb_samples:
            # here we prefer [split_data[i] for i in indices] over split_data[indices]
            # this is because split_data[indices] will make a deep copy of the data which may be time consuming
            # for large data
            if use_advanced_indexing:
                split_data = split_data[indices]
            else:
                split_data = [[split_data[i]] for i in indices]
        if isinstance(split_data, list) and len(split_data) == nb_samples:
            split_data = [split_data[i] for i in indices]

        data[split_name] = split_data

    if transforms is None:
        # do nothing: there is no transform
        pass
    elif isinstance(transforms, collections.Sequence):
        # we have a list of transforms, apply each one of them
        for transform in transforms:
            data = transform(data)
    else:
        # anything else should be a functor
        data = transforms(data)

    return data



import sequence
import logging
import os
import collections
import numpy as np


logger = logging.getLogger(__name__)


class CallbackReportingAugmentations(Callback):
    """
    Export sample augmentations.

    Augmentation are detected using the ``uid_name`` of a sample. Samples with the same uid over several epochs
    """
    def __init__(self, nb_samples=10, nb_augmentation=5, table_name='augmentations', split_name=None, uid_name='sample_uid'):
        """

        Args:
            nb_samples: the number of samples to export
            nb_augmentation: the number of augmentations per sample to export
            table_name: the SQL table where to export the augmentations
            split_name: the name of the split, typically the training split. If None, we will
                use the default training name from the options
            uid_name: this is the UID name that will be used to detect the samples
                (augmentations will be aggregated by `uid_name`)
        """
        self.nb_samples = nb_samples
        self.nb_augmentation = nb_augmentation
        self.table_name = table_name
        self.split_name = split_name
        self.uid_name = uid_name
        self.init_done = False

    def first_epoch(self, options):
        # set the default parameter of the graph
        config_path = options.workflow_options.sql_database_view_path
        update_json_config(config_path, {
            self.table_name: {
                'default': {
                    'with_column_title_rotation': '0',
                    'Scatter Y Axis': self.uid_name,
                }
            }
        })
        self.init_done = True

    def create_or_recreate_table(self, options):
        # destroy previous table
        sql_database = options.workflow_options.sql_database
        cursor = sql_database.cursor()
        table_truncate(cursor, self.table_name)

        # remove the binary/image store of the previous table
        root = os.path.dirname(options.workflow_options.sql_database_path)
        create_or_recreate_folder(os.path.join(root, 'static', self.table_name))

        # create a fresh table
        sql_table = TableStream(
            cursor=sql_database.cursor(),
            table_name=self.table_name,
            table_role='data_samples')
        return sql_table, root

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('started CallbackReportingAugmentations.__call__')
        if self.split_name is None:
            self.split_name = options.workflow_options.train_split

        if not self.init_done:
            self.first_epoch(options)

        sql_table, root = self.create_or_recreate_table(options)

        for dataset_name, dataset in datasets.items():
            logger.info('collecting samples for dataset={}'.format(dataset_name))

            split = dataset.get(self.split_name)
            if split is None:
                continue

            if not isinstance(split, sequence.Sequence):
                logger.warning('split is not a `trw.train.Sequence`, can\'t subsample '
                               'the sequence. This split is discarded!')
                continue

            # run the augmentations on the subsampled split
            # then collect the samples by UID, these represent our augmentations
            samples_by_uid = collections.defaultdict(list)
            split_subsampled = split.subsample(self.nb_samples)
            for augmentation_id in range(self.nb_augmentation):
                nb_samples_recorded = 0
                for batch in split_subsampled:
                    batch = {name: to_value(values) for name, values in batch.items()}
                    uids = to_value(batch.get(self.uid_name))
                    if uids is None:
                        logger.error('no UID found in the dataset! Can\'t link the augmentations')
                        return
                    assert uids is not None, 'we must have a unique UID for each sample!'
                    nb_samples = len(uids)
                    for index, uid in enumerate(uids):
                        sample = get_batch_n(
                            batch,
                            nb_samples,
                            [index],
                            None,
                            use_advanced_indexing=True)
                        samples_by_uid[uid].append(sample)
                    nb_samples_recorded += len(uids)
                    if nb_samples_recorded >= self.nb_samples:
                        # we might have used a resampled sampler, so double check the number
                        # of samples too and not rely solely on the en of the sequence iterator
                        break

            # export the samples
            for uid, samples in samples_by_uid.items():
                nb_samples = len(samples)
                if len(samples) > 1:
                    samples = collate_list_of_dicts(samples, device=torch.device('cpu'))
                    samples['dataset'] = np.asarray([dataset_name] * nb_samples)
                    samples['split'] = np.asarray([self.split_name] * nb_samples)
                    for n in range(nb_samples):
                        name = f'{uid}_{n}_{len(history)}'
                        export_sample(
                            root,
                            sql_table,
                            base_name=name,
                            batch=samples,
                            sample_ids=[n],
                            name_expansions=[],  # we already expanded in the basename!
                        )

        logger.info('successfully completed CallbackReportingAugmentations.__call__')
