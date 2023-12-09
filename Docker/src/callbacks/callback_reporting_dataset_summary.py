


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

from .callback import Callback
from .callback_reporting_model_summary import export_table
from utilities import update_json_config
import collections
import numpy as np
import logging


logger = logging.getLogger(__name__)


def _data_summary(datasets, max_nb_samples=None):
    """

    Args:
        datasets: the datasets
        max_nb_samples: if ``None``, go through all the batches of a sequence, else only consider ``max_nb_samples``
            to generate the statistics

    Returns:
        dict to be interpreted as a table with columns (dataset, split, shape, max, min, mean, std, nb_batches)
    """
    stats_to_average = ['mean', 'std']

    dataset_column = []
    split_column = []
    feature_column = []
    shape_column = []
    max_column = []
    min_column = []
    mean_column = []
    std_column = []
    nb_batches_column = []
    for dataset_name, dataset in datasets.items():
        logger.info(f'logging dataset={dataset_name}')
        for split_name, split in dataset.items():
            logger.info(f'logging split={split_name}')
            nb_samples = 0
            features_stats = collections.defaultdict(collections.OrderedDict)
            nb_batches = 0
            stats = None
            for batch_id, batch in enumerate(split):
                nb_samples += len_batch(batch)
                nb_batches += 1
                for feature_name, feature_value in batch.items():
                    feature_value = to_value(feature_value)
                    stats = features_stats[feature_name]
                    if isinstance(feature_value, np.ndarray):
                        if batch_id == 0:
                            stats['shape'] = feature_value.shape
                            stats['max'] = np.max(feature_value)
                            stats['min'] = np.min(feature_value)
                            stats['mean'] = np.mean(feature_value)
                            stats['std'] = np.std(feature_value)
                        else:
                            stats['max'] = max(np.max(feature_value), stats['max'])
                            stats['min'] = min(np.min(feature_value), stats['min'])
                            stats['mean'] += np.mean(feature_value)
                            stats['std'] += np.std(feature_value)

                if max_nb_samples is not None and nb_samples > max_nb_samples:
                    # we have examined enough samples to get reliable statistics
                    break

            if nb_samples <= 1 or stats is None:
                continue  # no data!

            for feature_name, feature_stats in features_stats.items():
                for stat_name in list(feature_stats.keys()):
                    if stat_name in stats_to_average:
                        feature_stats[stat_name] /= nb_batches

            for feature_name, features_stat in features_stats.items():
                if len(features_stat) == 0:
                    continue
                dataset_column.append(dataset_name)
                split_column.append(split_name)
                feature_column.append(feature_name)
                shape_column.append(str(features_stat['shape']))
                max_column.append(features_stat['max'])
                min_column.append(features_stat['min'])
                mean_column.append(features_stat['mean'])
                std_column.append(features_stat['std'])
                nb_batches_column.append(nb_batches)

            logger.info(f'logging split={split_name} done!')

        logger.info(f'logging dataset={dataset_name} done!')

    return collections.OrderedDict([
        ('dataset', dataset_column),
        ('split', split_column),
        ('feature', feature_column),
        ('shape', shape_column),
        ('max', max_column),
        ('min', min_column),
        ('mean', mean_column),
        ('std', std_column),
        ('nb_batches', nb_batches_column),
    ])


class CallbackReportingDatasetSummary(Callback):
    """
    Summarizes the data (min value, max value, number of batches, shapes) for each split of each dataset
    """
    def __init__(self, max_nb_samples=None, table_name='data_summary'):
        self.table_name = table_name
        self.max_nb_samples = max_nb_samples
        self.init_done = False

    def first_epoch(self, options):
        # set the default parameter of the graph
        config_path = options.workflow_options.sql_database_view_path
        update_json_config(config_path, {
            self.table_name: {
                'default': {
                    'with_column_title_rotation': '0',
                }
            }
        })
        self.init_done = True

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('CallbackReportingDatasetSummary started')
        self.first_epoch(options)
        data_stats = _data_summary(datasets, max_nb_samples=self.max_nb_samples)

        sql_database = options.workflow_options.sql_database
        export_table(
            options,
            self.table_name,
            data_stats,
            table_role='data_tabular',
            clear_existing_data=True)

        sql_database.commit()
        logger.info('CallbackReportingDatasetSummary successfully completed!')
