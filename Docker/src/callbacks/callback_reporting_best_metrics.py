import time

from .callback import Callback
import logging
import collections

from .callback_reporting_model_summary import export_table
from utilities import update_json_config

logger = logging.getLogger(__name__)


def collect_best_metrics(current_metrics, history_step, metric_to_discard, epoch):
    """
    Collect the best metrics between existing best metrics and a new history step

    The best metric dictionary is encoded  ``dataset_name#split_name#output_name#metric_name``.

    Args:
        current_metrics: store the existing best metrics
        history_step: new time step to evaluate
        metric_to_discard: metric names to discard
        epoch: the ``history_step`` epoch

    Returns:
        dict representing the current best metrics
    """
    for dataset_name, dataset in history_step.items():
        for split_name, split in dataset.items():
            for output_name, output in split.items():
                for metric_name, metric_value in output.items():
                    if metric_name in metric_to_discard:
                        continue
                    name = f'{dataset_name}#{split_name}#{output_name}#{metric_name}'
                    best_value_step = current_metrics.get(name)
                    if best_value_step is not None:
                        best_value, best_step = best_value_step
                        if metric_value is not None and metric_value < best_value:
                            current_metrics[name] = (metric_value, epoch)
                    else:
                        current_metrics[name] = (metric_value, epoch)
    return current_metrics


class CallbackReportingBestMetrics(Callback):
    """
    Report the best value of the history and epoch for each metric

    This can be useful to accurately get the best value of a metric and in particular
    at which step it occurred.
    """
    def __init__(self, table_name='best_metrics', metric_to_discard=None, epoch_start=0):
        """

        Args:
            table_name: the table name to be used for storage in the SQL database
            metric_to_discard: None or a list of metrics to discard
            epoch_start: epoch before this value will not be used
        """
        self.table_name = table_name
        self.metric_to_discard = metric_to_discard
        self.epoch_start = epoch_start
        self.best_values = collections.OrderedDict()
        self.best_values_step = 0
        if self.metric_to_discard is None:
            self.metric_to_discard = []

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
        logger.info('CallbackReportingBestMetrics.__call__ started')

        if not self.init_done:
            self.first_epoch(options)

        # collect the min value of each metric by output/split/dataset
        epoch_start = min(self.epoch_start, self.best_values_step)
        for index, history_step in enumerate(history[epoch_start:]):
            collect_best_metrics(self.best_values, history_step, self.metric_to_discard, index + epoch_start)

        datasets = []
        splits = []
        outputs = []
        metrics = []
        values = []
        epochs = []
        for name, (value, epoch) in self.best_values.items():
            dataset, split, output, metric = name.split('#')
            datasets.append(dataset)
            splits.append(split)
            outputs.append(output)
            metrics.append(metric)
            values.append(value)
            epochs.append(epoch)

        sql_time_start = time.perf_counter()
        table = collections.OrderedDict([
            ('dataset', datasets),
            ('split', splits),
            ('output', outputs),
            ('metric', metrics),
            ('value', values),
            ('best epoch', epochs),
        ])
        export_table(options, self.table_name, table, table_role='data_tabular', clear_existing_data=True)
        sql_time_end = time.perf_counter()

        self.best_values_step = len(history) - 1
        logger.info(f'SQL writing time={sql_time_end - sql_time_start}')
        logger.info('successfully completed CallbackReportingBestMetrics.__call__!')

