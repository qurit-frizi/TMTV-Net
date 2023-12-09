import collections

from .callback import Callback
import logging

from .callback_reporting_model_summary import export_table
from utilities import update_json_config

logger = logging.getLogger(__name__)


class CallbackReportingRecordHistory(Callback):
    """
    This callback records the history to the reporting layer
    """
    def __init__(self, table_name='history'):
        self.table_name = table_name
        self.init_done = False

    def first_epoch(self, options):
        # set the default parameter of the graph
        config_path = options.workflow_options.sql_database_view_path
        update_json_config(config_path, {
            self.table_name: {
                'default': {
                    'X Axis': 'epoch',
                    'Y Axis': 'value',
                    'Group by': 'metric',
                    'discard_axis_y': 'epoch',
                    'discard_axis_x': 'value',
                    'discard_group_by': 'epoch',
                }
            }
        })
        self.init_done = True

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('started CallbackReportingRecordHistory.__call__')
        if not self.init_done:
            self.first_epoch(options)

        sql_database = options.workflow_options.sql_database
        dataset_values = []
        split_values = []
        output_values = []
        metric_values = []
        values = []
        for dataset_name, dataset in history[-1].items():
            for split_name, split in dataset.items():
                for output_name, output in split.items():
                    for metric_name, metric in output.items():
                        if metric is not None:
                            dataset_values.append(dataset_name)
                            split_values.append(split_name)
                            output_values.append(output_name)
                            metric_values.append(metric_name)
                            values.append(metric)

        batch = collections.OrderedDict([
            ('epoch', len(history)),
            ('dataset', dataset_values),
            ('split', split_values),
            ('output', output_values),
            ('metric', metric_values),
            ('value', values)
        ])

        export_table(
            options,
            self.table_name,
            batch,
            table_role='data_graph',
            clear_existing_data=False)

        sql_database.commit()
        logger.info('successfully completed CallbackReportingRecordHistory.__call__')
