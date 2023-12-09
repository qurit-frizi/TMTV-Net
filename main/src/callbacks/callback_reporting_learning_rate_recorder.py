import collections

from .callback import Callback
from .callback_reporting_model_summary import export_table
from utilities import update_json_config
import logging


logger = logging.getLogger(__name__)


class CallbackReportingLearningRateRecorder(Callback):
    """
    Report the weight statistics of each layer
    """
    def __init__(self):
        self.table_name = 'learning_rate'
        self.lr_optimizers = collections.defaultdict(list)
        self.epochs = []
        self.initialized = False

    def first_time(self, options):
        self.initialized = True

        # set the default parameter of the graph
        config_path = options.workflow_options.sql_database_view_path

        update_json_config(config_path, {
            self.table_name: {
                'default': {
                    'X Axis': 'epoch',
                    'Y Axis': 'value',
                    'Group by': 'optimizer',
                    'discard_axis_y': 'epoch',
                    'discard_axis_x': 'value',
                    'discard_group_by': 'epoch',
                    'number_of_columns': 2,
                }
            }
        })

    def __call__(self, options, history, model, losses, outputs,
                 datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.debug('CallbackReportingLearningRateRecorder.__call__')

        optimizers = kwargs.get('optimizers')
        if optimizers is None:
            return

        if not self.initialized:
            self.first_time(options)

        epoch = len(history)
        table = collections.defaultdict(list)
        for optimizer_name, optimizer in optimizers.items():
            if len(optimizer.param_groups) != 1:
                logger.warning('Multiple param groups, don\'t know how to record this!')
                continue

            lr = optimizer.param_groups[0].get('lr')
            table['optimizer'].append(optimizer_name)
            table['epoch'].append(epoch)
            table['value'].append(lr)

        export_table(
            options,
            self.table_name,
            table,
            table_role='data_graph',
            clear_existing_data=False)

        logger.debug('CallbackReportingLearningRateRecorder.__call__ done!')
