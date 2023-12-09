import os
import logging
import collections

from .callback import Callback
import utilities
import analysis_plots


logger = logging.getLogger(__name__)


class CallbackLearningRateRecorder(Callback):
    """
    Record the learning rate of the optimizers.

    This is useful as a debugging tool.
    """
    def __init__(self, dirname='lr_recorder'):
        self.dirname = dirname
        self.lr_optimizers = collections.defaultdict(list)
        self.output_path = None

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        optimizers = kwargs.get('optimizers')
        if optimizers is None:
            return
        if self.output_path is None:
            self.output_path = os.path.join(options.workflow_options.current_logging_directory, self.dirname)

        epoch = len(history)
        for optimizer_name, optimizer in optimizers.items():
            if len(optimizer.param_groups) != 1:
                logger.warning('Multiple param groups, don\'t know how to record this!')
                continue
            lr = optimizer.param_groups[0].get('lr')
            if lr is None:
                continue
            self.lr_optimizers[optimizer_name].append((epoch, lr))

    def __del__(self):
        if self.output_path is not None:
            list_of_lr_optimizers = {'optimizer ' + name: [l] for name, l in self.lr_optimizers.items()}

            utilities.create_or_recreate_folder(self.output_path)
            analysis_plots.plot_group_histories(
                self.output_path,
                history_values=list_of_lr_optimizers,
                title='Learning rate by epoch',
                xlabel='Epochs',
                ylabel='Learning rate')
