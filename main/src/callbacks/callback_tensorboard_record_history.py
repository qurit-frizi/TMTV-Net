from .callback_tensorboard import CallbackTensorboardBased
import logging


logger = logging.getLogger(__name__)


class CallbackTensorboardRecordHistory(CallbackTensorboardBased):
    """
    This callback records the history to a tensorboard readable log
    """
    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        root = options.workflow_options.current_logging_directory
        logger.info('started CallbackTensorboardRecordHistory.__call__')

        logger_tb = CallbackTensorboardBased.create_logger(root)
        if logger_tb is None:
            return

        for dataset_name, dataset in history[-1].items():
            for split_name, split in dataset.items():
                for output_name, output in split.items():
                    for metric_name, metric in output.items():
                        if metric is not None:
                            tag = '{}/{}/{}-{}'.format(output_name, metric_name, dataset_name, split_name)

                            # we can't have space in the tag name
                            tag = tag.replace(' ', '_')
                            logger_tb.add_scalar(tag, float(metric), global_step=len(history) - 1)

        logger.info('successfully completed CallbackTensorboardRecordHistory.__call__')
