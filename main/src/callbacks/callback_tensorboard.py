import logging
import os
from .callback import Callback

logger = logging.getLogger(__name__)


with_tensorboardX = True
try:
    # tensorboard and tensorboardX are optional dependencies. If this module can not be imported
    # then no tensorboard logging will be performed
    import tensorboardX
except ImportError:
    with_tensorboardX = False
    logger.error('package `tensorboardX` could not be imported. Tensorboard callbacks will be disabled!')


class CallbackTensorboardBased(Callback):
    """
    Tensorboard based callback. Manages a single `tensorboardX.SummaryWriter` instance
    """
    _tensorboard_logger = None


    @staticmethod
    def create_logger(path):
        """
        Create a `tensorboardX.SummaryWriter` instance. If an instance already exists or
        tensorboardX could not be imported, no logger will be created
        :param path: where to write the tensorboard log
        :return: a logger or None if logger creation failed
        """
        if with_tensorboardX and CallbackTensorboardBased._tensorboard_logger is None:
            # we must have a unique name so that we can load several tensorboard logfiles
            # in tensorboard (e.g., to compare multiple models)
            log_path = os.path.join(path, 'tensorboard')
            CallbackTensorboardBased._tensorboard_logger = tensorboardX.SummaryWriter(log_dir=log_path)
        return CallbackTensorboardBased._tensorboard_logger

    @staticmethod
    def get_tensorboard_logger():
        """
        :return: None if the tensorboad logger was not created or a `tensorboardX.SummaryWriter`
        """
        return CallbackTensorboardBased._tensorboard_logger

    @staticmethod
    def remove_tensorboard_logger():
        """
        Remove the current `tensorboardX.SummaryWriter`
        """
        CallbackTensorboardBased._tensorboard_logger = None


class CallbackClearTensorboardLog(CallbackTensorboardBased):
    """
    Remove any existing logger

    This is useful when we train multiple models so that they have their own tensorboard log file
    """
    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        CallbackTensorboardBased.remove_tensorboard_logger()
        logger.debug('CallbackTensorboardBased.remove_tensorboard_logger called!')

