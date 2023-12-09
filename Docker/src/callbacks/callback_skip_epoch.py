import collections

from .callback import Callback


class CallbackSkipEpoch(Callback):
    """
    Run its callbacks every few epochs
    """
    def __init__(self, nb_epochs, callbacks, include_epoch_zero=False):
        """

        Args:
            nb_epochs: the number of epochs to skip
            callbacks: a list of callbacks to be called
            include_epoch_zero: if ``True``, epoch 0 will be included, if ``False`` all
                callbacks are discarded for epoch 0
        """
        self.include_epoch_zero = include_epoch_zero
        self.nb_epochs = nb_epochs
        self.callbacks = callbacks
        assert isinstance(callbacks, collections.Sequence), f'callbacks must be a sequence! Got type={type(callbacks)}'

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        if len(history) % self.nb_epochs == 0 or (self.include_epoch_zero and len(history) == 1):
            for callback in self.callbacks:
                callback(options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch)
