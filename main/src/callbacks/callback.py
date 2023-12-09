class Callback:
    """
    Defines a callback function that may be called before training, during training, after training
    """
    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        pass
