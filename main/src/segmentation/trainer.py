import pickle
from typing import Any
import torch
from .callbacks import default_per_epoch_callbacks, default_post_training_callbacks, default_pre_training_callbacks
from data_parallel_extented import gather_extended, DataParallelExtended
from torch import nn
from options import Options
from trainer_v2 import TrainerV2



try:
    from torch.cuda.amp import autocast
except ModuleNotFoundError:
    # PyTorch version did not support autocast
    def do_nothing_fn():
        pass
    autocast = lambda: lambda x: do_nothing_fn()


class DataParallelExtendedAutocast(nn.DataParallel):
    """
    Customized version of :class:`torch.nn.DataParallel` to support model with
    complex outputs such as :class:`Output`
    """
    def __init__(self, *arg, **argv):
        super().__init__(*arg, **argv)

    def gather(self, outputs, output_device):
        return gather_extended(outputs, output_device, dim=self.dim)

    @autocast()
    def forward(self, *inputs, **kwargs):
        return nn.DataParallel.forward(self, *inputs, **kwargs)


def load_model(
        model: nn.Module,
        path: str,
        device: torch.device = None,
        pickle_module: Any = pickle,
        strict: bool = True,
        flatten_parameter_name: bool = True):
    model_state = torch.load(path, map_location=device, pickle_module=pickle_module)
    
    if flatten_parameter_name:
        # if we used multiple GPUs, the model
        # will be nested. flatten the name
        model_state = {
            name.replace('module.model.', 'model.'): value for name, value in model_state.items()
        }

    model.load_state_dict(model_state, strict=strict)



def run_trainer(
        configuration, 
        datasets, 
        model, 
        optimizers_fn):
    """
    Configure the trainer so that all experiments share the same
    training pipeline.
    """
    
    device = configuration.training['device']
    devices = device.split(';')
    mixed_precision_enabled = configuration.training.get('mixed_precision_enabled')
    if len(devices) > 1:
        devices = [torch.device(d) for d in devices]
        if mixed_precision_enabled:
            model = DataParallelExtendedAutocast(model, device_ids=devices)
        else:
            model = DataParallelExtended(model, device_ids=devices)

    options = Options(
        num_epochs=configuration.training['nb_epochs'],
        logging_directory=configuration.training['logging_directory'],
        device=devices[0],
        gradient_update_frequency=configuration.training['gradient_update_frequency'],
        mixed_precision_enabled=mixed_precision_enabled
    )
    options.runtime.configuration = configuration
    
    trainer = TrainerV2(
        callbacks_pre_training=default_pre_training_callbacks(configuration=configuration, model=model),
        callbacks_per_epoch=default_per_epoch_callbacks(configuration=configuration, model=model),
        callbacks_post_training=default_post_training_callbacks(configuration=configuration, model=model),
        skip_eval_epoch_0=True
    )

    trainer.fit(
        options,
        datasets=datasets,
        model=model,
        log_path=configuration.training['run_name'],
        eval_every_X_epoch=configuration.training['eval_every_X_epoch'],
        optimizers_fn=optimizers_fn)