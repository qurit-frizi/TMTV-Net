from utilities import postprocess_batch
from .callback_tensorboard import CallbackTensorboardBased
import utilities
import os
import logging
import torch


logger = logging.getLogger(__name__)


class CallbackTensorboardRecordModel(CallbackTensorboardBased):
    """
    This callback will export the model to tensorboard

    @TODO ONNX is probably adding hooks and are not removed. To be investigated.
    """
    def __init__(self, dataset_name=None, split_name=None, onnx_folder='onnx'):
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.onnx_folder = onnx_folder

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        root = options.workflow_options.current_logging_directory
        logger.info('root={}'.format(root))
        logger_tb = CallbackTensorboardBased.create_logger(root)
        if logger_tb is None:
            return

        if self.dataset_name is None:
            if self.dataset_name is None:
                self.dataset_name = next(iter(datasets))

        if self.split_name is None:
            self.split_name = next(iter(datasets[self.dataset_name]))

        device = options.workflow_options.device

        # ONNX export MUST be in eval mode!
        model.eval()

        batch = next(iter(datasets[self.dataset_name][self.split_name]))
        batch = utilities.transfer_batch_to_device(batch, device=device)
        postprocess_batch(self.dataset_name, self.split_name, batch, callbacks_per_batch)

        class NoDictModel(torch.nn.Module):
            # cahce the model input as onnx doesn't handle dict like input
            # or output
            def __init__(self, model, batch):
                super().__init__()
                self.model = model
                self.batch = batch

            def __call__(self, *input, **kwargs):
                with torch.no_grad():
                    r = self.model(self.batch)
                outputs = [o.output for name, o in r.items()]
                return outputs

        # at the moment, few issues with the onnx export with the support
        # of return dictionary is spotty. There are 2 ways options:
        # 1) export to onnx format and use `logger_tb.add_onnx_graph` to export the graph -> this requires the `onnx` additional dependency. This only works for the
        #    latest PyTorch. Additionally, onnx should be installed via conda
        # 2) export directly using `logger_tb.add_graph(NoDictModel(model, batch), input_to_model=torch.Tensor())`, but in the dev build, this fails
        # option 2 is preferable but doesn't work right now

        try:
            # option 1)
            root_onnx = os.path.join(root, self.onnx_folder)
            utilities.create_or_recreate_folder(root_onnx)
            onnx_filepath = os.path.join(root_onnx, 'model.onnx')
            logger.info('exporting ONNX model to `{}`'.format(onnx_filepath))
            with utilities.CleanAddedHooks(model):  # make sure no extra hooks are kept
                with open(onnx_filepath, 'wb') as f:
                    torch.onnx.export(NoDictModel(model, batch), torch.Tensor(), f)  # fake input. The input is already binded in the wrapper!
                logger_tb.add_onnx_graph(onnx_filepath)
                # else there is an assert here. Not sure what is happening

                # option 2)
                #logger_tb.add_graph(NoDictModel(model, batch), input_to_model=torch.Tensor())
            logger.info('successfully exported!')
        except Exception as e:
            logger.error('ONNX export failed! Exception=', str(e))
