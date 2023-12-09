import os

from .callback import Callback

def to_value(v):
    """
    Convert where appropriate from tensors to numpy arrays

    Args:
        v: an object. If ``torch.Tensor``, the tensor will be converted to a numpy
            array. Else returns the original ``v``

    Returns:
        ``torch.Tensor`` as numpy arrays. Any other type will be left unchanged
    """
    if isinstance(v, torch.Tensor):
        return v.cpu().data.numpy()
    return v

import graph_reflection
import utilities
import sample_export
import logging
import collections
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


def default_export_filter(filter, path, min_value=-0.05, max_value=0.05):
    """
    Default export of the filter. If interpretable as a 2D image, export a .png, else export a  .npy array

    Args:
        filter: a filter
        path: where export
        min_value: clipping min value of a filter. If `None`, no clipping
        max_value: clipping max value of a filter. If `None`, no clipping

    Returns:
        None
    """
    i = sample_export.as_rgb_image(filter)
    if i is not None:
        i = sample_export.as_image_ui8(i, min_value=min_value, max_value=max_value)
        i = i.transpose((1, 2, 0))
        image = Image.fromarray(i)
        image.save(path + '.png')
    else:
        np.save(path + '.npy', filter)


class CallbackExportConvolutionKernel(Callback):
    """
    Simply export convolutional kernel.

    This can be useful to check over the time if the weights have converger.
    """
    def __init__(self,
                 export_frequency=500,
                 dirname='convolution_kernels',
                 find_convolution_fn=graph_reflection.find_first_forward_convolution,
                 dataset_name=None,
                 split_name=None,
                 export_filter_fn=default_export_filter):
        """
        Args:
            dirname: folder name where to export the explanations
            find_convolution_fn: a function (or a list of functions) to find the convolution kernel
                from the model. Must accept arguments `model, batch`
            dataset_name: the dataset name will be used for the recording
            split_name: the split name will be used for the recording
            export_frequency: export the kernels every `export_frequency` epochs
            export_filter: the function used to export the filters
        """
        self.dirname = dirname
        if not isinstance(find_convolution_fn, collections.Sequence):
            find_convolution_fn = [find_convolution_fn]
        self.find_convolution_fns = find_convolution_fn
        self.kernels = None
        self.kernel_root_path = None
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.export_frequency = export_frequency
        self.export_filter_fn = export_filter_fn

    def first_time(self, options, datasets, model):
        # here we only want to collect the kernels a single time per epoch, so fix the dataset/split names
        if self.dataset_name is None or self.split_name is None:
            self.dataset_name, self.split_name = utilities.find_default_dataset_and_split_names(
                datasets,
                default_dataset_name=self.dataset_name,
                default_split_name=self.split_name)

        if self.dataset_name is None or self.split_name is None:
            logger.error('can\'t find a dataset name or split name!')
            return

        self.kernel_root_path = os.path.join(options.workflow_options.current_logging_directory, self.dirname)
        utilities.create_or_recreate_folder(self.kernel_root_path)

        # find the requested kernels
        kernels = []
        batch = next(iter(datasets[self.dataset_name][self.split_name]))
        for fn in self.find_convolution_fns:
            result = fn(model, batch)
            if result is not None:
                assert 'matched_module' in result, 'must be a dict with key `matched_module`'
                kernel = result['matched_module'].weight
                kernels.append(kernel)
            else:
                logger.error('can\'t find a convolution kernel!')

        logger.info(f'number of convolution kernel found={len(kernels)}')
        self.kernels = kernels

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        epoch = len(history)
        if epoch == 1 or epoch % self.export_frequency == 0:
            logger.info('started CallbackExportConvolutionKernel.__call__')
            if self.kernels is None:
                self.first_time(options, datasets, model)


            for kernel_id, kernel in enumerate(self.kernels):
                kernel_np = to_value(kernel)
                for filter in range(kernel_np.shape[0]):
                    path = os.path.join(self.kernel_root_path, f'kernel{kernel_id}_filter{filter}_iter_{epoch}')
                    self.export_filter_fn(kernel_np[filter], path)

            logger.info('successfully completed CallbackExportConvolutionKernel.__call__!')
