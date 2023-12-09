import collections

from .callback import Callback
from .callback_reporting_model_summary import export_table
from utilities import update_json_config, find_default_dataset_and_split_names



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



def collect_hierarchical_parameter_name(base_name, model, parameter_to_name=None, with_grad_only=False):
    """
        Create a meaningful name of the module's parameters based on the module hierarchy
        Args:
            base_name: the base name
            model: the model
            parameter_to_name: where to store the module to name conversion
            with_grad_only: only the parameters requiring gradient are collected
        Returns:
            a dictionary with mapping nn.Parameter to string
        """
    if parameter_to_name is None:
        parameter_to_name = collections.OrderedDict()

    for child_id, child in enumerate(model.children()):
        child_name = base_name + '/' + type(child).__name__ + f'_{child_id}'
        for name, parameter in child.named_parameters(recurse=False):
            if with_grad_only and not parameter.requires_grad:
                # discard if not gradient
                continue
            parameter_name = child_name + '/' + name
            parameter_to_name[parameter] = parameter_name

        collect_hierarchical_parameter_name(
            child_name,
            child,
            parameter_to_name=parameter_to_name,
            with_grad_only=with_grad_only)

    return parameter_to_name


import logging
import torch.nn as nn


logger = logging.getLogger(__name__)


def extract_metrics(p: nn.parameter.Parameter):
    return collections.OrderedDict([
        ('mean', to_value(p.mean())),
        ('max', to_value(p.max())),
        ('min', to_value(p.min())),
        ('std', to_value(p.std())),
        ('norm2', to_value(p.norm())),
    ])


class CallbackReportingLayerWeights(Callback):
    """
    Report the weight statistics of each layer
    """
    def __init__(self, dataset_name=None, split_name=None, table_name='layer_weights'):
        """

        Args:
            split_name: Samples from this split will be used to collect statistics. If `None`, a split
                will be automatically selected
            table_name: the name of the SQL table where the results will be stored
        """
        self.split_name = split_name
        self.dataset_name = dataset_name
        self.table_name = table_name

    def first_time(self, options, datasets):
        # here we only want to collect the kernels a single time per epoch, so fix the dataset/split names
        if self.dataset_name is None or self.split_name is None:
            self.dataset_name, self.split_name = find_default_dataset_and_split_names(
                datasets,
                default_dataset_name=self.dataset_name,
                default_split_name=self.split_name)

            # set the default parameter of the graph
            config_path = options.workflow_options.sql_database_view_path

            update_json_config(config_path, {
                self.table_name: {
                    'default': {
                        'X Axis': 'epoch',
                        'Y Axis': 'metric_value',
                        'Group by': 'parameter',
                        'discard_axis_y': 'epoch',
                        'discard_axis_x': 'metric_value',
                        'discard_group_by': 'epoch',
                        'number_of_columns': 2,
                    }
                }
            })

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch,
                 **kwargs):

        if self.dataset_name is None or self.split_name is None:
            self.first_time(options, datasets)

        if self.dataset_name is None or self.split_name is None:
            logger.error('can\'t find a dataset name or split name!')
            return

        parameter_to_name = collect_hierarchical_parameter_name(type(model).__name__, model, with_grad_only=True)

        parameter_names = []
        epochs = []
        metrics = []
        metric_values = []
        for p in model.parameters():
            name = parameter_to_name.get(p)
            if name is None:
                # parameters that don't have gradients are not of interest
                # so discard them
                continue

            metrics_kvp = extract_metrics(p)
            for metric_name, metric_value in metrics_kvp.items():
                parameter_names.append(name)
                epochs.append(len(history))
                metrics.append(metric_name)
                metric_values.append(metric_value)

        table = collections.OrderedDict([
            ('parameter', parameter_names),
            ('epoch', epochs),
            ('metric', metrics),
            ('metric_value', metric_values),
        ])

        logger.info('exporting layer gradient to SQL...')

        export_table(
            options,
            self.table_name,
            table,
            table_role='data_graph',
            clear_existing_data=False)


