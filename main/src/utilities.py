import json
import time
import logging
from typing import Optional, Any

import torch
import shutil
import os
import numpy as np
import collections
import datetime
import traceback as traceback_module
import io
import torch.nn as nn
from basic_typing import History, DatasetsInfo
from options import Options

logger = logging.getLogger(__name__)


def safe_filename(filename):
    """
    Clean the filename so that it can be used as a valid filename
    """
    return filename.\
        replace('=', ''). \
        replace('\n', ''). \
        replace('/', '_'). \
        replace('\\', '_'). \
        replace('$', ''). \
        replace(';', ''). \
        replace('*', '_')


def log_info(msg):
    """
    Log the message to a log file as info
    :param msg:
    :return:
    """
    logger.info(msg)


def log_and_print(msg):
    """
    Log the message to a log file as info
    :param msg:
    :return:
    """
    logger.info(msg)
    # make sure we flush the output to have reliable reading
    # in case the output is redirected to a file (e.g., SLURM job)
    print(msg, flush=True)


def log_console(msg):
    """
    Log the message to the console
    :param msg:
    :return:
    """

    # make sure we flush the output to have reliable reading
    # in case the output is redirected to a file (e.g., SLURM job)
    print(msg, flush=True)


# from .utils import to_value, recursive_dict_update


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


def recursive_dict_update(dict, dict_update):
    """
    This adds any missing element from ``dict_update`` to ``dict``, while keeping any key not
        present in ``dict_update``

    Args:
        dict: the dictionary to be updated
        dict_update: the updated values
    """
    for updated_name, updated_values in dict_update.items():
        if updated_name not in dict:
            # simply add the missing name
            dict[updated_name] = updated_values
        else:
            values = dict[updated_name]
            if isinstance(values, collections.Mapping):
                # it is a dictionary. This needs to be recursively
                # updated so that we don't remove values in the existing
                # dictionary ``dict``
                recursive_dict_update(values, updated_values)
            else:
                # the value is not a dictionary, we can update directly its value
                dict[updated_name] = values


def create_or_recreate_folder(path, nb_tries=3, wait_time_between_tries=2.0):
    """
    Check if the path exist. If yes, remove the folder then recreate the folder, else create it

    Args:
        path: the path to create or recreate
        nb_tries: the number of tries to be performed before failure
        wait_time_between_tries: the time to wait before the next try

    Returns:
        ``True`` if successful or ``False`` if failed.
    """
    assert len(path) > 6, 'short path? just as a precaution...'
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

    def try_create():
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            print('[ignored] create_or_recreate_folder error:', str(e))
            return False

    # multiple tries (e.g., for windows if we are using explorer in the current path)
    import threading
    for i in range(nb_tries):
        is_done = try_create()
        if is_done:
            return True
        threading.Event().wait(wait_time_between_tries)  # wait some time for the FS to delete the files
    return False


def make_unique_colors():
    """
    Return a set of unique and easily distinguishable colors
    :return: a list of RBG colors
    """
    return [
        (79, 105, 198),  # indigo
        (237, 10, 63),  # red
        (175, 227, 19),  # inchworm
        (193, 84, 193),  # fuchsia
        (175, 89, 62),  # brown
        (94, 140, 49),  # maximum green
        (255, 203, 164),  # peach
        (200, 200, 205),  # blue-Gray
        (255, 255, 255),  # white
        (115, 46, 108),  # violet(I)
        (157, 224, 147),  # granny-smith
    ]


def make_unique_colors_f():
    """
    Return a set of unique and easily distinguishable colors
    :return: a list of RBG colors
    """
    return [
        (79 / 255.0, 105 / 255.0, 198 / 255.0),  # indigo
        (237 / 255.0, 10 / 255.0, 63 / 255.0),  # red
        (175 / 255.0, 227 / 255.0, 19 / 255.0),  # inchworm
        (193 / 255.0, 84 / 255.0, 193 / 255.0),  # fuchsia
        (175 / 255.0, 89 / 255.0, 62 / 255.0),  # brown
        (94 / 255.0, 140 / 255.0, 49 / 255.0),  # maximum green
        (255 / 255.0, 203 / 255.0, 164 / 255.0),  # peach
        (200 / 255.0, 200 / 255.0, 205 / 255.0),  # blue-Gray
        (245 / 255.0, 245 / 255.0, 245 / 255.0),  # white (not fully white, the background is often white!)
        (115 / 255.0, 46 / 255.0, 108 / 255.0),  # violet(I)
        (157 / 255.0, 224 / 255.0, 147 / 255.0),  # granny-smith
    ]


def get_class_name(mapping, classid):
    classid = int(classid)
    if mapping is None:
        return None
    return mapping['mappinginv'].get(classid)


def get_classification_mappings(datasets_infos, dataset_name, split_name):
    """
        Return the output mappings of a classification output from the datasets infos

        :param datasets_infos: the info of the datasets
        :param dataset_name: the name of the dataset
        :param split_name: the split name
        :return: a dictionary {outputs: {'mapping': {name->ID}, 'mappinginv': {ID->name}}}
        """
    if datasets_infos is None or dataset_name is None or split_name is None:
        return None
    dataset_infos = datasets_infos.get(dataset_name)
    if dataset_infos is None:
        return None

    split_datasets_infos = dataset_infos.get(split_name)
    if split_datasets_infos is None:
        return None

    return split_datasets_infos.get('output_mappings')


def get_classification_mapping(datasets_infos, dataset_name, split_name, output_name):
    """
    Return the output mappings of a classification output from the datasets infos

    :param datasets_infos: the info of the datasets
    :param dataset_name: the name of the dataset
    :param split_name: the split name
    :param output_name: the output name
    :return: a dictionary {'mapping': {name->ID}, 'mappinginv': {ID->name}}
    """
    if output_name is None:
        return None
    output_mappings = get_classification_mappings(datasets_infos, dataset_name, split_name)
    if output_mappings is None:
        return None
    return output_mappings.get(output_name)


def set_optimizer_learning_rate(optimizer, learning_rate):
        """
        Set the learning rate of the optimizer to a specific value

        Args:
            optimizer: the optimizer to update
            learning_rate: the learning rate to set

        Returns:
            None
        """

        # manually change the learning rate. References:
        # - https://discuss.pytorch.org/t/change-learning-rate-in-pytorch/14653
        # - https://discuss.pytorch.org/t/adaptive-learning-rate/320/36
        for param_group in optimizer.param_groups:
            # make sure we have an exising learning rate parameters. If not, it means pytorch changed
            # OR the current optimizer is not supported
            assert 'lr' in param_group, 'internally, the optimizer is not using a learning rate!'
            param_group['lr'] = learning_rate


def transfer_batch_to_device(batch, device, non_blocking=True):
    """
    Transfer the Tensors and numpy arrays to the specified device. Other types will not be moved.

    Args:
        batch: the batch of data to be transferred
        device: the device to move the tensors to
        non_blocking: non blocking memory transfer to GPU

    Returns:
        a batch of data on the specified device
    """

    device_batch = collections.OrderedDict()
    for name, value in batch.items():
        if isinstance(value, np.ndarray):
            # `torch.from_numpy` to keep the same dtype as our input
            device_batch[name] = torch.as_tensor(value).to(device, non_blocking=non_blocking)
        elif isinstance(value, torch.Tensor) and value.device != device:
            device_batch[name] = value.to(device, non_blocking=non_blocking)
        else:
            device_batch[name] = value
    return device_batch


class NullableContextManager:
    """
    Accept `None` context manager. In that case do nothing, else execute
    the context manager enter and exit.

    This is a helper class to simplify the handling of possibly None context manager.
    """
    def __init__(self, base_context_manager: Optional[Any]):
        self.base_context_manager = base_context_manager

    def __enter__(self):
        if self.base_context_manager is not None:
            self.base_context_manager.__enter__()

    def __exit__(self, type, value, traceback):
        if self.base_context_manager is not None:
            self.base_context_manager.__exit__(type, value, traceback)


class CleanAddedHooks:
    """
    Context manager that automatically track added hooks on the model and remove them when
    the context is released
    """
    def __init__(self, model):
        self.initial_hooks = {}
        self.model = model
        self.nb_hooks_removed = 0  # record the number of hooks deleted after the context is out of scope

    def __enter__(self):
        self.initial_module_hooks_forward, self.initial_module_hooks_backward = CleanAddedHooks.record_hooks(self.model)
        return self

    def __exit__(self, type, value, traceback):
        def remove_hooks(hooks_initial, hooks_final, is_forward):
            for module, hooks in hooks_final.items():
                if module in hooks_initial:
                    added_hooks = hooks - hooks_initial[module]
                else:
                    added_hooks = hooks

                for hook in added_hooks:
                    if is_forward:
                        self.nb_hooks_removed += 1
                        del module._forward_hooks[hook]
                    else:
                        self.nb_hooks_removed += 1
                        del module._backward_hooks[hook]

        all_hooks_forward, all_hooks_backward = CleanAddedHooks.record_hooks(self.model)
        remove_hooks(self.initial_module_hooks_forward, all_hooks_forward, is_forward=True)
        remove_hooks(self.initial_module_hooks_backward, all_hooks_backward, is_forward=False)

        if traceback is not None:
            io_string = io.StringIO()
            traceback_module.print_tb(traceback, file=io_string)

            print('Exception={}'.format(io_string.getvalue()))
            logger.error('CleanAddedHooks: exception={}'.format(io_string.getvalue()))

        return True

    @staticmethod
    def record_hooks(module_source):
        """
        Record hooks
        Args:
            module_source: the module to track the hooks

        Returns:
            at tuple (forward, backward). `forward` and `backward` are a dictionary of hooks ID by module
        """
        modules_kvp_forward = {}
        modules_kvp_backward = {}
        for module in module_source.modules():
            if len(module._forward_hooks) > 0:
                modules_kvp_forward[module] = set(module._forward_hooks.keys())

            if len(module._backward_hooks) > 0:
                modules_kvp_backward[module] = set(module._backward_hooks.keys())
        return modules_kvp_forward, modules_kvp_backward


def get_device(module, batch=None):
    """
    Return the device of a module. This may be incorrect if we have a module split accross different devices
    """
    try:
        p = next(module.parameters())
        return p.device
    except StopIteration:
        # the model doesn't have parameters!
        pass

    if batch is not None:
        # try to guess the device from the batch
        for name, value in batch.items():
            if isinstance(value, torch.Tensor):
                return value.device

    # we can't make an appropriate guess, just fail!
    return None


class RuntimeFormatter(logging.Formatter):
    """
    Report the time since this formatter is instantiated
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()

    def formatTime(self, record, datefmt=None):
        return str(datetime.timedelta(seconds=record.created - self.start_time))


def find_default_dataset_and_split_names(datasets, default_dataset_name=None, default_split_name=None, train_split_name=None):
    """
    Return a good choice of dataset name and split name, possibly not the train split.

    Args:
        datasets: the datasets
        default_dataset_name: a possible dataset name. If `None`, find a suitable dataset, if not, the dataset
            must be present
        default_split_name: a possible split name. If `None`, find a suitable split, if not, the dataset
            must be present. if `train_split_name` is specified, the selected split name will be different from `train_split_name`
        train_split_name: if not `None`, exclude the train split

    Returns:
        a tuple (dataset_name, split_name)
    """
    if default_dataset_name is None:
        default_dataset_name = next(iter(datasets))
    else:
        if default_dataset_name not in datasets:
            return None, None

    if default_split_name is None:
        available_splits = datasets[default_dataset_name].keys()
        for split_name in available_splits:
            if split_name != train_split_name:
                default_split_name = split_name
                break
    else:
        if default_split_name not in datasets[default_dataset_name]:
            return None, None

    return default_dataset_name, default_split_name


def make_triplet_indices(targets):
    """
    Make random index triplets (anchor, positive, negative) such that ``anchor`` and ``positive``
        belong to the same target while ``negative`` belongs to a different target

    Args:
        targets: a 1D integral tensor in range [0..C]

    Returns:
        a tuple of indices (samples, samples_positive, samples_negative)
    """
    # group samples by class
    samples_by_class = collections.defaultdict(list)
    targets = to_value(targets)
    for index, c in enumerate(targets):
        samples_by_class[c].append(index)

    # create the (sample, sample+, sample-) groups
    samples_all = []
    samples_positive_all = []
    samples_negative_all = []
    for c, c_indexes in samples_by_class.items():
        samples = c_indexes.copy()
        samples_positive = c_indexes
        np.random.shuffle(c_indexes)

        other = [idx for cc, idx in samples_by_class.items() if cc != c]
        other = np.concatenate(other)

        # sample with replacement in case the ``negative`` sample are less
        # than the ``positive`` samples
        samples_negative = np.random.choice(other, len(samples))

        samples_all.append(samples)
        samples_positive_all.append(samples_positive)
        samples_negative_all.append(samples_negative)

    samples_all = np.concatenate(samples_all)
    samples_positive_all = np.concatenate(samples_positive_all)
    samples_negative_all = np.concatenate(samples_negative_all)
    min_samples = min(len(samples_all), len(samples_negative_all))
    return samples_all[:min_samples], samples_positive_all[:min_samples], samples_negative_all[:min_samples]


def make_pair_indices(targets, same_target_ratio=0.5):
    """
    Make random indices of pairs of samples that belongs or not to the same target.

    Args:
        same_target_ratio: specify the ratio of same target to be generated for sample pairs
        targets: a 1D integral tensor in range [0..C] to be used to group the samples
            into same or different target

    Returns:
        a tuple with (samples_0 indices, samples_1 indices, same_target)
    """
    # group samples by class
    samples_by_class = collections.defaultdict(list)
    classes = to_value(targets)
    for index, c in enumerate(classes):
        samples_by_class[c.item()].append(index)
    samples_by_class = {name: np.asarray(value) for name, value in samples_by_class.items()}

    # create the (sample, sample+, sample-) groups
    samples_0 = []
    samples_1 = []
    same_target = []
    for c, c_indexes in samples_by_class.items():
        samples = c_indexes.copy()
        np.random.shuffle(c_indexes)
        nb_same_targets = int(same_target_ratio * len(c_indexes))

        other = [idx for cc, idx in samples_by_class.items() if cc != c]
        other = np.concatenate(other)
        np.random.shuffle(other)

        samples_0.append(samples)
        samples_positive = c_indexes[:nb_same_targets]
        same_target += [1] * len(samples_positive)
        # expect to have more negative than positive, so for the negative
        # pick the remaining
        samples_negative = other[:len(c_indexes) - len(samples_positive)]
        same_target += [0] * len(samples_negative)
        samples_1.append(np.concatenate((samples_positive, samples_negative)))

    # in case the assumption was wrong (we, in fact, have more positive than negative)
    # shorten the batch
    samples_0 = samples_0[:len(samples_1)]

    return np.concatenate(samples_0), np.concatenate(samples_1), np.asarray(same_target)


def update_json_config(path_to_json, config_update):
    """
    Update a JSON document stored on a local drive.

    Args:
        path_to_json: the path to the local JSON configuration
        config_update: a possibly nested dictionary

    """
    if os.path.exists(path_to_json):
        with open(path_to_json, 'r') as f:
            text = f.read()
        config = json.loads(text)
    else:
        config = collections.OrderedDict()

    recursive_dict_update(config, config_update)

    json_str = json.dumps(config, indent=3)
    with open(path_to_json, 'w') as f:
        f.write(json_str)


def prepare_loss_terms(outputs, batch, is_training):
    """
    Return the loss_terms for the given outputs
    """
    from outputs import Output

    loss_terms = collections.OrderedDict()
    for output_name, output in outputs.items():
        assert isinstance(output, Output), f'output must be a `Output`' \
                                                       f' instance. Got={type(output)}'
        loss_term = output.evaluate_batch(batch, is_training)
        if loss_term is not None:
            loss_terms[output_name] = loss_term
    return loss_terms


def default_sum_all_losses(dataset_name, batch, loss_terms):
    """
    Default loss is the sum of all loss terms
    """
    sum_losses = 0.0
    for name, loss_term in loss_terms.items():
        loss = loss_term.get('loss')
        if loss is not None:
            # if the loss term doesn't contain a `loss` attribute, it means
            # this is not used during optimization (e.g., embedding output)
            sum_losses += loss
    return sum_losses


def postprocess_batch(dataset_name, split_name, batch, callbacks_per_batch, batch_id=None):
    """
    Post process a batch of data (e.g., this can be useful to add additional
    data to the current batch)

    Args:
        dataset_name (str): the name of the dataset the `batch` belongs to
        split_name (str): the name of the split the `batch` belongs to
        batch: the current batch of data
        callbacks_per_batch (list): the callbacks to be executed for each batch.
            Each callback must be callable with `(dataset_name, split_name, batch)`.
            if `None`, no callbacks
        batch_id: indicate the current batch within an epoch. May be ``None``. This can be useful
            for embedding optimizer within a module (e.g., scheduler support)
    """

    # always useful: for example if we do a model composed of multiple sub-models (one per dataset)
    # we would need to know what sub-model to use
    batch['dataset_name'] = dataset_name
    batch['split_name'] = split_name
    if batch_id is not None:
        batch['batch_id'] = batch_id

    if callbacks_per_batch is not None:
        for callback in callbacks_per_batch:
            callback(dataset_name, split_name, batch)


def apply_spectral_norm(
        module,
        n_power_iterations=1,
        eps=1e-12,
        dim=None,
        name='weight',
        discard_layer_types=(torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d)):
    """
    Apply spectral norm on every sub-modules

    Args:
        module: the parent module to apply spectral norm
        discard_layer_types: the layers_legacy of this type will not have spectral norm applied
        n_power_iterations: number of power iterations to calculate spectral norm
        eps: epsilon for numerical stability in calculating norms
        dim: dimension corresponding to number of outputs, the default is ``0``,
            except for modules that are instances of ConvTranspose{1,2,3}d, when it is ``1``
        name: name of weight parameter

    Returns:
        the same module as input module
    """
    def apply_sn(m):
        for layer in discard_layer_types:
            if isinstance(m, layer):
                # abort: don't apply SN on this layer
                return

        if hasattr(m, name):
            logger.info(f'applying spectral norm={m}')
            nn.utils.spectral_norm(m, n_power_iterations=n_power_iterations, eps=eps, dim=dim, name=name)

    module.apply(apply_sn)
    return module


def apply_gradient_clipping(module: nn.Module, value):
    """
    Apply gradient clipping recursively on a module as callback.

    Every time the gradient is calculated, it is intercepted and clipping applied.

    Args:
        module: a module where sub-modules will have their gradients clipped
        value: the maximum value of the gradient

    """
    assert value >= 0
    for p in module.parameters():
        p.register_hook(lambda gradient: torch.clamp(gradient, -value, value))


class RunMetadata:
    def __init__(self,
                 options: Optional[Options],
                 history: Optional[History],
                 outputs: Optional[Any],
                 datasets_infos: Optional[DatasetsInfo] = None,
                 class_name: Optional[str] = None):

        self.class_name = class_name
        self.datasets_infos = datasets_infos
        self.outputs = outputs
        self.history = history
        self.options = options
