import datetime
import json
import os
from typing import Dict, Any
from argparse import Namespace
import logging


def read_splits(configuration: Namespace) -> Dict:
    """
    Read the dataset splits based on the configuration.

    use configuration.datasets['splits_path'] or configuration.datasets['splits_full_path']
    """
    splits_full_path = configuration.datasets.get('splits_path')
    splits_name = configuration.datasets.get('splits_name')
    assert int(splits_full_path is not None) + int(splits_name is not None) == 1, \
        'conflicting configuration. Define `splits_full_path` or `splits_name`'

    if splits_full_path is not None:
        path = splits_full_path
    else:
        here = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(here, 'config', splits_name)

    with open(path, 'r') as f:
        data = json.load(f)
    return data


def default(env_name, default_value, output_type: Any = str):
    """
    Read the value of an environment variable or return a default value
    if not defined
    """
    value = os.environ.get(env_name)
    if value is not None:
        return output_type(value)
    return output_type(default_value)


def was_started_within_vscode() -> bool:
    """
    Return True if the script was started from within VSCode.

    VSCode will define an environment variable `STARTED_WITHIN_VSCODE=1`
    when started.
    """
    started_within_vscode = os.getenv('STARTED_WITHIN_VSCODE') == '1'
    return started_within_vscode


def adjust_batch_size_and_lr(configuration: Namespace) -> None:
    """
    Depending if the code was launched within vscode, smaller batch size
    may be used to minimize GPU memory footprint during debugging. This behaviour is controlled 
    by an environment varaible `STARTED_WITHIN_VSCODE` and configuration.training['vscode_batch_size_reduction_factor'].
    Default reduction factor = 5. configuration.training['learning_rate'] is modified accordingly.
    """
    if was_started_within_vscode():
        vscode_batch_size_reduction_factor = configuration.training.get('vscode_batch_size_reduction_factor')
        if vscode_batch_size_reduction_factor is None:
            vscode_batch_size_reduction_factor = 5
        else:
            vscode_batch_size_reduction_factor = int(vscode_batch_size_reduction_factor)

        logging.info(f'vscode_batch_size_reduction_factor={vscode_batch_size_reduction_factor}')
        if vscode_batch_size_reduction_factor != 1:
            learning_rate = configuration.training.get('learning_rate')
            batch_size = int(configuration.training.get('batch_size'))
            assert batch_size is not None

            # reduce the batch size for quick experiment to limit GPU memory size
            # (i.e., we don't care here about the training performance/quality)
            new_batch_size = max(2, int(int(batch_size) / vscode_batch_size_reduction_factor))
            vscode_batch_size_reduction_factor_actual = float(batch_size) / new_batch_size
            configuration.training['batch_size'] = new_batch_size
            logging.info(f'new_batch_size={new_batch_size}, old={batch_size}')

            if learning_rate is not None:
                # when the batch size is scaled, we MUST scale the learning rate by
                # a similar factor, else that would change drastically the learning
                # dynamics
                new_learning_rate = float(learning_rate) / vscode_batch_size_reduction_factor_actual
                configuration.training['learning_rate'] = new_learning_rate
                logging.info(f'new_learning_rate={new_learning_rate}, old={learning_rate}')


def configure_startup(configuration: Namespace) -> None:
    """
    Initialize the logging.

    Depending if the code was launched within vscode, add a suffix `_vscode`.
    This behaviour is controlled by an environment varaible `STARTED_WITHIN_VSCODE`

    Adjust learning rate & batch size if `STARTED_WITHIN_VSCODE`.
    """
    logging_directory = configuration.training['logging_directory']

    if was_started_within_vscode():
        # consider runs started from vscode differently:
        # - typically, those are for debugging purpose only as they are short lived 
        # - logging folder can be erased
        configuration.training['run_name'] = configuration.training['run_name'] + '_vscode'
    else:
        experiment_path = os.path.join(logging_directory, configuration.training['run_name'])
        assert not os.path.exists(experiment_path), f'experiment_path={experiment_path} exists already!'
        ' Change name or delete the folder.'

    os.makedirs(logging_directory, exist_ok=True)
    run_name = configuration.training['run_name']
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        filename=f'{logging_directory}/training_{time_str}_{run_name}.log', 
        encoding='utf-8', 
        level=logging.DEBUG, 
        filemode='w'
    )

    #
    # beware: there is a problem with OpenMP for some
    # distributions:
    #   see: https://github.com/pytorch/pytorch/issues/17199
    #        https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58378
    # since the data augmentation is using multiple processes, 
    # we don't need to multi-thread the tensor processing
    #
    import torch
    torch.set_num_threads(1)

    adjust_batch_size_and_lr(configuration)
    return logging_directory
