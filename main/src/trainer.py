import torch
import torch.optim
import torch.nn
import collections
import logging
import numpy as np
import numbers
import os
import time
import itertools
from outputs import Output

try:
    from torch.cuda.amp import autocast
except ModuleNotFoundError:
    # PyTorch version did not support autocast
    autocast = None  # type: ignore



def len_batch(batch):
    """

    Args:
        batch: a data split or a `collections.Sequence`

    Returns:
        the number of elements within a data split
    """
    if isinstance(batch, (collections.Sequence, torch.Tensor)):
        return len(batch)

    assert isinstance(batch, collections.Mapping), 'Must be a dict-like structure! got={}'.format(type(batch))

    for name, values in batch.items():
        if isinstance(values, (list, tuple)):
            return len(values)
        if isinstance(values, torch.Tensor) and len(values.shape) != 0:
            return values.shape[0]
        if isinstance(values, np.ndarray) and len(values.shape) != 0:
            return values.shape[0]
    return 0


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

import outputs
from callbacks import callback_epoch_summary, callback_export_classification_report, callback_explain_decision, \
    callback_export_convolution_kernel, callback_export_history, callback_learning_rate_finder, \
    callback_learning_rate_recorder, callback_reporting_augmentations, callback_reporting_best_metrics, \
    callback_reporting_dataset_summary, callback_reporting_epoch_summary, callback_reporting_export_samples, \
    callback_reporting_layer_statistics, callback_reporting_model_summary, callback_reporting_start_server, \
    callback_save_last_model, callback_worst_samples_by_epoch, callback_zip_sources, \
    callback_reporting_learning_rate_recorder, callback_profiler

from utilities import prepare_loss_terms, postprocess_batch, transfer_batch_to_device, \
    log_and_print, default_sum_all_losses, NullableContextManager

logger = logging.getLogger(__name__)


def create_losses_fn(datasets, generic_loss):
    """
    Create a dictionary of loss functions for each of the dataset

    Args:
        datasets: the datasets
        generic_loss: a loss function

    Returns:
        A dictionary of losses for each of the dataset
    """
    losses_fn = collections.OrderedDict()
    for dataset_name in datasets.keys():
        losses_fn[dataset_name] = generic_loss
    return losses_fn


def aggregate_values(values):
    if len(values) == 0:
        return None
    value = values[0]
    if isinstance(value, np.ndarray):
        # concatenate tensors (e.g., softmax output)
        if len(value.shape) == 0:
            return np.average(values)
        else:
            return np.concatenate(values)
    elif isinstance(value, numbers.Number):
        # average numbers (e.g., losses)
        return np.sum(values) / len(values)
    elif isinstance(value, torch.Tensor):
        if len(value.shape) > 0:
            return torch.cat(values)
        else:
            return torch.sum(torch.stack(values)) / len(values)
    elif isinstance(value, Output):
        return values[0]
    elif isinstance(value, list):
        return list(itertools.chain.from_iterable(values))
    else:
        assert 0, 'this type=`{}` is not handled!'.format(type(value))


def aggregate_list_of_dicts(list_of_dicts):
    if len(list_of_dicts) == 0:
        return {}

    keys = list_of_dicts[0].keys()
    aggregated = collections.OrderedDict()
    for key in keys:
        values = [dict[key] for dict in list_of_dicts]
        values = [v for v in values if v is not None]
        aggregated[key] = aggregate_values(values)
    return aggregated


def aggregate_list_of_metrics(list_of_metrics):
    if len(list_of_metrics) == 0:
        return {}

    keys = list_of_metrics[0].keys()
    aggregated = collections.OrderedDict()
    for key in keys:
        values = [dict[key] for dict in list_of_metrics]
        aggregated_values = key.aggregate_metrics(values)
        for name, value in aggregated_values.items():
            aggregated[name] = value
    return aggregated


def generic_aggregate_loss_terms(loss_terms_history):
    """
    Aggregate the loss terms for all the internal_nodes of an epoch

    Args:
        loss_terms_history: a list of loss terms

    Returns:
        a tuple `output, history`. `output` is maintained alive only during the current epoch.
            `history` is kept in memory during the whole training
    """

    if loss_terms_history is None or len(loss_terms_history) == 0:
        return {}, []

    output_names = loss_terms_history[0].keys()

    # aggregate outputs and metrics by output name
    aggregated_outputs = collections.OrderedDict()
    aggregated_metrics = collections.OrderedDict()
    for output_name in output_names:
        loss_term_outputs = []
        loss_term_metrics_results = []
        if output_name == 'overall_loss':
            continue
        for loss_term in loss_terms_history:
            loss_term_output = loss_term[output_name]
            loss_term_metrics_result = loss_term_output.get('metrics_results')
            if loss_term_metrics_result is not None:
                del loss_term_output['metrics_results']
                loss_term_metrics_results.append(loss_term_metrics_result)
            loss_term_outputs.append(loss_term_output)

        aggregated_outputs[output_name] = aggregate_list_of_dicts(loss_term_outputs)
        aggregated_metrics[output_name] = aggregate_list_of_metrics(loss_term_metrics_results)

    # keep the `overall_loss` in the metrics
    overall_losses = []
    for loss_terms in loss_terms_history:
        loss = loss_terms.get('overall_loss')
        if loss is not None:
            overall_losses.append(loss['loss'])

    if len(overall_losses) > 0:
        loss = aggregate_values(overall_losses)
        aggregated_metrics['overall_loss'] = {'loss': loss}

    return aggregated_outputs, aggregated_metrics


def loss_term_cleanup(loss_terms):
    """
    Perform cleanup on all the loss terms

    Requires ``outputs.Output.output_ref_tag`` tag for each loss term, else no cleanup will be done
    for this loss term.

    Args:
        loss_terms: the loss terms to be cleaned up
    """
    for name, loss_term in loss_terms.items():
        ref = loss_term.get(Output.output_ref_tag)
        if ref is not None:
            ref.loss_term_cleanup(loss_term)


def train_loop(
        options,
        device,
        dataset_name,
        split_name,
        split,
        optimizer,
        per_step_scheduler,
        model,
        loss_fn,
        history,
        callbacks_per_batch,
        callbacks_per_batch_loss_terms,
        gradient_scaler=None):
    """
    Run the train loop (i.e., the model parameters will be updated)

    Note:
        If `callbacks_per_batch` or `callbacks_per_batch_loss_terms` raise an exception
        `StopIteration`, the train loop will be stopped

    Args:
        device: the device to be used to optimize the model
        dataset_name: the name of the dataset
        split_name: the name of the split
        split: a dictionary of feature name and values
        optimizer: an optimizer to optimize the model
        per_step_scheduler: scheduler to be applied per-batch
        model: the model to be optimized
        loss_fn: the loss function
        history: a list of history step
        callbacks_per_batch: the callbacks to be performed on each batch. if `None`, no callbacks to be run
        callbacks_per_batch_loss_terms: the callbacks to be performed on each loss term. if `None`, no callbacks to be run
        gradient_scaler: if mixed precision is enabled, this is the scale to be used for the gradient update

    Notes:
        if ``optimizer`` is None, there MUST be a ``.backward()`` to free graph and memory.
    """
    # make sure the model is in training mode (e.g., batch norm, dropout)
    model.train()

    all_loss_terms = []
    
    total_batch_processing_time = 0.0
    batch_processing_last = time.perf_counter()
    loop_started = time.perf_counter()
    total_collate_and_postprocess = 0.0
    nb_samples = 0

    # start by zeroing the gradients. In particular to
    # handle the `gradient_update_frequency` the order
    # of the `zero_grad` and `step` is "reversed"
    if optimizer is not None:
        optimizer.zero_grad()

    try:
        for i, batch in enumerate(split):
            # to simulate larger effective batch size (and save GPU memory)
            # we can update gradient every few batches
            update_parameters = (i + 1) % options.training_parameters.gradient_update_frequency == 0

            assert isinstance(batch, collections.Mapping), 'batch must be a mapping of (feature name, feature values)'
            # calculate the time for batch processing. In particular
            # this may be significant when using large data augmentations
            # and useful to optimize the data processing pipeline
            current_batch_processing = time.perf_counter() - batch_processing_last
            total_batch_processing_time += current_batch_processing

            total_collate_and_postprocess_start = time.perf_counter()
            batch = transfer_batch_to_device(batch, device)
            
            postprocess_batch(dataset_name, split_name, batch, callbacks_per_batch, batch_id=i)
            total_collate_and_postprocess_end = time.perf_counter()
            total_collate_and_postprocess += total_collate_and_postprocess_end - total_collate_and_postprocess_start

            with NullableContextManager(autocast() if gradient_scaler is not None else None):
                assert model.training
                outputs = model(batch)
                if outputs is None:
                    # skip this batch
                    continue

                assert isinstance(outputs, collections.Mapping), 'model must create a dict of outputs'
                loss_terms = prepare_loss_terms(outputs, batch, is_training=True)
                loss = loss_fn(dataset_name, batch, loss_terms)

                # the loss is averaged over the frequency updates
                loss /= options.training_parameters.gradient_update_frequency

            if optimizer is not None and isinstance(loss, torch.Tensor):
                if isinstance(loss, torch.Tensor):
                    # if there is no optimizer, it means we did not want to change the parameters
                    if gradient_scaler is None:
                        loss.backward()
                    else:
                        gradient_scaler.scale(loss).backward()
                else:
                    logger.warning('No backward calculated for={}/{}'.format(dataset_name, split_name))
            loss_terms['overall_loss'] = {'loss': float(to_value(loss))}
            
            if callbacks_per_batch_loss_terms is not None:
                for callback in callbacks_per_batch_loss_terms:
                    callback(
                        dataset_name=dataset_name,
                        split_name=split_name,
                        batch=batch,
                        loss_terms=loss_terms,
                        model=model,
                        optimizer=optimizer,
                        per_step_scheduler=per_step_scheduler
                    )

            # call optimizer step after the callbacks (e.g., a callback could be used to clip the gradient)
            if update_parameters:
                if optimizer is not None:
                    if gradient_scaler is None:
                        optimizer.step()
                    else:
                        gradient_scaler.step(optimizer)
                        gradient_scaler.update()
                    optimizer.zero_grad()

                if per_step_scheduler is not None:
                    per_step_scheduler.step()

            # once we are done, we want to perform some cleanup. For example, we do NOT want to keep CUDA based
            # tensors in the output so we can run clean up to transfer CUDA based memory to numpy
            loss_term_cleanup(loss_terms)

            all_loss_terms.append(loss_terms)
            batch_processing_last = time.perf_counter()
            nb_samples += len_batch(batch)

    except StopIteration:
        pass
    loop_ended = time.perf_counter()
    
    logger.debug('nb_samples={}, train_loop total_batch_processing_time={}, loop_time={},'
                 ' collate_and_postprocess={}, dataset_name={}, split_name={}'.format(
        nb_samples,
        total_batch_processing_time,
        loop_ended - loop_started,
        total_collate_and_postprocess,
        dataset_name,
        split_name))
    return all_loss_terms


def eval_loop(
        options,
        device,
        dataset_name,
        split_name,
        split,
        model,
        loss_fn,
        history,
        callbacks_per_batch=None,
        callbacks_per_batch_loss_terms=None):
    """
    Run the eval loop (i.e., the model parameters will NOT be updated)
    
    Note:
        If `callback_per_batch` or `callbacks_per_batch_loss_terms` raise `StopIteration`, the eval loop will be stopped
    :param device:
    :param dataset_name:
    :param split_name:
    :param split:
    :param model:
    :param loss_fn:
    :param history:
    :param callbacks_per_batch:
    :param callbacks_per_batch_loss_terms:
    :return:
    """
    all_loss_terms = []

    # make sure the model is in eval mode so that non essential operations are removed (e.g., batch norm, dropout)
    model.eval()

    try:
        for i, batch in enumerate(split):
            assert isinstance(batch, collections.Mapping), 'batch must be a mapping of (feature name, feature values)'
            batch = transfer_batch_to_device(batch, device=device)
            postprocess_batch(dataset_name, split_name, batch, callbacks_per_batch, batch_id=i)
            with torch.no_grad():  # do not keep track of the gradient as we are just evaluating
                outputs = model(batch)
                if outputs is None:
                    # skip this batch
                    continue
                loss_terms = prepare_loss_terms(outputs, batch, is_training=False)
                loss = loss_fn(dataset_name, batch, loss_terms)
                loss_terms['overall_loss'] = {'loss': float(to_value(loss))}
                all_loss_terms.append(loss_terms)

                if callbacks_per_batch_loss_terms is not None:
                    for callback in callbacks_per_batch_loss_terms:
                        callback(
                            dataset_name=dataset_name,
                            split_name=split_name,
                            batch=batch,
                            loss_terms=loss_terms,
                            model=model)
                        
                # clean the loss terms (e.g., free memory)
                loss_term_cleanup(loss_terms)

    except StopIteration:
        pass
    return all_loss_terms


def approximate_batch_size_from_loss_terms(all_loss_terms):
    """
    Calculate on approximation of the number of samples from the loss terms. Error can be up to the number of
    samples within one batch
    """
    for name, values in all_loss_terms[0].items():
        s = len_batch(values)
        if s != 0:
            return s * len(all_loss_terms)
    return 0


def epoch_train_eval(
        options,
        datasets,
        optimizers,
        model,
        losses,
        schedulers,
        per_step_schedulers,
        history,
        callbacks_per_batch,
        callbacks_per_batch_loss_terms,
        run_eval,
        force_eval_mode,
        eval_loop_fn=eval_loop,
        train_loop_fn=train_loop):
    """

    Args:
        options:
        datasets:
        optimizers:
        model:
        losses:
        schedulers:
        per_step_schedulers:
        history:
        callbacks_per_batch:
        callbacks_per_batch_loss_terms:
        run_eval:
        force_eval_mode:
        eval_loop_fn:
        train_loop_fn:

    Returns:

    """
    device = options.workflow_options.device
    train_split_name = options.workflow_options.train_split
    history_by_dataset_epoch = collections.OrderedDict()
    outputs_by_dataset_epoch = collections.OrderedDict()
    for dataset_name, dataset in datasets.items():
        optimizer = None
        if optimizers is not None:
            optimizer = optimizers.get(dataset_name)
        loss_fn = losses[dataset_name]
        scheduler = None
        if schedulers is not None:
            scheduler = schedulers.get(dataset_name)

        per_step_scheduler = None
        if per_step_schedulers is not None:
            per_step_scheduler = per_step_schedulers.get(dataset_name)

        dataset_history = collections.OrderedDict()
        dataset_outputs = collections.OrderedDict()
        for split_name, split in dataset.items():
            time_start = time.perf_counter()
            if split_name == train_split_name and not force_eval_mode:
                # * only the split `train_split_name` is considered as training, all
                # other splits are for evaluation only
                # * if we don't have optimizers, we still want to have
                # gradients (e.g., for model with their own internal optimizers)
                all_loss_terms = train_loop_fn(
                    options,
                    device,
                    dataset_name,
                    split_name,
                    split,
                    optimizer,
                    per_step_scheduler,
                    model,
                    loss_fn,
                    history,
                    callbacks_per_batch=callbacks_per_batch,
                    callbacks_per_batch_loss_terms=callbacks_per_batch_loss_terms,
                    gradient_scaler=options.training_parameters.gradient_scaler)
            else:
                if not run_eval or eval_loop_fn is None:
                    # we should not run the evaluation. Skip this!
                    continue

                all_loss_terms = eval_loop_fn(
                    options,
                    device,
                    dataset_name,
                    split_name,
                    split,
                    model,
                    loss_fn,
                    history,
                    callbacks_per_batch=callbacks_per_batch,
                    callbacks_per_batch_loss_terms=callbacks_per_batch_loss_terms)
            time_end = time.perf_counter()
            assert isinstance(all_loss_terms, collections.Sequence), '`all_loss_terms` must be a sequence'

            if len(all_loss_terms) != 0:
                epoch_outputs, epoch_history = generic_aggregate_loss_terms(all_loss_terms)
                epoch_history['info'] = {
                    'time': time_end - time_start,
                    'nb_samples': approximate_batch_size_from_loss_terms(all_loss_terms)
                }
                dataset_history[split_name] = epoch_history
                dataset_outputs[split_name] = epoch_outputs

        history_by_dataset_epoch[dataset_name] = dataset_history
        outputs_by_dataset_epoch[dataset_name] = dataset_outputs

        if scheduler is not None:
            scheduler.step()

    return outputs_by_dataset_epoch, history_by_dataset_epoch


default_logger = log_and_print


def default_pre_training_callbacks(
        logger=default_logger,
        with_lr_finder=False,
        with_export_augmentations=True,
        with_reporting_server=True,
        with_profiler=False,
        additional_callbacks=None):
    """
    Default callbacks to be performed before the fitting of the model
    """
    callbacks = []

    if with_reporting_server:
        callbacks.append(callback_reporting_start_server.CallbackReportingStartServer())

    callbacks += [
        callback_zip_sources.CallbackZipSources(folders_to_record=os.path.join(os.path.dirname(__file__), '..', '..')),

        callback_reporting_model_summary.CallbackReportingModelSummary(),
        callback_reporting_dataset_summary.CallbackReportingDatasetSummary(),
        callback_reporting_export_samples.CallbackReportingExportSamples(table_name='random_samples'),
    ]

    if with_profiler:
        callbacks.append(callback_profiler.CallbackProfiler())
    
    if with_export_augmentations:
        callbacks.append(callback_reporting_augmentations.CallbackReportingAugmentations())

    if with_lr_finder:
        # this may take some time, hence the reason it is disabled by default
        callbacks.append(callback_learning_rate_finder.CallbackLearningRateFinder())

    if additional_callbacks is not None:
        callbacks += additional_callbacks

    return callbacks


def default_per_epoch_callbacks(
        logger=default_logger,
        with_worst_samples_by_epoch=True,
        with_activation_statistics=False,
        convolutional_kernel_export_frequency=None,
        additional_callbacks=None):
    """
    Default callbacks to be performed at the end of each epoch
    """
    callbacks = [
        callback_learning_rate_recorder.CallbackLearningRateRecorder(),
        callback_epoch_summary.CallbackEpochSummary(logger=logger),
        callback_reporting_epoch_summary.CallbackReportingRecordHistory(),
        callback_reporting_best_metrics.CallbackReportingBestMetrics(),
        callback_reporting_learning_rate_recorder.CallbackReportingLearningRateRecorder(),
    ]

    if convolutional_kernel_export_frequency is not None:
        callbacks.append(callback_export_convolution_kernel.CallbackExportConvolutionKernel(
            export_frequency=convolutional_kernel_export_frequency))

    if with_worst_samples_by_epoch:
        callbacks.append(callback_worst_samples_by_epoch.CallbackWorstSamplesByEpoch())

    if with_activation_statistics:
        callbacks.append(callback_reporting_layer_statistics.CallbackReportingLayerStatistics())

    if additional_callbacks is not None:
        callbacks += additional_callbacks

    return callbacks


def default_post_training_callbacks(
        embedding_name='embedding',
        dataset_name=None,
        split_name=None,
        discard_train_error_export=False,
        export_errors=True,
        explain_decision=True,
        additional_callbacks=None):
    """
    Default callbacks to be performed after the model has been trained
    """
    callbacks = [
        callback_save_last_model.CallbackSaveLastModel(),
    ]

    if export_errors:
        callbacks.append(callback_reporting_export_samples.CallbackReportingExportSamples())

    callbacks += [
        callback_export_classification_report.CallbackExportClassificationReport(),
        callback_export_history.CallbackExportHistory(),
    ]

    if explain_decision:
        callbacks.append(callback_explain_decision.CallbackExplainDecision(split_name=split_name))

    if additional_callbacks is not None:
        callbacks += additional_callbacks

    return callbacks


def trainer_callbacks_per_batch(dataset_name, split_name, batch):
    """
    Postprocessing step to be run on the batches (e.g., if we have functors, run the functor and replace it)
    
    :param dataset_name:
    :param split_name:
    :param batch:
    :return:
    """
    for name, value in batch.items():
        # if we have a callable as a batch value, run it and replace its value by the results of the functor
        # (e.g., GAN `z` randomly generated)
        if isinstance(value, collections.Callable):
            batch[name] = value(batch)


def strip_unpickable(outputs):
    """
    Remove the objects that cannot be pickled
    """
    if outputs is None:
        return None

    # TODO not very nice code. Can we generalize this?
    o_d = collections.OrderedDict()
    for dataset_name, dataset in outputs.items():
        o_s = collections.OrderedDict()
        for split_name, split in dataset.items():
            o_n = collections.OrderedDict()
            for output_name, output in split.items():
                o_o = collections.OrderedDict()
                for metric_name, metric in output.items():
                    if 'output_ref' != metric_name:
                        o_o[metric_name] = metric
                o_n[output_name] = o_o
            o_s[split_name] = o_n
        o_d[dataset_name] = o_s
    return o_d



