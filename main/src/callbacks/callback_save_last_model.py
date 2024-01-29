import math
from typing import List, Callable, Optional, Sequence

import torch

from basic_typing import Datasets

def safe_lookup(dictionary, *keys, default=None):
    """
    Recursively access nested dictionaries
    Args:
        dictionary: nested dictionary
        *keys: the keys to access within the nested dictionaries
        default: the default value if dictionary is ``None`` or it doesn't contain
            the keys
    Returns:
        None if we can't access to all the keys, else dictionary[key_0][key_1][...][key_n]
    """
    if dictionary is None:
        return default

    for key in keys:
        dictionary = dictionary.get(key)
        if dictionary is None:
            return default

    return dictionary

from .callback import Callback
from outputs import OutputEmbedding
from utilities import RunMetadata
import os
import logging

logger = logging.getLogger(__name__)


class ModelWithLowestMetricBase:
    def update(self, metric_value, model, metadata, root_path):
        raise NotImplementedError()


class ModelWithLowestMetric(ModelWithLowestMetricBase):
    def __init__(self, dataset_name, split_name, output_name, metric_name, minimum_metric=0.2):
        """

        Args:
            dataset_name: the dataset name to be considered for the best model
            split_name: the split name to be considered for the best model
            metric_name: the metric name to be considered for the best model
            minimum_metric: consider only the metric lower than this threshold
            output_name: the output to be considered for the best model selection
        """
        self.output_name = output_name
        self.metric_name = metric_name
        self.split_name = split_name
        self.dataset_name = dataset_name
        self.minimum_metric = minimum_metric
        self.best_metric = 1e10

    def update(self, metric_value, model, metadata, root_path):
        """
        Check the metrics and export the model if thresholds are satisfied
        """
        if metric_value is not None and metric_value < self.minimum_metric and metric_value < self.best_metric:
            self.best_metric = metric_value
            export_path = os.path.join(
                root_path,
                f'best.model')
            from ..train.trainer_v2 import TrainerV2
            TrainerV2.save_model(model, metadata, export_path)


def exclude_large_embeddings(outputs: Datasets, counts_greater_than=10000) -> Optional[Datasets]:
    """
    Remove from the outputs embeddings larger than a specified threshold.

    Args:
        outputs: the outputs to check
        counts_greater_than: the number of elements above which the embedding will be stripped

    Returns:
        outputs
    """
    if outputs is None:
        return outputs

    for dataset_name, dataset in outputs.items():
        for split_name, split in dataset.items():
            # first collect the outputs to be discarded
            outputs_to_remove = []
            for output_name, output in split.items():
                output_ref = output.get('output_ref')
                if output_ref is not None and isinstance(output_ref, OutputEmbedding):
                    count = output['output'].reshape(-1).shape[0]
                    if count >= counts_greater_than:
                        outputs_to_remove.append(output_name)

            # first collect the outputs to be discarded
            for output_to_remove in outputs_to_remove:
                split[output_to_remove] = {}
    return outputs


def should_not_export_model(last_step, revert_if_nan_metrics):
    for name_dataset, dataset in last_step.items():
        for name_split, split in dataset.items():
            for name_metric, metrics in split.items():
                for m in revert_if_nan_metrics:
                    value = metrics.get(m)
                    if value is not None and math.isnan(value):
                        # NaN should NOT be exported
                        logger.warning(f'NaN detected! {name_dataset}/{name_split}/{name_metric}/{m}')
                        return True
    return False


class CallbackSaveLastModel(Callback):
    """
    Save the current model to disk as well as metadata (history, outputs, infos).

    This callback can be used during training (e.g., checkpoint) or at the end of the training.

    Optionally, record the best model for a given dataset, split, output and metric.
    """

    def __init__(
            self,
            model_name='last',
            with_outputs=False,
            is_versioned=False,
            rolling_size=None,
            keep_model_with_best_metric: ModelWithLowestMetric = None,
            revert_if_nan_metrics: Optional[Sequence[str]] = ('loss',),
            post_process_outputs: Optional[Callable[[Datasets], Datasets]] = exclude_large_embeddings,
    ):
        """
        Args:
            model_name: the root name of the model
            with_outputs: if True, the outputs will be exported along the model
            is_versioned: if versioned, model name will include the current epoch so that we can have multiple
                versions of the same model
            rolling_size: the number of model files that are kept on the drive. If more models are exported,
                the oldest model files will be erased
            keep_model_with_best_metric: if not None, the best model for a given metric will be recorded
            post_process_outputs: a function to post-process the outputs just before export. For example,
                if can be used to remove large embeddings to save smaller output files.
            revert_if_nan_metrics: if any of the metrics have NaN, reload the model from the last checkpoint
        """
        if keep_model_with_best_metric is not None:
            assert isinstance(keep_model_with_best_metric, ModelWithLowestMetricBase), \
                'must be ``None`` or ``ModelWithLowestMetric`` instance'
        self.keep_model_with_best_metric = keep_model_with_best_metric
        self.model_name = model_name
        self.with_outputs = with_outputs
        self.is_versioned = is_versioned
        self.rolling_size = rolling_size
        self.last_models: List[str] = []
        self.post_process_outputs = post_process_outputs
        self.revert_if_nan_metrics = revert_if_nan_metrics
        self.last_model_path = None

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch,
                 **kwargs):

        metadata = RunMetadata(
            options=options,
            history=history,
            outputs=outputs,
            datasets_infos=datasets_infos,
        )

        if not self.with_outputs:
            # discard the outputs (e.g., for large outputs)
            metadata.outputs = None
        elif self.post_process_outputs is not None:
            metadata.outputs = self.post_process_outputs(metadata.outputs)

        if self.is_versioned:
            name = f'{self.model_name}_e_{len(history)}.model'
        else:
            name = f'{self.model_name}.model'
        export_path = os.path.join(options.workflow_options.current_logging_directory, name)

        # verify the metrics are not NaN
        from trainer_v2 import TrainerV2
        from utilities import get_device
        if self.revert_if_nan_metrics is not None and len(history) > 0:
            should_not_export = should_not_export_model(history[-1], self.revert_if_nan_metrics)

            if should_not_export:
                if self.last_model_path is not None:
                    # revert the model
                    device = get_device(model)
                    model_state = torch.load(self.last_model_path, map_location=device)
                    model.load_state_dict(model_state)

                    # do not export it again so return!
                    logger.info(f'model was reverted from={self.last_model_path}')
                    return
                else:
                    # abort!
                    logger.info('model was not reverted, no previously exported model!')
                    return

        logger.info('started CallbackSaveLastModel.__call__ path={}'.format(export_path))
        TrainerV2.save_model(model, metadata, export_path)
        self.last_model_path = export_path

        if self.rolling_size is not None and self.rolling_size > 0:
            self.last_models.append(export_path)

            if len(self.last_models) > self.rolling_size:
                model_location_to_delete = self.last_models.pop(0)
                model_result_location_to_delete = model_location_to_delete + '.metadata'
                logger.info(f'deleted model={model_location_to_delete}')
                os.remove(model_location_to_delete)
                os.remove(model_result_location_to_delete)

        if self.keep_model_with_best_metric is not None:
            # look up the correct metric and record the model and results
            # if we obtain a better (lower) metric.
            metric_value = safe_lookup(
                history[-1],
                self.keep_model_with_best_metric.dataset_name,
                self.keep_model_with_best_metric.split_name,
                self.keep_model_with_best_metric.output_name,
                self.keep_model_with_best_metric.metric_name,
            )

            if metric_value is not None:
                self.keep_model_with_best_metric.update(
                    metric_value, 
                    model, 
                    metadata, 
                    options.workflow_options.current_logging_directory, 
                )

        logger.info('successfully completed CallbackSaveLastModel.__call__')
