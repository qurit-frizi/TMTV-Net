import collections
import logging
import numpy as np

from .callback_reporting_export_samples import CallbackReportingExportSamples

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
from .callback import Callback
import outputs

logger = logging.getLogger(__name__)


def select_classification_errors(batch, loss_terms):
    nb_samples = len_batch(batch)
    indices_errors = collections.defaultdict(list)
    for name, loss_term in loss_terms.items():
        ref = loss_term.get('output_ref')
        if ref is None or not isinstance(ref, outputs.OutputClassification):
            continue

        truth_values = to_value(loss_term['output_truth'])
        found_values = to_value(loss_term['output'])
        samples_with_errors = np.where(found_values != truth_values)[0]
        for i in samples_with_errors:
            indices_errors[i].append(name)

    samples = []
    samples_error = [''] * nb_samples
    for index, values in indices_errors.items():
        samples.append(index)
        samples_error[index] = '|'.join(values)

    # add additional data in the batch so that we can easily display the errors
    batch['samples_error'] = samples_error
    return samples


class CallbackReportingClassificationErrors(Callback):
    def __init__(
            self,
            max_samples=10,
            table_name='errors',
            loss_terms_inclusion=None,
            feature_exclusions=None,
            dataset_exclusions=None,
            split_exclusions=None,
            clear_previously_exported_samples=True,
            format='{dataset_name}_{split_name}_s{id}_e{epoch}',
            reporting_config_keep_last_n_rows=None,
            reporting_config_subsampling_factor=1.0):
        """
        Export samples with classification errors

        Args:
            max_samples: the maximum number of samples to be exported (per dataset and per split)
            table_name: the root of the export directory
            loss_terms_inclusion: specifies what output name from each loss term will be exported. if None, defaults to ['output']
            feature_exclusions: specifies what feature should be excluded from the export
            dataset_exclusions: specifies what dataset should be excluded from the export
            split_exclusions: specifies what split should be excluded from the export
            format: the format of the files exported. Sometimes need evolution by epoch, other time we may want
                samples by epoch so make this configurable
            reporting_config_keep_last_n_rows: Only visualize the last ``reporting_config_keep_last_n_rows``
                samples. Prior samples are discarded. This parameter will be added to the reporting configuration.
            reporting_config_subsampling_factor: Specifies how the data is sub-sampled. Must be in range [0..1]
                or ``None``. This parameter will be added to the reporting configuration.
            clear_previously_exported_samples: if ``True``, the table will be emptied before each sample export
        """

        # this callback is a simple re-parameterization of the export samples
        self.impl = CallbackReportingExportSamples(
            max_samples=max_samples,
            table_name=table_name,
            loss_terms_inclusion=loss_terms_inclusion,
            feature_exclusions=feature_exclusions,
            dataset_exclusions=dataset_exclusions,
            split_exclusions=split_exclusions,
            format=format,
            reporting_config_keep_last_n_rows=reporting_config_keep_last_n_rows,
            reporting_config_subsampling_factor=reporting_config_subsampling_factor,
            select_sample_to_export=select_classification_errors,
            clear_previously_exported_samples=clear_previously_exported_samples,
        )

    def __call__(
            self,
            options,
            history,
            model,
            losses,
            outputs,
            datasets,
            datasets_infos,
            callbacks_per_batch,
            **kwargs):

        self.impl(
            options,
            history,
            model,
            losses,
            outputs,
            datasets,
            datasets_infos,
            callbacks_per_batch,
            **kwargs)
