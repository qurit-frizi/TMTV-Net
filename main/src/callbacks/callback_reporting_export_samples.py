import os
import functools
import logging

import torch
from export import export_sample
from flatten import flatten
import collections


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

from table_sqlite import table_truncate, TableStream
from .callback import Callback
import utilities
import outputs
from utilities import update_json_config, create_or_recreate_folder

logger = logging.getLogger(__name__)


def expand_classification_mapping(batch, loss_term_name, loss_term, classification_mappings, suffix='_str'):
    """
    Expand as string the class name for the classification outputs

    Args:
        batch:
        loss_term:
        classification_mappings: classification_mappings: a nested dict recording the class name/value
            associated with a set of ``output_name``

            {``output_name``:
                {'mapping': {name, value}},
                {'mappinginv': {value, name}}
            }

        suffix: the suffix appended to output or target name
    """
    output_ref = loss_term.get('output_ref')
    if isinstance(output_ref, outputs.OutputClassification):
        target_name = output_ref.classes_name
        if target_name is not None and classification_mappings is not None:
            mapping = classification_mappings.get(target_name)
            if mapping is not None:
                output = to_value(loss_term['output'])
                if len(output.shape) == 1:
                    output_str = [utilities.get_class_name(mapping, o) for o in output]
                    batch[loss_term_name + suffix] = output_str

                    # if we record the loss term output, record also the
                    # target name as string.
                    target_name_str = target_name + suffix
                    if target_name_str not in batch:
                        target_values = batch.get(target_name)
                        if target_values is not None and len(target_values.shape) == 1:
                            target_values = [utilities.get_class_name(mapping, o) for o in target_values]
                            batch[target_name_str] = target_values


def select_all(batch, loss_terms):
    nb_samples = len_batch(batch)
    return range(nb_samples)


def callbacks_per_loss_term(
        dataset_name,
        split_name,
        batch,
        loss_terms,
        root,
        datasets_infos,
        loss_terms_inclusion,
        feature_exclusions,
        dataset_exclusions,
        split_exclusions,
        exported_cases,
        max_samples,
        epoch,
        sql_table,
        format,
        select_fn,
        **kwargs):
    # process the exclusion
    if dataset_name in dataset_exclusions:
        raise StopIteration()
    if split_name in split_exclusions:
        raise StopIteration()

    # copy to the current batch the specified loss terms
    classification_mappings = utilities.get_classification_mappings(datasets_infos, dataset_name, split_name)
    for loss_term_name, loss_term in loss_terms.items():
        for loss_term_inclusion in loss_terms_inclusion:
            if loss_term_inclusion in loss_term:
                name = f'term_{loss_term_name}_{loss_term_inclusion}'
                value = loss_term[loss_term_inclusion]
                batch[name] = to_value(value)

                # special handling of `losses`: in 2D regression, the output will be a 2D error maps
                # but it could be useful to have the average error instead (e.g., to plot the worst samples).
                if loss_term_inclusion == 'losses' and len(value.shape) > 2:
                    batch[name + '_avg'] = to_value(torch.mean(flatten(value), dim=1))
                expand_classification_mapping(batch, loss_term_name, loss_term, classification_mappings)

    for feature_exclusion in feature_exclusions:
        if feature_exclusion in batch:
            del batch[feature_exclusion]

    # force recording of epoch
    batch['epoch'] = epoch

    # calculate how many samples to export
    nb_batch_samples = len_batch(batch)
    nb_samples_exported = len(exported_cases)
    nb_samples_to_export = min(max_samples - nb_samples_exported, nb_batch_samples)
    if nb_samples_to_export <= 0:
        raise StopIteration()

    # export the features
    samples_to_export = select_fn(batch, loss_terms)
    samples_to_export = samples_to_export[:nb_samples_to_export]
    for n in samples_to_export:
        id = n + nb_samples_exported
        exported_cases.append(id)
        name = format.format(dataset_name=dataset_name, split_name=split_name, id=id, epoch=epoch)
        export_sample(
            root,
            sql_table,
            base_name=name,
            batch=batch,
            sample_ids=[n],
            name_expansions=[],  # we already expanded in the basename!
        )


class CallbackReportingExportSamples(Callback):
    def __init__(
            self,
            max_samples=50,
            table_name='samples',
            loss_terms_inclusion=None,
            feature_exclusions=None,
            dataset_exclusions=None,
            split_exclusions=None,
            clear_previously_exported_samples=True,
            format='{dataset_name}_{split_name}_s{id}_e{epoch}',
            reporting_config_keep_last_n_rows=None,
            reporting_config_subsampling_factor=1.0,
            reporting_scatter_x='split_name',
            reporting_scatter_y='dataset_name',
            reporting_color_by=None,
            reporting_display_with=None,
            reporting_binning_x_axis=None,
            reporting_binning_selection=None,
            select_sample_to_export=select_all
    ):
        """
        Export random samples from our datasets.

        Args:
            max_samples: the maximum number of samples to be exported (per dataset and per split)
            table_name: the root of the export directory
            loss_terms_inclusion: specifies what output name from each loss term will be exported.
                if None, defaults to ['output']
            feature_exclusions: specifies what feature should be excluded from the export
            dataset_exclusions: specifies what dataset should be excluded from the export
            split_exclusions: specifies what split should be excluded from the export
            format: the format of the files exported. Sometimes need evolution by epoch, other time we may want
                samples by epoch so make this configurable
            reporting_config_keep_last_n_rows: Only visualize the last ``reporting_config_keep_last_n_rows``
                samples. Prior samples are discarded. This parameter will be added to the reporting configuration.
            reporting_config_subsampling_factor: Specifies how the data is sub-sampled. Must be in range [0..1]
                or ``None``. This parameter will be added to the reporting configuration.
            select_sample_to_export: a function taking a ``(batch, loss_terms)`` and returning a list of indices of the
                samples to be exported
            clear_previously_exported_samples: if ``True``, the table will be emptied before each sample export
        """

        self.format = format
        self.max_samples = max_samples
        self.table_name = table_name
        if loss_terms_inclusion is None:
            self.loss_terms_inclusion = ['output', 'output_raw', 'losses']
        else:
            self.loss_terms_inclusion = loss_terms_inclusion

        if feature_exclusions is not None:
            self.feature_exclusions = feature_exclusions
        else:
            self.feature_exclusions = []

        if dataset_exclusions is not None:
            self.dataset_exclusions = dataset_exclusions
        else:
            self.dataset_exclusions = []

        if split_exclusions is not None:
            self.split_exclusions = split_exclusions
        else:
            self.split_exclusions = []

        # record the viewing configuration
        self.reporting_config_exported = False
        self.reporting_config_keep_last_n_rows = reporting_config_keep_last_n_rows
        self.reporting_config_subsampling_factor = reporting_config_subsampling_factor
        self.reporting_scatter_x = reporting_scatter_x
        self.reporting_scatter_y = reporting_scatter_y
        self.reporting_color_by = reporting_color_by
        self.reporting_display_with = reporting_display_with
        self.reporting_binning_x_axis = reporting_binning_x_axis
        self.reporting_binning_selection = reporting_binning_selection

        self.select_sample_to_export = select_sample_to_export
        self.clear_previously_exported_samples = clear_previously_exported_samples

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch,
                 **kwargs):

        logger.info('started CallbackExportSamples.__call__')
        device = options.workflow_options.device

        if not self.reporting_config_exported:
            # export how the samples should be displayed by the reporting
            config_path = options.workflow_options.sql_database_view_path
            update_json_config(config_path, {
                self.table_name: {
                    'data': {
                        'keep_last_n_rows': self. reporting_config_keep_last_n_rows,
                        'subsampling_factor': self.reporting_config_subsampling_factor,
                    },

                    'default': {
                        'Scatter X Axis': self.reporting_scatter_x,
                        'Scatter Y Axis': self.reporting_scatter_y,
                        'Color by': self.reporting_color_by,
                        'Display with': self.reporting_display_with,
                        'Binning X Axis': self.reporting_binning_x_axis,
                        'Binning selection': self.reporting_binning_selection,
                    }
                }
            })
            self.reporting_config_exported = True

        sql_database = options.workflow_options.sql_database
        if self.clear_previously_exported_samples:
            cursor = sql_database.cursor()
            table_truncate(cursor, self.table_name)
            sql_database.commit()

            # also remove the binary/image store
            root = os.path.dirname(options.workflow_options.sql_database_path)
            create_or_recreate_folder(os.path.join(root, 'static', self.table_name))

        sql_table = TableStream(
            cursor=sql_database.cursor(),
            table_name=self.table_name,
            table_role='data_samples')

        from trainer import eval_loop
        logger.info(f'export started..., N={self.max_samples}')
        for dataset_name, dataset in datasets.items():
            root = os.path.join(options.workflow_options.current_logging_directory, 'static', self.table_name)
            if not os.path.exists(root):
                utilities.create_or_recreate_folder(root)

            for split_name, split in dataset.items():
                exported_cases = []
                eval_loop(options, device, dataset_name, split_name, split, model, losses[dataset_name],
                          history=None,
                          callbacks_per_batch=callbacks_per_batch,
                          callbacks_per_batch_loss_terms=[
                              functools.partial(
                                  callbacks_per_loss_term,
                                  root=options.workflow_options.current_logging_directory,
                                  datasets_infos=datasets_infos,
                                  loss_terms_inclusion=self.loss_terms_inclusion,
                                  feature_exclusions=self.feature_exclusions,
                                  dataset_exclusions=self.dataset_exclusions,
                                  split_exclusions=self.split_exclusions,
                                  exported_cases=exported_cases,
                                  max_samples=self.max_samples,
                                  epoch=len(history),
                                  sql_table=sql_table,
                                  format=self.format,
                                  select_fn=self.select_sample_to_export
                              )])

        sql_database.commit()
        logger.info('successfully completed CallbackExportSamples.__call__!')
