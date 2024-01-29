from functools import partial
import os
from startup_utilities import was_started_within_vscode
from callback_inference import CallbackInference, find_root_sequence
from corelib import inference_process_wholebody_3d, test_time_inference

from segmentation.metrics import calculate_metrics_autopet
from .callback_report import CallbackWriteReport
from callbacks import ModelWithLowestMetric
import torch
from callbacks.callback_save_last_model import ModelWithLowestMetricBase
from callbacks.callback_learning_rate_recorder import CallbackLearningRateRecorder
from callbacks.callback_epoch_summary import CallbackEpochSummary
from callbacks.callback_reporting_epoch_summary import CallbackReportingRecordHistory
from callbacks.callback_reporting_best_metrics import CallbackReportingBestMetrics
from callbacks.callback_reporting_learning_rate_recorder import CallbackReportingLearningRateRecorder
from callbacks.callback_skip_epoch import CallbackSkipEpoch
from callbacks.callback_save_last_model import CallbackSaveLastModel
from callbacks.callback_reporting_export_samples import CallbackReportingExportSamples
from callbacks.callback_export_history import CallbackExportHistory

def get_output_fn(outputs):
    return torch.softmax(outputs['seg'].output, dim=1)

def get_output_features_fn(outputs):
    return outputs['features'].output

def create_inference(configuration, fov_half_size=None, tile_step=None, test_time_augmentation_axis=None, get_output_fn=get_output_fn, nb_outputs=2, postprocessing_fn = partial(torch.argmax, dim=1), no_output_ref_collection=False, internal_type=torch.float32):
    if configuration is not None:
        sequence_model = configuration.training.get('sequence_model')
        if sequence_model is not None:
            inference_sequence = partial(inference_process_wholebody_3d,
                feature_names=('suv', 'seg', 'sequence_label', 'sequence_input', 'sequence_output'),
                output_truth_name='seg',
                multiple_of=None,
                main_input_name='suv',
                tiling_strategy='none',  # classification all at once!
                postprocessing_fn=partial(torch.argmax, dim=1),
                get_output=get_output_fn,
                nb_outputs=nb_outputs,
                no_output_ref_collection=no_output_ref_collection,
                internal_type=internal_type,
            )
            return inference_sequence


    if fov_half_size is None:
        fov_half_size = configuration.data.get('fov_half_size')

    if fov_half_size is None:
        # this is not a windowing based inference!
        # TODO To be handled (e.g., sequence model)
        return None

    if tile_step is None:
        tile_step = fov_half_size

    inference_3d = partial(inference_process_wholebody_3d,
        feature_names=('ct', 'ct_lung', 'suv', 'seg', 'ct_soft', 'suv_hot', 'cascade.inference.output_found', 'z_coords', 'y_coords', 'x_coords'),
        output_truth_name='seg',
        main_input_name='suv',

        tile_shape=fov_half_size * 2,
        tile_step=tile_step,
        tile_margin=0,
        multiple_of=fov_half_size * 2,

        tile_weight='weighted_central',
        tiling_strategy='tiled_3d',
        postprocessing_fn=partial(torch.argmax, dim=1),
        get_output=get_output_fn,
        nb_outputs=nb_outputs,
        invalid_indices_value=0.0,  # default to `no segmentation`
        no_output_ref_collection=no_output_ref_collection,
        internal_type=internal_type,
    )

    if test_time_augmentation_axis is None:
        test_time_augmentation_axis = configuration.training.get('test_time_augmentation_axis')

    if test_time_augmentation_axis:
        def flip_batch(batch, axis):
            new_batch = {}
            discard_features = ('bounding_boxes_min_max',)
            for name, value in batch.items():
                if isinstance(value, torch.Tensor) and len(value.shape) == 3:
                    # TODO: performance penalty: the full image is copied
                    # multiple times (one time per axis per augmentation)
                    new_batch[name] = torch.flip(value, [axis])
                elif isinstance(value, torch.Tensor) and len(value.shape) == 4 and name not in discard_features:
                    # this is for the probability map. Since we have
                    # an additional `C` component, the axis to be flipped
                    # is the next one
                    assert value.shape[0] == 2, f'name={name}, shape={value.shape}'
                    new_batch[name] = torch.flip(value, [axis + 1])
                else:
                    new_batch[name] = value
            return new_batch

        transforms = [
            partial(flip_batch, axis=0),
            partial(flip_batch, axis=1),
            partial(flip_batch, axis=2),
        ]

        # the inverse of axis flip is the same axis flip
        transforms_inv = [
            partial(flip_batch, axis=0),
            partial(flip_batch, axis=1),
            partial(flip_batch, axis=2),
        ]

        tta_fn = partial(test_time_inference, 
            inference_fn=inference_3d, 
            transforms=transforms, 
            transforms_inv=transforms_inv
        )
        inference_3d = tta_fn
    return inference_3d


def create_inference_callback(configuration, skip_train=True, inference_kwargs={}, output_dir_name='wholebody_inference'):
    export_inference_prob = configuration.training.get('export_inference_prob')
    export_3d_volumes = configuration.training.get('inference_export_3d_volumes')
    if export_3d_volumes is None:
        # default to export
        export_3d_volumes = True

    def get_data_fn(sequence):
        root_seq = find_root_sequence(sequence)
        load_case_fn = configuration.data.get('load_case')
        if load_case_fn is None:
            load_case_fn = configuration.data.get('load_case_valid')
        assert load_case_fn is not None, '`load_case_valid` or `load_case` must be defined in `configuration.data`'
        preprocessing = configuration.data['preprocessing']
        return root_seq.map([partial(load_case_fn, configuration=configuration), preprocessing])

    return CallbackInference(
        inference_fn=create_inference(configuration, **inference_kwargs), 
        get_data_fn=get_data_fn, 
        metrics_fn=calculate_metrics_autopet, 
        max_value_mip=0.3,  # PET values are normalized if `max_value_name` not in the batch 
        max_value_name='suv_display_target',
        result_path='projects/segmentation/experiments/baseline/results',
        mip_output_scaling=100.0,
        skip_train=skip_train,
        export_inference_prob=export_inference_prob,
        export_3d_volumes=export_3d_volumes,
        output_dir_name=output_dir_name
    )


def create_inference_features_callback(configuration, skip_train=True, nb_outputs=32):
    inference_kwargs={
        'get_output_fn': get_output_features_fn,
        'nb_outputs': nb_outputs,
        # return the raw value of the feature map!
        'postprocessing_fn': lambda x: x
    }

    callback = create_inference_callback(
        configuration, 
        skip_train=skip_train, 
        inference_kwargs=inference_kwargs, output_dir_name='wholebody_inference_features'
    )
    return callback


def default_pre_training_callbacks(configuration, model):
    callbacks = []

    if was_started_within_vscode():
        callbacks += [
            # only start the reporting server when debug mode (i.e.,  started from VSCode)
            callbacks.CallbackReportingStartServer(),

            #create_inference_features_callback(configuration),  # TODO REMOVE
            #create_inference_callback(configuration),  # TODO REMOVE
            #CallbackWriteReport(),  # TODO REMOVE
            #callbacks.CallbackSaveLastModel(keep_model_with_lowest_metric=ModelWithLowestMetric(
            #    dataset_name='auto_pet', split_name='test', output_name='seg', metric_name='1-dice[class=1]', lowest_metric=0.0)),

        ]

    callbacks += [
        #CallbackExperimentTracking(),
        #callbacks.CallbackReportingModelSummary(),
        #callbacks.CallbackReportingDatasetSummary(),
        #callbacks.CallbackReportingExportSamples(table_name='random_samples'),
        #callbacks.CallbackProfiler(),
        #callbacks.CallbackReportingAugmentations(),
        #callbacks.CallbackLearningRateFinder()
    ]

    return callbacks


class ModelWithHighestMetric(ModelWithLowestMetricBase):
    def __init__(self, dataset_name, split_name, output_name, metric_name, minimum_metric=0.74):
        """

        Args:
            dataset_name: the dataset name to be considered for the best model
            split_name: the split name to be considered for the best model
            metric_name: the metric name to be considered for the best model
            minimum_metric: consider only the metric higher than this threshold
            output_name: the output to be considered for the best model selection
        """
        self.output_name = output_name
        self.metric_name = metric_name
        self.split_name = split_name
        self.dataset_name = dataset_name
        self.minimum_metric = minimum_metric
        self.best_highest_metric = -1e10

    def update(self, metric_value, model, metadata, root_path):
        if metric_value is not None and metric_value >= self.minimum_metric and metric_value >= self.best_highest_metric:
            self.best_highest_metric = metric_value
            export_path = os.path.join(
                root_path,
                f'best_{self.metric_name}_e{len(metadata.history)}_{metric_value}.model')
            from trainer_v2 import save_model
            save_model(model, metadata, export_path)


def default_per_epoch_callbacks(configuration, model):
    #debug_mode = was_started_within_vscode()
    debug_mode = False

    eval_inference_every_X_epoch = configuration.training.get('eval_inference_every_X_epoch')
    if eval_inference_every_X_epoch is None:
        eval_inference_every_X_epoch = configuration.training['eval_every_X_epoch']

    rolling_checkpoints = configuration.training.get('rolling_checkpoints')
    callbacks = [
        CallbackLearningRateRecorder(),
        CallbackEpochSummary(),
        CallbackReportingRecordHistory(),
        CallbackReportingBestMetrics(),
        CallbackReportingLearningRateRecorder(),
        CallbackSkipEpoch(
            nb_epochs=eval_inference_every_X_epoch,
            include_epoch_zero=debug_mode,
            callbacks=[
                create_inference_callback(configuration),
                CallbackSaveLastModel(keep_model_with_best_metric=ModelWithHighestMetric(
                    dataset_name='auto_pet', split_name='test', output_name='autopet', metric_name='dice_foreground', minimum_metric=0.62)),
                CallbackSaveLastModel(keep_model_with_best_metric=ModelWithHighestMetric(
                    dataset_name='auto_pet', split_name='test', output_name='autopet', metric_name='dice', minimum_metric=0.49)),
            ]),
    ]

    if rolling_checkpoints is not None:
        callback = CallbackSkipEpoch(
            nb_epochs=configuration.training['eval_every_X_epoch'],
            include_epoch_zero=debug_mode,
            callbacks=[
               CallbackSaveLastModel(
                    model_name='rolling_save', 
                    rolling_size=rolling_checkpoints, 
                    is_versioned=True
                )
        ])
        callbacks.append(callback)

    return callbacks


def default_post_training_callbacks(configuration, model):

    inference_discard_train = configuration.training.get('inference_discard_train')
    if inference_discard_train is None:
        inference_discard_train = True

    export_inference_features = configuration.training.get('export_inference_features')

    callbacks = []
    if export_inference_features:
        callbacks.append(
            create_inference_features_callback(configuration, skip_train=inference_discard_train),
        )

    callbacks += [
        CallbackSaveLastModel(),
        create_inference_callback(configuration, skip_train=inference_discard_train),
        CallbackReportingExportSamples(),
        CallbackExportHistory(),
        CallbackWriteReport(),
    ]

    return callbacks