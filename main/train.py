if __name__ == '__main__':
    import sys
    sys.path.append('src')
    import torch
    torch.cuda.empty_cache()
    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy('file_system')

    import os
    import traceback
    from argparse import Namespace
    from functools import partial 

    from transforms import TransformRandomFlip, criteria_is_array_n_or_above, TransformSqueeze, TransformUnsqueeze
    from datasets import create_datasets_reservoir_map
    from startup_utilities import default, configure_startup, read_splits
    from src_transforms import transform_feature_3d_v2
    from transforms_augmentation_affine import transform_augmentation_random_affine_transform
    from transforms_clean_features import TransformCleanFeatures
    from segmentation import run_trainer, load_model
    from segmentation import load_case_hdf5, load_case_hdf5_random128_m3200
    from segmentation import PreprocessDataV4_lung_soft_tissues_hot as Preprocessing
    from segmentation.model_unet_multiclass_deepsupervision_configured_v1 import SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_boundary_sensitivity as Model
    from data import root_datasets, root_logging
    from segmentation import TransformGeneric, train_suv_augmentations
    import numpy as np
    from collate import default_collate_fn
    from utils.graceful_killer import GracefulKiller
    from optimizers_v2 import OptimizerAdamW 


    configuration = {
        'datasets': {
            'data_root': default('data_root', root_datasets),
            'datasets_version': 'v1',
            'splits_name': default('splits_name', 'config.json'),
        },
        'training': {
            'device': default('device', 'cuda:0'),
            # 'device': default('device', 'cpu'),
            'batch_size': 2,
            'data_loading_workers': default('data_loading_workers', 2, output_type=int),
            'map_workers': default('map_workers', 3, output_type=int),
            'nb_epochs': default('nb_epochs', 2, output_type=int),
            'learning_rate': 1e-3,
            'weight_decay': 1e-6,
            'eval_every_X_epoch': default('eval_every_X_epoch', 200, output_type=int),
            'eval_inference_every_X_epoch': default('eval_inference_every_X_epoch', 200, output_type=int),
            'run_name': default('run_name', os.path.splitext(os.path.basename(__file__))[0]),
            'logging_directory': default('logging_directory', root_logging),
            'vscode_batch_size_reduction_factor': 1,
            'gradient_update_frequency': 1,
            '3d_based_model': True,
            'mixed_precision_enabled': True,
        },
        'data': {
            'fov_half_size': np.asarray((64, 48, 48)),
            'samples_per_patient': 1,
            'preprocessing': Preprocessing(),
            'load_case_train': load_case_hdf5_random128_m3200,
            'load_case_valid': load_case_hdf5,
            'config_start': None,
        },
        'tracking': {
            'derived_from': 'hdf5 preprocessing',
            'info': 'Extra sensitivity loss is used'
        }
    }
    configuration = Namespace(**configuration)


    def main():
        configure_startup(configuration)

        config_start = configuration.data.get('config_start')

        criteria_fn = partial(criteria_is_array_n_or_above, dim=3)

        features_fn_train = partial(transform_feature_3d_v2, configuration=configuration, sample_volume_name='ct', only_valid_z=True, nb_samples=configuration.data['samples_per_patient'])

        features_fn_valid = partial(transform_feature_3d_v2, configuration=configuration, sample_volume_name='ct', only_valid_z=True, nb_samples=4)

        datasets = create_datasets_reservoir_map(
            read_splits(configuration), 
            configuration=configuration,
            load_case_fn=None,
            load_case_train_fn=configuration.data['load_case_train'],
            load_case_valid_fn=configuration.data['load_case_valid'], 
            preprocess_data_train_fn=[
                TransformGeneric(train_suv_augmentations(), 'suv'),
                configuration.data['preprocessing'],
                TransformUnsqueeze(axis=0, criteria_fn=criteria_fn),
                TransformRandomFlip(axis=1),
                TransformRandomFlip(axis=2),
                TransformRandomFlip(axis=3), 
                TransformSqueeze(axis=0),
                transform_augmentation_random_affine_transform,
                TransformCleanFeatures(['bounding_boxes_min_max']) # incompatible type (#BB different for each case!)
            ],
            preprocess_data_test_fn=[
                configuration.data['preprocessing'],
                TransformCleanFeatures(['bounding_boxes_min_max']) # incompatible type (#BB different for each case!)
            ],
            transform_train=[
                partial(default_collate_fn, device=None),
                features_fn_train,
                ],
            transform_test=[
                features_fn_valid,     
            ],
            max_reservoir_samples=96 * 2,  # nb_samples % (samples_per_patient * batch_size) should be 0 for optimality
            min_reservoir_samples=48,
            nb_map_workers=configuration.training['map_workers'],
            nb_reservoir_workers=configuration.training['data_loading_workers'],
            max_reservoir_jobs_at_once_factor=40
        )

        try:                    
            optimizer = OptimizerAdamW(
                learning_rate=configuration.training['learning_rate'], 
                weight_decay=configuration.training['weight_decay']).scheduler_cosine_annealing_warm_restart_decayed(
                    T_0=configuration.training['eval_inference_every_X_epoch'],
                    decay_factor=0.9
            ).clip_gradient_norm()

            model = Model()
            if config_start is not None:
                load_model(model, config_start, device=torch.device(configuration.training['device'].split(';')[0]), strict=True)
            run_trainer(configuration, datasets, model, optimizer)
        except Exception as e:
            print(f'Exception caught={e}')
            print('-------------- Stacktrace --------------')
            traceback.print_exc()
            print('----------------------------------------')
        
        del datasets
        print('Datasets deleted!')

    main()
    print('Last step is done!')

    killer = GracefulKiller()
    killer.exit_gracefully(None, None)