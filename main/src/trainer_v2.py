import logging
import os
import pickle
import sqlite3
import traceback
from io import StringIO
from typing import Tuple, Optional, Any, Dict

from torch import nn

import torch

from utilities import RunMetadata
from utilities import default_sum_all_losses, create_or_recreate_folder, RuntimeFormatter
from trainer import default_per_epoch_callbacks, default_pre_training_callbacks, \
    default_post_training_callbacks, trainer_callbacks_per_batch, epoch_train_eval, \
    create_losses_fn, strip_unpickable


class ExceptionAbortRun(BaseException):
    """
    The run has been pruned due to performance reason
    """
    def __init__(self, history, metrics=None, reason=None):
        self.reason = reason
        self.history = history
        self.metrics = metrics

    def __str__(self):
        return f'ExceptionAbortRun(reason={self.reason})'

from graceful_killer import GracefulKiller
from load_module import find_global_name

logger = logging.getLogger(__name__)


class TrainerV2:
    def __init__(
            self,
            callbacks_per_batch=None,
            callbacks_per_batch_loss_terms=None,
            callbacks_per_epoch=default_per_epoch_callbacks(),
            callbacks_pre_training=default_pre_training_callbacks(),
            callbacks_post_training=default_post_training_callbacks(),
            trainer_callbacks_per_batch=trainer_callbacks_per_batch,
            run_epoch_fn=epoch_train_eval,
            logging_level=logging.DEBUG,
            skip_eval_epoch_0=True):
        """

        Args:
            callbacks_per_batch:
            callbacks_per_batch_loss_terms:
            callbacks_per_epoch:
            callbacks_pre_training:
            callbacks_post_training:
            trainer_callbacks_per_batch:
            run_epoch_fn:
            skip_eval_epoch_0: if ``True``, validation/test will not be run for epoch 0
        """
        self.callbacks_per_batch = callbacks_per_batch
        self.callbacks_per_epoch = callbacks_per_epoch
        self.callbacks_pre_training = callbacks_pre_training
        self.callbacks_post_training = callbacks_post_training
        self.callbacks_per_batch_loss_terms = callbacks_per_batch_loss_terms
        self.trainer_callbacks_per_batch = trainer_callbacks_per_batch
        self.run_epoch_fn = run_epoch_fn
        self.skip_eval_epoch_0 = skip_eval_epoch_0
        self.logging_level = logging_level

    @staticmethod
    def save_model(model, metadata: RunMetadata, path, pickle_module=pickle):
        """
        Save a model to file

        Args:
            model: the model to serialize
            metadata: an optional result file associated with the model
            path: the base path to save the model
            pickle_module: the serialization module that will be used to save the model and results

        """
        sql_database = None
        if metadata is not None:
            import copy
            if metadata.options is not None:
                # we don't want this function to have side effects so copy
                # the result
                sql_database = metadata.options.workflow_options.sql_database

                # strip what can't be pickled
                if sql_database is not None:
                    metadata.options.workflow_options.sql_database = None

            metadata_cp = copy.copy(metadata)
            if metadata_cp.outputs is not None:
                metadata_cp.outputs = strip_unpickable(metadata_cp.outputs)
        else:
            # we MUST have at least the class name!
            metadata_cp = RunMetadata(options=None, history=None, outputs=None)

        # record the original fully qualified class name so that we can re-instantiate it
        metadata_cp.class_name = str(model.__class__.__name__)
        module = model.__class__.__module__
        if len(module) > 0:
            metadata_cp.class_name = module + '.' + metadata_cp.class_name

        metadata_cp_path = path + '.metadata'
        with open(metadata_cp_path, 'wb') as f:
            pickle_module.dump(metadata_cp, f)
        torch.save(model.state_dict(), path, pickle_module=pickle_module)

        if sql_database is not None and metadata.options is not None:
            # TODO find a cleaner and generic way of doing this...
            metadata.options.workflow_options.sql_database = sql_database

    @staticmethod
    def load_state(
            model: nn.Module,
            path: str,
            device: torch.device = None,
            pickle_module: Any = pickle,
            strict: bool = True) -> None:
        """
        Load the state of a model

        Args:
            model: where to load the state
            path: where the model's state was saved
            device: where to locate the model
            pickle_module: how to read the model parameters and metadata
            strict: whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function
        """
        model_state = torch.load(path, map_location=device, pickle_module=pickle_module)
        model.load_state_dict(model_state, strict=strict)

    @staticmethod
    def load_model(
            path: str,
            model_kwargs: Optional[Dict[Any, Any]] = None,
            with_result: bool = False,
            device: torch.device = None,
            pickle_module: Any = pickle) -> Tuple[nn.Module, RunMetadata]:
        """
        Load a previously saved model

        Construct a model from the :attr:`RunMetadata.class_name` class and with arguments :obj:`model_kwargs`

        Args:
            path: where to store the model. result's will be loaded from `path + '.result'`
            model_kwargs: arguments used to instantiate the model stored in :attr:`RunMetadata.class_name`
            with_result: if True, the results of the model will be loaded
            device: where to load the model. For example, models are typically trained on GPU,
                but for deployment, CPU might be good enough. If `None`, use the same device as
                when the model was exported
            pickle_module: the de-serialization module to be used to load model and results

        Returns:
            a tuple `model, metadata`
        """
        assert os.path.exists(path), f'model={path} could not be found!'

        result_path = path + '.metadata'
        with open(result_path, 'rb') as f:
            metadata = pickle_module.load(f)
        if not with_result:
            metadata.outputs = None

        class_name = metadata.class_name
        class_type = find_global_name(class_name)
        if model_kwargs is None:
            model_kwargs = {}
        model = class_type(**model_kwargs)

        TrainerV2.load_state(model, path, device=device, pickle_module=pickle_module)
        return model, metadata

    def fit(self,
            options,
            datasets,
            model: nn.Module,
            optimizers_fn,
            losses_fn=default_sum_all_losses,
            loss_creator=create_losses_fn,
            log_path=None,
            with_final_evaluation=True,
            history=None,
            erase_logging_folder=True,
            eval_every_X_epoch=1) -> RunMetadata:
        """
        Fit the model

        Args:
            options:
            datasets:  a functor returning a dictionary of datasets. Alternatively, datasets infos can be specified.
                        `inputs_fn` must return one of:

                        * datasets: dictionary of dataset
                        * (datasets, datasets_infos): dictionary of dataset and additional infos

                        We define:

                        * datasets: a dictionary of dataset. a dataset is a dictionary of splits.
                          a split is a dictionary of batched features.
                        * Datasets infos are additional infos useful for the debugging of the
                          dataset (e.g., class mappings, sample UIDs). Datasets infos are
                          typically much smaller than datasets should be loaded in
                          loadable in memory
            model: a `Module` or a `ModuleDict`
            optimizers_fn:
            losses_fn:
            loss_creator:
            log_path: the path of the logs to be exported during the training of the model.
                if the `log_path` is not an absolute path, the options.workflow_options.logging_directory
                is used as root
            with_final_evaluation:
            history:
            erase_logging_folder: if `True`, the logging will be erased when fitting starts
            eval_every_X_epoch: evaluate the model every `X` epochs

        Returns:
       """

        # reset the abort state
        GracefulKiller.abort_event.clear()

        # set up our log path. This is where all the analysis of the model will be exported
        logging_directory = options.workflow_options.logging_directory
        if log_path is None:
            log_path = os.path.join(
                logging_directory,
                'default_r{}'.format(options.workflow_options.trainer_run))
        elif not os.path.isabs(log_path):
            log_path = os.path.join(logging_directory, log_path)

        options.workflow_options.current_logging_directory = log_path

        if history is None:
            # no prior history
            history = []

        # now clear our log path to remove previous files if needed
        if erase_logging_folder:
            create_or_recreate_folder(log_path)
        elif not os.path.exists(log_path):
            os.makedirs(log_path)

        if len(logging.root.handlers) == 0:
            # there is no logger configured, so add a basic one
            logging.basicConfig(
                filename=os.path.join(logging_directory, 'trainer_logging.log'),
                format='%(asctime)s %(levelname)s %(name)s %(message)s',
                level=self.logging_level,
                filemode='w')
        else:
            logging.root.setLevel(self.logging_level)

        # create the reporting SQL database
        sql_path = os.path.join(log_path, 'reporting_sqlite.db')
        sql = sqlite3.connect(sql_path)
        options.workflow_options.sql_database = sql
        options.workflow_options.sql_database_path = sql_path
        options.workflow_options.sql_database_view_path = sql_path.replace('.db', '.json')

        # here we want to have our logging per training run, so add a handler
        handler = logging.FileHandler(os.path.join(log_path, 'trainer.log'))
        formatter = RuntimeFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        handler.setFormatter(formatter)
        handler.setLevel(self.logging_level)
        logging.root.addHandler(handler)

        # instantiate the datasets, model, optimizers and losses
        logger.info('started Trainer.fit(). Options={}'.format(options))

        def clean_up(datasets):
            if datasets is not None:
                # make sure the datasets are closed properly: threads and processes
                # are stopped in a controlled manner to avoid memory leaks
                for dataset_name, dataset in datasets.items():
                    for split_name, split in dataset.items():
                        if hasattr(split, 'close'):
                            logger.info(f'closing dataset={dataset_name} split={split_name}')
                            split.close()
                            logger.info(f'closed dataset={dataset_name} split={split_name}!')

                # resource are released, just continue the shutdown
                logger.info(f'datasets all closed!')

            # increment the number of runs
            options.workflow_options.trainer_run += 1

            logger.info('removing logging handlers...')
            logging.root.removeHandler(handler)

            logger.info('training completed!')

            sql.commit()
            sql.close()

        datasets_infos = None  # TODO REFACTOR THIS
        assert datasets is not None, '`datasets` is None!'
        if isinstance(datasets, tuple):
            if len(datasets) == 2:
                logger.info('inputs_fn specified `datasets, datasets_infos`')
                datasets, datasets_infos = datasets
            else:
                assert 0, 'expected tuple `datasets` or `datasets, datasets_infos`'

        assert isinstance(model, torch.nn.Module), f'the model MUST be a `torch.nn.Module`, got={type(model)}'
        if isinstance(model, torch.nn.ModuleDict):
            # if we have sub-models, we MUST define a `forward` method
            # to orchestrate the calls of sub-models
            assert 'forward' in dir(model)

        outputs_epoch = None
        try:
            # migrate the model to the specified device
            device = options.workflow_options.device

            logger.info('model moved to device={}'.format(device))
            model.to(device)

            # instantiate the optimizer and scheduler
            logger.info('creating optimizers...')
            if optimizers_fn is not None:
                optimizers, schedulers, per_step_scheduler_fn = optimizers_fn(datasets, model)
                logger.info('optimizers created successfully!')
            else:
                logger.info('optimizer fn is None! No optimizer created.')
                optimizers, schedulers, per_step_scheduler_fn = None, None, None

            logger.info('creating losses...')
            losses = loss_creator(datasets, losses_fn)
            logger.info('losses created successfully!')

            num_epochs = options.training_parameters.num_epochs

            callbacks_per_epoch = []
            if self.callbacks_per_epoch is not None:
                callbacks_per_epoch += self.callbacks_per_epoch

            callbacks_per_batch = []
            if self.trainer_callbacks_per_batch is not None:
                callbacks_per_batch.append(self.trainer_callbacks_per_batch)
            if self.callbacks_per_batch is not None:
                callbacks_per_batch += self.callbacks_per_batch

            callbacks_per_batch_loss_terms = []
            if self.callbacks_per_batch_loss_terms is not None:
                callbacks_per_batch_loss_terms += self.callbacks_per_batch_loss_terms
            logger.info('callbacks created successfully!')

            # run the callbacks  before training
            if self.callbacks_pre_training is not None:
                logger.info('running pre-training callbacks...')
                for callback in self.callbacks_pre_training:
                    try:
                        callback(options, history, model, losses=losses, outputs=None,
                                 datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch,
                                 optimizers_fn=optimizers_fn, optimizers=optimizers, trainer=self)
                    except Exception as e:
                        f = StringIO()
                        traceback.print_exc(file=f)
                        print(f'callback={callback} failed with exception={e}. Stacktrace=\n{f.getvalue()}')
                        logger.error(f'callback={callback} failed with exception={e}. Stacktrace=\n{f.getvalue()}')
                logger.info('pre-training callbacks completed!')

            for epoch in range(num_epochs):
                logger.info('started training epoch {}'.format(epoch))
                run_eval = (epoch == 0 and not self.skip_eval_epoch_0) or (epoch + 1) % eval_every_X_epoch == 0

                outputs_epoch, history_epoch = self.run_epoch_fn(
                    options,
                    datasets,
                    optimizers,
                    model,
                    losses,
                    schedulers,
                    per_step_scheduler_fn,
                    history,
                    callbacks_per_batch,
                    callbacks_per_batch_loss_terms,
                    run_eval=run_eval,
                    force_eval_mode=False)
                history.append(history_epoch)

                logger.info('finished training epoch {}'.format(epoch))

                last_epoch = epoch + 1 == num_epochs

                logger.info('callbacks started')
                for callback in callbacks_per_epoch:
                    try:
                        callback(options, history, model, losses=losses, outputs=outputs_epoch,
                                 datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch,
                                 optimizers_fn=optimizers_fn, optimizers=optimizers, last_epoch=last_epoch, trainer=self)
                    except Exception as e:
                        f = StringIO()
                        traceback.print_exc(file=f)
                        logger.error(f'callback={callback} failed with exception={e}.\n Stack={f.getvalue()}')

                logger.info(f'callbacks epoch {epoch} finished')

            # finally run the post-training callbacks
            if with_final_evaluation:
                logger.info('started final evaluation...')

                outputs_epoch, history_epoch = self.run_epoch_fn(
                    options=options,
                    datasets=datasets,
                    optimizers=None,
                    model=model,
                    losses=losses,
                    schedulers=None,
                    per_step_schedulers=None,
                    history=history,
                    callbacks_per_batch=callbacks_per_batch,
                    callbacks_per_batch_loss_terms=callbacks_per_batch_loss_terms,
                    run_eval=True,
                    force_eval_mode=True)
                logger.info('finished final evaluation...')
                history.append(history_epoch)

            if self.callbacks_post_training is not None:
                logger.info('started post training callbacks...')
                for callback in self.callbacks_post_training:
                    try:
                        callback(options, history, model, losses=losses, outputs=outputs_epoch,
                                 datasets=datasets, datasets_infos=datasets_infos, callbacks_per_batch=callbacks_per_batch,
                                 optimizers_fn=optimizers_fn, trainer=self)
                    except Exception as e:
                        f = StringIO()
                        traceback.print_exc(file=f)
                        print(f'callback={callback} failed with exception={e}.\n Stack={f.getvalue()}')
                        logger.error(f'callback={callback} failed with exception={e}.\n Stack={f.getvalue()}')

                logger.info('finished post training callbacks...')

        except (KeyboardInterrupt, RuntimeError, ExceptionAbortRun) as e:
            # since we are about to exit the process, explicitly
            # dispose the datasets to make sure resources are properly disposed of
            logger.info(f'Exception received. closing datasets explicitly. Exception={e}', exc_info=True)
            clean_up(datasets)

            # since the resources are released, we can now re-raise the exception
            raise e

        # do not explicitly clean up the datasets since these were
        # created outside the trainer
        clean_up(datasets=None)
        return RunMetadata(
            history=history,
            options=options,
            outputs=outputs_epoch,
            datasets_infos=datasets_infos
        )
