from functools import partial
import time
import copy
import logging
import os
from packaging.version import Version

import torch
import torch.utils.data


from utilities import create_or_recreate_folder, find_default_dataset_and_split_names, get_device
from .callback import Callback
from collate import default_collate_fn

logger = logging.getLogger(__name__)


# pytorch compatibility layer
if Version(torch.__version__) >= Version('1.2'):
    from torch.utils.data import IterableDataset
else:
    class IterableDataset:
        def __iter__(self):
            raise NotImplementedError('requires pytorch >= 1.2')


class MyIterableDataset(IterableDataset):
    def __getitem__(self, index):
        return self.sequence[index]

    def __init__(self, sequence):
        super().__init__()
        self.sequence = sequence

    def __iter__(self):
        return iter(self.sequence)


class PerBatchProfilerStep:
    def __init__(self, profiler):
        self.profiler = profiler

    def __call__(self, *args, **kwargs):
        self.profiler.step()


class CallbackProfiler(Callback):
    """
    Run the torch.profiler while training the model

    A profiler log will be created in the folder <output_root>/static/<table_name>

    To visualize the output:
    - pip install torch_tb_profiler
    - tensorboard --logdir=<output_root>/static/model_profiler
    - in a browser: http://localhost:6006/#pytorch_profiler

    Alternatively, traces can be loaded using chrome partially:
    - open chrome and open page: chrome://tracing
    - load trace chrome_trace.json
    """
    def __init__(
            self,
            dataset_name=None,
            split_name=None,
            table_name='model_profiler',
            with_preprocessed_batch=False,
            schedule_kwargs=None):
        """

        Args:
            dataset_name:
            split_name:
            table_name:
            with_preprocessed_batch: if True, the batches will be preprocessed and won't appear in the
                profiler results (requires large RAM as the whole epoch will be stored in RAM)
        """
        if schedule_kwargs is None:
            schedule_kwargs = {'wait': 2, 'warmup': 1, 'active': 10, 'repeat': 1}

        self.split_name = split_name
        self.dataset_name = dataset_name
        self.root_output = None
        self.table_name = table_name
        self.with_preprocessed_batch = with_preprocessed_batch
        self.schedule_kwargs = schedule_kwargs

    def first_time(self, options, datasets):
        root = os.path.dirname(options.workflow_options.sql_database_path)
        self.root_output = os.path.join(root, 'static', self.table_name)
        create_or_recreate_folder(self.root_output)

        if self.split_name is None or self.dataset_name is None:
            if self.split_name is None:
                # default to training split, this is what we want to
                # measure in most cases
                self.split_name = options.workflow_options.train_split

            self.dataset_name, self.split_name = find_default_dataset_and_split_names(
                datasets,
                default_dataset_name=self.dataset_name,
                default_split_name=self.split_name)

            logger.info(f'Profiler data: dataset_name={self.dataset_name}, split_name={self.split_name}')

            if self.split_name is None or self.dataset_name is None:
                # no suitable dataset found
                return

    def __call__(self, options, history, model_orig, losses, outputs, datasets, datasets_infos, callbacks_per_batch,
                 **kwargs):
        logger.info('CallbackProfiler profiling model...')

        if self.root_output is None:
            self.first_time(options, datasets)

        current_version = Version(torch.__version__)
        if current_version <= Version('1.8.1'):
            logger.error(f'PyTorch version={torch.__version__} does not support torch.profiler')
            return

        from torch.profiler import tensorboard_trace_handler

        # copy the model! We don't want to start the real training just yet
        # as this would influence the training if this callback is present
        # or not
        trainer = kwargs.get('trainer')
        optimizers = kwargs.get('optimizers')
        split = datasets[self.dataset_name][self.split_name]

        device = get_device(model_orig)

        if not self.with_preprocessed_batch:
            datasets_loader = {self.dataset_name: {self.split_name: split}}
        else:
            batches = [b for b in split]
            split_pytorch = torch.utils.data.DataLoader(
                MyIterableDataset(batches),
                num_workers=0,
                collate_fn=partial(default_collate_fn, device=device)
            )
            datasets_loader = {self.dataset_name: {self.split_name: split_pytorch}}
        time_start = time.perf_counter()
        profiler_schedule = torch.profiler.schedule(**self.schedule_kwargs)
        with torch.profiler.profile(
                schedule=profiler_schedule,
                profile_memory=False,
                with_stack=True,
                on_trace_ready=tensorboard_trace_handler(self.root_output),
        ) as profiler:
            # create a new model so that the memory used by the model is
            # also recorded by the profiler
            model = copy.deepcopy(model_orig)
            _, _ = trainer.run_epoch_fn(
                options,
                datasets_loader,
                optimizers,
                model,
                losses,
                None,
                None,
                history,
                callbacks_per_batch,
                [PerBatchProfilerStep(profiler)],
                run_eval=True,
                force_eval_mode=False)

        time_end = time.perf_counter()
        logger.info(f'Profiling_time (s)={time_end - time_start}')
        logger.info('CallbackProfiler profiling model done!')
