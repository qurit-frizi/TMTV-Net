import io
import logging
import collections
import time
from queue import Empty
import functools
import traceback
from sequence import Sequence
from job_executor2 import JobExecutor2, default_queue_timeout


logger = logging.getLogger(__name__)


def single_function_to_run(batch, function_to_run):
    """
    apply a list of functions on a batch of data
    """
    for fn in function_to_run:
        batch = fn(batch)
    return batch


class Metadata:
    def __init__(self):
        # indicates how long it takes to get the
        self.time_to_get_next_batch = 0
        self.nb_batches = 0
        self.time_processing = 0
        self.time_pinning = 0
        self.time_start = time.perf_counter()
        self.pin_queue_size = 0  # record the number of batches ready when map is retrieving the next batch


class SequenceMap(Sequence):
    def __init__(
            self,
            source_split,
            nb_workers,
            function_to_run,
            max_jobs_at_once=None,
            queue_timeout=default_queue_timeout,
            debug_job_report_timeout=30.0,
            collate_fn=None,
            max_queue_size_pin=None):
        """
        Transform a sequence using a given function.

        Args:
            source_split: the input sequence
            function_to_run: the mapping function
            nb_workers: the number of workers that will process the split. If 0, no workers will be created.
            max_jobs_at_once: the maximum number of results that can be pushed in the result queue per process
                at once. If 0, no limit. If None, it will be set equal to the number of workers
            queue_timeout: the timeout used to pull results from the output queue
            collate_fn: a function to collate the batch of data or `None`
            debug_job_report_timeout: timeout after which if no job were processed a job report will be
                printed (e.g., to debug pipeline stalling)
            max_queue_size_pin: defines the max number of batches prefected. If None, defaulting to
                a size based on the number of workers. This only controls the final queue sized of the pin
                thread (the workers queue can be independently set)

        Note:
            The map may create more samples than the original sequence
        """
        super().__init__(source_split)

        assert isinstance(source_split, Sequence), '`source_split` must be a `Sequence`'

        if isinstance(function_to_run, collections.Sequence):
            # if we have a list of transforms, wrap them in a single function
            self.function_to_run = functools.partial(single_function_to_run, function_to_run=function_to_run)
        else:
            self.function_to_run = function_to_run
        self.queue_timeout = queue_timeout
        self.collate_fn = collate_fn
        self.debug_job_report_timeout = debug_job_report_timeout

        logger.info(f'SequenceMap created={self}, nb_workers={nb_workers}, max_jobs_at_once={max_jobs_at_once}')

        self.job_executor = None
        if nb_workers != 0:
            if max_jobs_at_once is None:
                # default: each worker can push at least one item
                # before blocking
                max_jobs_at_once = nb_workers

            self.job_executor = JobExecutor2(
                nb_workers=nb_workers,
                function_to_run=self.function_to_run,
                max_queue_size_per_worker=max_jobs_at_once,
                max_queue_size_pin=max_queue_size_pin)

        self.iter_source = None
        self.jobs_processed = None
        self.jobs_queued = None
        self.sequence_iteration_finished = None
        self.main_thread_list = None
        self.main_thread_index = None

        # keep track of important statistics
        self.debug_metadata = None

    def subsample_uids(self, uids, uids_name, new_sampler=None):
        subsampled_source = self.source_split.subsample_uids(uids, uids_name, new_sampler)

        # do not use worker processes: we probably just want a much smaller sequence
        return SequenceMap(
            subsampled_source,
            nb_workers=0,
            function_to_run=self.function_to_run,
            collate_fn=self.collate_fn)

    def subsample(self, nb_samples):
        subsampled_source = self.source_split.subsample(nb_samples)

        # do not use worker processes: we probably just want a much smaller sequence
        return SequenceMap(
            subsampled_source,
            nb_workers=0,
            function_to_run=self.function_to_run,
            collate_fn=self.collate_fn)

    def fill_queue(self):
        try:
            while not self.job_executor.is_full():
                i = self.iter_source.next_item(blocking=False)
                self.jobs_queued += 1
                self.job_executor.put(i)
        except StopIteration:
            # we are done!
            self.sequence_iteration_finished = True

    def initializer(self):
        """
        Initialize the sequence to iterate through batches
        """
        if self.job_executor is not None:
            self.job_executor.reset()

        self.jobs_processed = 0
        self.jobs_queued = 0

        self.main_thread_list = None
        self.main_thread_index = None
        self.sequence_iteration_finished = False

    def __next_local(self, next_fn):
        """
        Get the next elements

        Handles single item or list of items returned by next_fn
        :param next_fn: return the next elements
        """
        if self.main_thread_list is None:
            items = None
            while items is None or len(items) == 0:
                items = next_fn()

            is_sequence = isinstance(items, collections.Sequence) and not isinstance(items, collections.Mapping)
            if is_sequence:
                # sequence: we need to locally store the sequence and iterate it
                self.main_thread_list = items
                self.main_thread_index = 0
            else:
                # not a sequence: we can safely return the item
                return items

        # manage the iteration of an existing sequence
        if self.main_thread_index >= len(self.main_thread_list):
            raise IndexError(f'BUG! list size={len(self.main_thread_list)}, index={self.main_thread_index}')
        item = self.main_thread_list[self.main_thread_index]
        self.main_thread_index += 1
        if self.main_thread_index == len(self.main_thread_list):
            # we have exhausted our current list of items, resume the `function_to_run` calls
            self.main_thread_list = None
        return item

    def __next__(self):
        return self.next_item(blocking=True)

    def has_background_jobs(self):
        return not self.job_executor.is_idle()

    def next_item(self, blocking):
        def single_process_next():
            while True:
                i = self.iter_source.__next__()

                try:
                    items = self.function_to_run(i)

                except Exception as e:
                    # case where we have a job that failed: discard
                    print('-------------- ERROR in worker function --------------')
                    print(e)
                    print('-------------- first job will be aborted --------------')
                    traceback.print_exc()
                    print('-------------------------------------------------------')
                    continue

                return items

        def multiple_process_next():
            assert self.main_thread_list is None
            nb_background_jobs = self.has_background_jobs_previous_sequences()

            # we are only done once all the jobs have been completed!
            if self.sequence_iteration_finished and \
                    nb_background_jobs == 0 and \
                    self.job_executor.pin_memory_queue.empty():

                # collect some useful statistics
                if self.debug_metadata.nb_batches != 0:
                    logger.debug(
                        f'SequenceMap={self}, nb_batches_processed={self.debug_metadata.nb_batches},'
                        f'sequence_time={time.perf_counter() - self.debug_metadata.time_start}, '
                        f'total_sequence_overhead={self.debug_metadata.time_to_get_next_batch}, '
                        f'overhead_sequence_time_by_batch='
                        f'{self.debug_metadata.time_to_get_next_batch / self.debug_metadata.nb_batches}, '
                        f'average_job_processing_time='
                        f'{self.debug_metadata.time_processing / self.debug_metadata.nb_batches}, '
                        f'average_batch_pin_time='
                        f'{self.debug_metadata.time_pinning / self.debug_metadata.nb_batches}, '
                        f'average_batch_prefetch_size='
                        f'{self.debug_metadata.pin_queue_size / self.debug_metadata.nb_batches}, '
                    )

                # stop the sequence
                raise StopIteration()

            report_timeout_start = time.time()
            next_item_start = time.perf_counter()
            while True:
                try:
                    metadata, items = self.job_executor.pin_memory_queue.get(True, timeout=self.queue_timeout)
                    if items is None:
                        continue  # the job has failed, get the next item!
                    self.jobs_processed += 1
                    # ok, we are good now!
                    break
                except Empty:
                    # no job available, make sure the worker of the other pools are not starved
                    self.fill_queue_all_sequences()
                    if not blocking:
                        raise StopIteration()

                    if time.time() - report_timeout_start > self.debug_job_report_timeout:
                        print('------------------- STALLING -------------------')
                        f = io.StringIO()
                        self.job_executor.job_report(f=f)
                        logger.error('------------------- STALLING -------------------')
                        logger.error(f.getvalue())
                        report_timeout_start = time.time()
                        print(f.getvalue())

                    nb_background_jobs = self.has_background_jobs_previous_sequences()

                    # we are only done once all the jobs have been completed!
                    if self.sequence_iteration_finished and \
                            nb_background_jobs == 0 and \
                            self.job_executor.pin_memory_queue.empty():
                        print('--------------- IDLE STOP ----------------')
                        raise StopIteration()

            next_item_end = time.perf_counter()
            self.debug_metadata.nb_batches += 1
            self.debug_metadata.time_to_get_next_batch += next_item_end - next_item_start
            self.debug_metadata.time_processing += metadata.job_processing_finished - metadata.job_created
            self.debug_metadata.time_pinning += metadata.job_pin_thread_received - metadata.job_results_queued
            self.debug_metadata.pin_queue_size += self.job_executor.pin_memory_queue.qsize()
            return items

        if self.job_executor is None:
            # use the main thread for the processing. In this case we need to mimic the behaviour
            # of the pool (i.e., if the `function_to_run` returns a list, we need to process one
            # item at a time
            items = self.__next_local(single_process_next)

        else:
            self.fill_queue()
            items = self.__next_local(multiple_process_next)

        if self.collate_fn is not None:
            items = self.collate_fn(items, source=self)
        return items

    def __iter__(self):
        self.initializer()
        self.iter_source = self.source_split.__iter__()
        self.debug_metadata = Metadata()
        return self

    def close(self):
        """
        Finish and join the existing pool processes
        """
        if self.job_executor is not None:
            self.job_executor.close()

        if self.source_split is not None:
            self.source_split.close()

