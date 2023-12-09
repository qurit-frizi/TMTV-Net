import copy
import io
import os
import sys
import threading
import time
import traceback

from time import sleep, perf_counter

from threadpoolctl import threadpool_limits
from typing import Callable, Optional

from basic_typing import Batch
import logging
import numpy as np
from queue import Queue as ThreadQueue, Empty

# Make sure we start a new process in an empty state so
# that Windows/Linux environment behave the similarly
from torch import multiprocessing

from graceful_killer import GracefulKiller

multiprocessing = multiprocessing.get_context("spawn")
#multiprocessing = multiprocessing.get_context("fork")
from multiprocessing import Event, Process, Queue, Value

# timeout used for the queues
default_queue_timeout = 1e-3


def flush_queue(queue):
    if queue is None:
        return
    try:
        while not queue.empty():
            print('Flushing queue item, queue=', queue)
            _ = queue.get(timeout=1.0)
            print('Flushing queue item, DONE queue=', queue)
    except Exception as e:
        print('Exception intercepted=', e, type(queue))


class JobMetadata:
    def __init__(self, job_session_id):
        self.job_created = time.perf_counter()
        self.job_processing_finished = None
        self.job_results_queued = None

        # pinning thread
        self.job_pin_thread_received = None
        self.job_pin_thread_queued = None

        self.job_session_id = job_session_id


def worker(
        input_queue: Queue,
        output_queue: Queue,
        transform: Callable[[Batch], Batch],
        global_abort_event: Event,
        local_abort_event: Event,
        synchronized_stop: Event,
        wait_time: float,
        seed: int) -> None:
    """
    Worker that will execute a transform on a process.

    Args:
        input_queue: the queue to listen to
        output_queue:  the queue to output the results
        transform: the transform to be applied on each data queued
        global_abort_event: specify when the jobs need to shutdown
        local_abort_event: specify when the jobs need to shutdown but only for a given job executor
        wait_time: process will sleep this amount of time when input queue is empty
        seed: an int to seed random generators
        synchronized_stop: the workers will NOT exit the process until this event is set to ensure
            the correct order of destruction of workers/threads/queues

    Returns:
        None
    """
    np.random.seed(seed)
    item = None
    job_session_id = None
    job_metadata = None
    while True:
        try:
            if not global_abort_event.is_set() and not local_abort_event.is_set():
                if item is None:
                    if not input_queue.empty():
                        try:
                            job_session_id, item = input_queue.get()
                            job_metadata = JobMetadata(job_session_id=job_session_id)
                        except Exception as e:
                            # possible exception:  `unable to open shared memory object </torch_XXX_YYYYY>
                            # we MUST queue a `None` to specify that we received something but there was an error
                            print(f'Exception <input_queue.get> in background worker PID={os.getpid()}, E={e}', flush=True)
                            item = None
                            # DO continue: we want to push `None`

                    else:
                        sleep(wait_time)
                        continue

                    if transform is not None and item is not None:
                        try:
                            item = transform(item)
                            assert job_metadata is not None
                            job_metadata.job_processing_finished = time.perf_counter()
                        except Exception as e:
                            # exception is intercepted and skip to next job
                            # here we send the `None` result anyway to specify the
                            # job failed. we MUST send the `None` so that jobs queued
                            # and jobs processed match.
                            print('-------------- ERROR in worker function --------------')
                            print(f'Exception in background worker PID={os.getpid()}, E={e}')
                            print('-------------- first job will be aborted --------------')
                            string_io = io.StringIO()
                            traceback.print_exc(file=string_io)
                            print(string_io.getvalue())
                            print('-------------------------------------------------------', flush=True)
                            item = None

                while True:
                    try:
                        assert job_metadata is not None
                        job_metadata.job_results_queued = time.perf_counter()
                        output_queue.put((job_metadata, item))
                        item = None
                        break  # success, get ready to get a new item from the queue

                    except Exception as e:
                        # exception is intercepted and skip to next job
                        print(f'Exception <output_queue.put> in background worker '
                              f'thread_id={os.getpid()}, E={e}, ITEM={item}, id={job_session_id}', flush=True)

                        # re-try to push on the queue!
                        sleep(wait_time)
                        continue

            else:
                flush_queue(input_queue)
                print(f'Worker={os.getpid()} Stopping (abort_event SET)!!', flush=True)
                synchronized_stop.wait()  # type: ignore
                print(f'Worker={os.getpid()} Stopped (abort_event SET)!!', flush=True)
                return

        except KeyboardInterrupt:
            # the main thread will handle the keyboard interrupt
            # using synchronized shutdown of the workers
            continue

        except Exception as e:
            # exception is intercepted and skip to next job
            print('-------------- ERROR in worker function --------------')
            print(f'Exception in background worker thread_id={os.getpid()}, E={e}, ITEM={item}, id={job_session_id}')
            print('-------------- Error detail --------------')
            string_io = io.StringIO()
            traceback.print_exc(file=string_io)
            print(string_io.getvalue())
            print('-------------------------------------------------------', flush=True)
            continue

        except:
            # critical issue, stop everything!
            print('-------------- ERROR (ANY) in worker function --------------')
            print(f'Exception in background worker thread_id={os.getpid()}, ITEM={item}, id={job_session_id}')
            print('-------------- Error detail --------------')
            string_io = io.StringIO()
            traceback.print_exc(file=string_io)
            print(string_io.getvalue())
            print('-------------------------------------------------------', flush=True)
            global_abort_event.set()  # type: ignore

    print(f'worker unreachable! thread_id={os.getpid()}', flush=True)


def collect_results_to_main_process(
        job_session_id: Value,
        jobs_queued: Value,
        worker_output_queue: Queue,
        output_queue: ThreadQueue,
        global_abort_event: Event,
        synchronized_stop: Event,
        local_abort_event: Event,
        stop_event: Event,
        wait_time: float) -> None:

    assert output_queue is not None
    item = None
    item_job_session_id = None
    while True:
        try:
            if stop_event.is_set():
                print(f'Thread={threading.get_ident()}, (stop_event set) shuting down!', flush=True)
                return

            if global_abort_event.is_set() or local_abort_event.is_set():
                flush_queue(worker_output_queue)
                flush_queue(output_queue)
                print(f'Thread={threading.get_ident()}, (abort_event set) shuting down!', flush=True)
                synchronized_stop.wait()
                print(f'Thread={threading.get_ident()}, (abort_event set) shutdown!', flush=True)
                return

            # if we don't have an item we need to fetch it first. If the queue we want to get it from it empty, try
            # again later
            if item is None and item_job_session_id is None:
                if not worker_output_queue.empty():
                    try:
                        job_metadata, item = worker_output_queue.get(timeout=wait_time)
                        item_job_session_id = job_metadata.job_session_id
                        job_metadata.job_pin_thread_received = time.perf_counter()

                    except Empty:
                        # even if the `current_queue` was not empty, another thread might have stolen
                        # the job result already. Just continue to the next queue
                        continue

                    except RuntimeError as e:
                        print(f'collect_results_to_main_process (RuntimeError) Queue={threading.get_ident()} GET error={e}', flush=True)
                        print('------------ Exception Traceback --------------')
                        traceback.print_exc(file=sys.stdout)
                        print('-----------------------------------------------', flush=True)

                        # the queue was sending something but failed
                        # discard this data and continue
                        item = None

                    except ConnectionError as e:
                        print(f'collect_results_to_main_process (ConnectionError) Queue={threading.get_ident()} GET error={e}', flush=True)
                        # the queue was sending something but failed
                        # discard this data and continue
                        item = None

                    if item is None:
                        # this job FAILED so there is no result to queue. Yet, we increment the
                        # job counter since this is used to monitor if the executor is
                        # idle
                        with jobs_queued.get_lock():
                            jobs_queued.value += 1

                        # fetch a new job result!
                        item_job_session_id = None
                        continue

                else:
                    sleep(wait_time)
                    continue

            if item is None and item_job_session_id is None:
                continue

            if item_job_session_id != job_session_id.value:
                # this is an old result belonging to the previous
                # job session. Discard it and process a new one
                item = None
                item_job_session_id = None
                with jobs_queued.get_lock():
                    jobs_queued.value += 1
                continue

            if not output_queue.full():
                job_metadata.job_pin_thread_queued = time.perf_counter()
                output_queue.put((job_metadata, item))

                item_job_session_id = None
                with jobs_queued.get_lock():
                    jobs_queued.value += 1

                item = None
            else:
                sleep(wait_time)
                continue
        except KeyboardInterrupt:
            # the main thread will handle the keyboard interrupt
            # using synchronized shutdown of the workers
            continue
        except Exception as e:
            print(f'Thread={threading.get_ident()}, thread shuting down (Exception)', flush=True)
            print('------------ Exception Traceback --------------')
            traceback.print_exc(file=sys.stdout)
            print('-----------------------------------------------', flush=True)
            global_abort_event.set()  # critical issue, stop everything!
            continue

    print(f'collect_results_to_main_process unreachable! thread_id={os.getpid()}', flush=True)


class JobExecutor2:
    """
    Execute jobs on multiple processes.

    At a high level, we have worker executing on multiple processes. Each worker will be fed
    by an input queue and results of the processing pushed to an output queue.

    Pushing data on a queue is very fast BUT retrieving it from a different process takes time.
    Even if PyTorch claims to have memory shared arrays, retrieving a large array from a queue
    has a linear runtime complexity (still true with pytorch 1.11). To limit this copy penalty, 
    we can use threads that copy from the worker process to the main process (`pinning` threads. 
    Here, sharing data between threads is almost free).

    Notes:
        - This class was designed for maximum speed and not reproducibility in mind.
            The processed of jobs will not keep their ordering.
        - the proper destruction of the job executor is the most difficult part with risk of process
          hangs or memory leaks:
            - first threads and processes are signaled to stop their processing and avoid pushing results to queues
            - queues are emptied to avoid memory leaks (in case of abrupt termination)
            - queues are joined
            - processes are joined
            - threads are joined
    """
    def __init__(
            self,
            nb_workers: int,
            function_to_run: Callable[[Batch], Batch],
            max_queue_size_per_worker: int = 2,
            max_queue_size_pin_thread_per_worker: int = 3,
            max_queue_size_pin: Optional[int] = None,
            wait_time: float = default_queue_timeout,
            wait_until_processes_start: bool = True,
            restart_crashed_worker: bool = True):
        """

        Args:
            nb_workers: the number of workers (processes) to execute the jobs
            function_to_run: the job to be executed
            max_queue_size_per_worker: define the maximum number of job results that can be stored
                before a process is blocked (i.e., larger queue size will improve performance but will
                require more memory to store the results). the pin_thread need to process the result before
                the blocked process can continue its processing.
            max_queue_size_pin_thread_per_worker: define the maximum number of results available on the main
                process (i.e., larger queue size will improve performance but will require more memory
                to store the results). Overriden by `max_queue_size_pin` if defined
            max_queue_size_pin: define the maximum number of results available on the main
                process. Overrides `max_queue_size_pin_thread_per_worker` if not None
            wait_time: the default wait time for a process or thread to sleep if no job is available
            wait_until_processes_start: if True, the main process will wait until the worker processes and
                pin threads are fully running
            restart_crashed_worker: if True, the worker will be restarted. The worker's crashed job result will
                be lost
        """
        self.wait_until_processes_start = wait_until_processes_start
        self.wait_time = wait_time
        self.max_queue_size_pin_thread_per_worker = max_queue_size_pin_thread_per_worker
        self.max_queue_size_per_worker = max_queue_size_per_worker
        self.max_queue_size_pin = max_queue_size_pin
        self.function_to_run = function_to_run
        self.nb_workers = nb_workers

        self.global_abort_event = GracefulKiller.abort_event
        self.local_abort_event = Event()
        self.synchronized_stop = Event()

        self.main_thread = threading.get_ident()

        self.worker_control = 0
        self.worker_input_queues = []
        self.worker_output_queues = []
        self.processes = []
        self.process_alive_check_time = time.perf_counter()
        self.restart_crashed_worker = restart_crashed_worker

        self.jobs_processed = Value('i', 0)
        self.jobs_queued = 0

        self.pin_memory_threads = []
        self.pin_memory_queue = None
        self.pin_memory_thread_stop_events = []

        # we can't cancel jobs, so instead record a session ID. If session of
        # the worker and current session ID do not match
        # it means the results of these tasks should be discarded
        self.job_session_id = Value('i', 0)

        self.start()

    def start(self, timeout: float = 10.0) -> None:
        """
        Start the processes and queues.

        Args:
            timeout:

        Returns:

        """

        # reset the events
        self.global_abort_event.clear()
        self.local_abort_event.clear()
        self.synchronized_stop.clear()

        if self.pin_memory_queue is None:
            queue_size = self.max_queue_size_pin_thread_per_worker * self.nb_workers
            if self.max_queue_size_pin is not None:
                queue_size = self.max_queue_size_pin
            else:
                queue_size = self.max_queue_size_pin_thread_per_worker * self.nb_workers
                
            self.pin_memory_queue = ThreadQueue(queue_size)

        if self.nb_workers == 0:
            # nothing to do, this will be executed synchronously on
            # the main process
            return

        if len(self.processes) != self.nb_workers:
            print(f'Starting jobExecutor={self}, on process={os.getpid()} nb_workers={self.nb_workers}')
            logging.debug(f'Starting jobExecutor={self}, on process={os.getpid()} nb_workers={self.nb_workers}')
            if len(self.processes) > 0 or len(self.pin_memory_threads) > 0:
                self.close()
            self.local_abort_event.clear()

            # first create the worker input/output queues
            for i in range(self.nb_workers):  # maxsize = 0
                self.worker_input_queues.append(Queue(maxsize=self.max_queue_size_per_worker))
                self.worker_output_queues.append(Queue(self.max_queue_size_per_worker))

            # allocate one thread per process to move the data from the
            # process memory space to the main process memory
            self.pin_memory_threads = []
            self.pin_memory_thread_stop_events = []
            for i in range(self.nb_workers):
                # stop event is used to notify the pinning thread
                # to stop its processing (e.g., so that it could be restarted)
                stop_event = Event()
                self.pin_memory_thread_stop_events.append(stop_event)

                pin_memory_thread = threading.Thread(
                    name=f'JobExecutorThreadResultCollector-{i}',
                    target=collect_results_to_main_process,
                    args=(
                        self.job_session_id,
                        self.jobs_processed,
                        self.worker_output_queues[i],
                        self.pin_memory_queue,
                        self.global_abort_event,
                        self.local_abort_event,
                        self.synchronized_stop,
                        stop_event,
                        self.wait_time
                    ))
                self.pin_memory_threads.append(pin_memory_thread)
                pin_memory_thread.daemon = False
                pin_memory_thread.start()
                print(f'Thread={pin_memory_thread.ident}, thread started')

            # make sure a single process can use only one thread
            with threadpool_limits(limits=1, user_api='blas'):
                for i in range(self.nb_workers):
                    p = Process(
                        target=worker,
                        name=f'JobExecutorWorker-{i}',
                        args=(
                            self.worker_input_queues[i],
                            self.worker_output_queues[i],
                            self.function_to_run,
                            self.global_abort_event,
                            self.local_abort_event,
                            self.synchronized_stop,
                            self.wait_time, i
                        ))
                    p.daemon = False
                    p.start()
                    self.processes.append(p)
                    print(f'Worker={p.pid} started!')
                    logging.debug(f'Child process={p.pid} for jobExecutor={self}')

            self.worker_control = 0

        if self.wait_until_processes_start:
            # wait until all the processes and threads are alive
            waiting_started = perf_counter()
            while True:
                wait_more = False
                for p in self.processes:
                    if not p.is_alive():
                        wait_more = True
                        continue
                for t in self.pin_memory_threads:
                    if not t.is_alive():
                        wait_more = True
                        continue

                if wait_more:
                    waiting_time = perf_counter() - waiting_started
                    if waiting_time < timeout:
                        sleep(self.wait_time)
                    else:
                        logging.error('the worker processes/pin threads were too slow to start!')

                break

        logging.debug(f'jobExecutor ready={self}')

    def close(self, timeout: float = 10) -> None:
        """
        Stops the processes and threads.

        Args:
            timeout: time allowed for the threads and processes to shutdown cleanly
                before using `terminate()`

        """
        if threading.get_ident() != self.main_thread:
            logging.error(f'attempting to close the executor from a '
                          f'thread={threading.get_ident()} that did not create it! ({self.main_thread})')
            return

        # notify all the threads and processes to be shut down
        logging.info(f'Setting `abort_event` to interrupt Processes and threads! (JobExecutor={self})')
        self.local_abort_event.set()

        # First, stop the queue BEFORE the threads/processes, else the
        # data may be corrupted and process may be terminated
        # we also need to remove all data on the queues
        for i, p in enumerate(self.processes):
            logging.info(f'closing worker input queue[{i}]')
            flush_queue(self.worker_input_queues[i])
            self.worker_input_queues[i].close()
            logging.info(f'joining worker input queue[{i}]')
            self.worker_input_queues[i].join_thread()
            logging.info(f'closing worker input queue[{i}] DONE')

            logging.info(f'closing worker output queue[{i}]')
            flush_queue(self.worker_output_queues[i])
            self.worker_output_queues[i].close()
            self.worker_output_queues[i].join_thread()

        logging.info(f'flushing pin_memory_queue')
        flush_queue(self.pin_memory_queue)
        logging.info(f'flushing pin_memory_queue dones!')

        # we are in good shape to close the workers & threads,
        # send the signal!
        logging.info(f'Synchronized_stop, executor={self}')
        self.synchronized_stop.set()

        # give some time to the threads/processes to shutdown normally
        shutdown_time_start = perf_counter()
        while True:
            wait_more = False
            if len(self.processes) != 0:
                for p in self.processes:
                    if p.is_alive():
                        wait_more = True
                        break
            if len(self.pin_memory_threads):
                for t in self.pin_memory_threads:
                    if t.is_alive():
                        wait_more = True
                        break

            shutdown_time = perf_counter() - shutdown_time_start
            if wait_more:
                if shutdown_time < timeout:
                    sleep(0.1)
                    continue
                else:
                    logging.error('a job did not respond to the shutdown request in the allotted time. '
                                  'It could be that it needs a longer timeout or a deadlock. The processes'
                                  'will now be forced to shutdown!')

            # done normal shutdown or timeout
            break

        logging.info(f'Synchronized_stop step 2')

        if len(self.processes) != 0:
            logging.debug(f'JobExecutor={self}: shutting down workers...')
            [i.terminate() for i in self.processes]
            logging.debug(f'JobExecutor={self}: workers terminated')

            for i, p in enumerate(self.processes):
                logging.debug(f'JobExecutor={self}: worker={p} joining')
                p.join()
                logging.debug(f'JobExecutor={self}: worker={p} joined')

            self.worker_input_queues = []
            self.worker_output_queues = []
            self.processes = []
            logging.debug(f'workers cleaning done!')

        if len(self.pin_memory_threads) > 0:
            logging.debug(f'cleaning threads')
            for thread in self.pin_memory_threads:
                logging.debug(f'joining thread={thread.ident}')
                thread.join()
                logging.debug(f'joined thread={thread.ident}')
                del thread
            self.pin_memory_threads = []

            del self.pin_memory_queue
            self.pin_memory_queue = None
            logging.debug(f'cleaning threads done!')

        logging.debug(f'close done! (job_executor={self})')

    def is_full(self) -> bool:
        """
        Check if the worker input queues are full.

        Returns:
            True if full, False otherwise
        """
        if self.nb_workers == 0:
            return False

        for i in range(self.nb_workers):
            queue = self.worker_input_queues[self.worker_control]
            if not queue.full():
                return False
            self.worker_control = (self.worker_control + 1) % self.nb_workers

        return True

    def put(self, data: Batch) -> bool:
        """
        Queue a batch of data to be processed.

        Warnings:
            if the queues are full, the batch will NOT be appended

        Args:
            data: a batch of data to process

        Returns:
            True if the batch was successfully appended, False otherwise.
        """
        if self.nb_workers == 0:
            # if no asynchronous worker used, put the result
            # directly on the pin queue
            batch_in = copy.deepcopy(data)
            job_metadata = JobMetadata(job_session_id=0)
            batch_out = self.function_to_run(batch_in)
            job_metadata.job_processing_finished = time.perf_counter()
            job_metadata.job_results_queued = job_metadata.job_processing_finished
            job_metadata.job_pin_thread_received = job_metadata.job_processing_finished
            job_metadata.job_pin_thread_queued = job_metadata.job_processing_finished
            self.pin_memory_queue.put((job_metadata, batch_out))
            self.jobs_queued += 1
            return True
        else:
            for i in range(self.nb_workers):
                queue = self.worker_input_queues[self.worker_control]
                if not queue.full():
                    queue.put((self.job_session_id.value, data))
                    self.worker_control = (self.worker_control + 1) % self.nb_workers
                    self.jobs_queued += 1
                    return True

            # all queues are full, we have to wait
            return False

    def is_idle(self) -> bool:
        """
        Returns:
            True if the executor is not currently processing jobs
        """

        with self.jobs_processed.get_lock():
            is_idle = self.jobs_processed.value == self.jobs_queued

        if self.restart_crashed_worker:
            current_time = time.perf_counter()
            delta = current_time - self.process_alive_check_time
            if delta > 0.5:
                self.process_alive_check_time = current_time
                self._check_process_killed_and_restart()

        return is_idle

    def _check_process_killed_and_restart(self):
        """
        Verify the workers are alive. If not, restart a new process.
        """
        for n, w in enumerate(self.processes):
            if not w.is_alive():
                logging.error(f'worker={w.pid} crashed. Attempting to restart a new worker!')
                # Often, if a process crashed, the queue are also in an incorrect state
                # so restart the queues just in case these two are related
                self.worker_output_queues[n] = Queue(self.max_queue_size_per_worker)

                # restart the worker process
                with threadpool_limits(limits=1, user_api='blas'):
                    p = Process(
                        target=worker,
                        name=f'JobExecutorWorker-{n}',
                        args=(
                            self.worker_input_queues[n],
                            self.worker_output_queues[n],
                            self.function_to_run,
                            self.global_abort_event,
                            self.local_abort_event,
                            self.synchronized_stop,
                            self.wait_time, n
                        ))
                    p.daemon = False
                    p.start()
                    self.processes[n] = p
                    logging.info(f'worker={w.pid} crashed and successfully restarted with pid={p.pid}')
                    print(f'worker={w.pid} crashed and successfully restarted with pid={p.pid}',
                          file=sys.stderr,
                          flush=True)

                # shutdown the pinning thread
                # 1) notify the thread using `pin_memory_thread_stop_events`
                # 2) wait for the termination of the thread
                self.pin_memory_thread_stop_events[n].set()
                stop_event = Event()
                self.pin_memory_thread_stop_events[n] = stop_event
                self.pin_memory_threads[n].join(timeout=5.0)
                if self.pin_memory_threads[n].is_alive():
                    logging.error(f'thread={self.pin_memory_threads[n].ident} did not respond to shutdown!')
                    print(f'thread={self.pin_memory_threads[n].ident} did not respond to shutdown!',
                          file=sys.stderr,
                          flush=True)

                # restart the pinning thread process
                pin_memory_thread = threading.Thread(
                    name=f'JobExecutorThreadResultCollector-{n}',
                    target=collect_results_to_main_process,
                    args=(
                        self.job_session_id,
                        self.jobs_processed,
                        self.worker_output_queues[n],
                        self.pin_memory_queue,
                        self.global_abort_event,
                        self.local_abort_event,
                        self.synchronized_stop,
                        stop_event,
                        self.wait_time
                    ))
                self.pin_memory_threads[n] = pin_memory_thread
                pin_memory_thread.daemon = False
                pin_memory_thread.start()
                print(f'Thread={pin_memory_thread.ident}, pinning thread re-started')

                # most likely, the job was killed during the processing,
                # so increment the job counters to fake a result
                with self.jobs_processed.get_lock():
                    # we may have lost a maximum of queue size results
                    self.jobs_processed.value += self.max_queue_size_per_worker

    def job_report(self, f=sys.stdout):
        """
        Summary of the executor state. Useful for debugging.
        """
        f.write(f'JobExecutor={self}, Main process={os.getpid()}, main thread={threading.get_ident()}\n')
        f.write(f'NbProcesses={len(self.processes)}, NbThreads={len(self.pin_memory_threads)}\n')
        for p in self.processes:
            f.write(f'  worker PID={p.pid}, is_alive={p.is_alive()}\n')

        for i, q in enumerate(self.worker_input_queues):
            f.write(f'  worker_input_queue {i} is_empty={q.empty()}, is_full={q.full()}\n')

        for i, q in enumerate(self.worker_output_queues):
            f.write(f'  worker_output_queue {i} is_empty={q.empty()}, is_full={q.full()}\n')

        q = self.pin_memory_queue
        f.write(f'  pin_memory_queue is_empty={q.empty()}, is_full={q.full()}\n')

        for t in self.pin_memory_threads:
            f.write(f'  thread IDENT={t.ident}, is_alive={t.is_alive()}\n')

        f.write(f'nb_jobs_received={self.jobs_queued}, nb_jobs_processed={self.jobs_processed.value}, job_session_id={self.job_session_id.value}\n')

    def reset(self):
        """
        Reset the input and output queues as well as job session IDs.

        The results of the jobs that have not yet been calculated will be discarded
        """

        # here we could clear the queues for a faster implementation.
        # Unfortunately, this is not an easy task to properly
        # counts all the jobs processed or discarded due to the
        # multi-threading. Instead, all tasks queued are executed
        # and we use a `job_session_id` to figure out the jobs to be
        # discarded

        # empty the current queue results, they are not valid anymore!
        try:
            while not self.pin_memory_queue.empty():
                self.pin_memory_queue.get()
        except EOFError:  # in case the other process was already terminated
            pass
        # discard the results of the jobs that will not have the
        # current `job_session_id`
        with self.job_session_id.get_lock():
            self.job_session_id.value += 1

    def __del__(self):
        #logging.debug(f'JobExecutor={self}: destructor called')
        self.close()
