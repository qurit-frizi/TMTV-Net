import signal
import time
from multiprocessing import Event
from optional_import import optional_import
psutil = optional_import('psutil')


class GracefulKiller:
    """
    Coordinate the shutdown of the different processes making sure ALL
    child processes are stopped or terminated.
    """
    # this is the main event that all threads/processes MUST listen to shutdown
    abort_event = Event()

    def __init__(self, timeout=5.0):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.timeout = timeout

    def exit_gracefully(self, signum, frame):
        self.abort_event.set()
        time_start = time.perf_counter()

        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                # give some time for the process to normally shutdown
                while time.time() - time_start < self.timeout:
                    time.sleep(0.1)

                # if the process was not stopped within the allowed
                # time, kill it
                if child.status() != psutil.STATUS_STOPPED:
                    child.kill()
                    child.terminate()
            except:
                print(f'Killing process FAILED={child.pid}')

        # finally, kill himself!
        current_process.kill()
