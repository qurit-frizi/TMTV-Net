from abc import ABC, abstractmethod
from typing import Dict, Any, Sequence, List

from basic_typing import History

from params import HyperParameters
import pickle


Metrics = Dict[str, float]


class RunResult:
    """
    Represent the result of a run
    """
    def __init__(self, metrics: Metrics, hyper_parameters: HyperParameters, history: History, info: Any = None):
        """
        Args:
            metrics: the metrics to be recorded
            info: additional info related to the run
            hyper_parameters: hyper parameter that led to these metrics
        """
        self.info = info
        self.hyper_parameters = hyper_parameters
        self.metrics = metrics
        self.history = history


class RunStore(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the store
        """
        pass

    @abstractmethod
    def save_run(self, run_result: RunResult) -> None:
        """
        Save the results of a run

        Args:
            run_result: the results to record
        """

    @abstractmethod
    def load_all_runs(self) -> Sequence[RunResult]:
        """
        Load all the runs
        """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RunStoreFile(RunStore):
    """
    Linear file based store.

    Notes:
         * we don't keep the file open, since we may have
           several stores reading from this location (single writer
           but many reader). Reading and writing must happen on the same
           thread
    """
    def __init__(self, store_location: str, serializer=pickle):
        super().__init__()
        self.serializer = serializer
        self.store_location = store_location

        # do not keep file descriptor open BUT at least create an empty file,
        # meaning the store exists but is currently empty.
        with open(self.store_location, mode='ab') as f:
            pass

    def close(self) -> None:
        """
        Close the store
        """
        pass

    def save_run(self, run_result: RunResult) -> None:
        """
        Save the results of a run

        Args:
            run_result: the results to record
        """
        with open(self.store_location, mode='ab') as f:
            self.serializer.dump(run_result, f)

    def load_all_runs(self) -> List[RunResult]:
        """
        Load all the runs
        """
        results = []
        with open(self.store_location, mode='rb') as f:
            while True:
                try:
                    r = self.serializer.load(f)
                    results.append(r)
                except EOFError:
                    break

        return results
