import collections
from numbers import Number
from typing import Callable, Sequence, Optional, Tuple



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

        
from basic_typing import HistoryStep, History
from .callback import Callback
from store import RunStore
import logging


logger = logging.getLogger(__name__)


class CallbackEarlyStopping(Callback):
    """
    Use historical runs to evaluate if a run is promising. If not, early stop will raise :class:`ExceptionAbortRun`
    """
    def __init__(
            self,
            store: RunStore,
            loss_fn: Callable[[HistoryStep], float],
            raise_stop_fn: Optional[Callable[[float, History], Tuple[bool, str]]] = None,
            checkpoints: Sequence[float] = (0.1, 0.25, 0.5, 0.75),
            discard_if_among_worst_X_performers: float = 0.6,
            only_consider_full_run: bool =True,
            min_number_of_runs: int = 10):
        """

        Args:
            store: how to retrieve previous runs
            loss_fn: extract a loss value from an history step. This will be used to rank the runs
            checkpoints: define the number of checks (expressed as fraction of total epochs) to evaluate
                this run against the historical database of runs.
            discard_if_among_worst_X_performers: for each checkpoint, the current run is ranked among all the
                runs using `loss_fn` and `store`. If the runs is X% worst performer, discard the run
            min_number_of_runs: collect at least this number of runs before applying the early stopping.
                larger number means better estimation of the worst losses.
            raise_stop_fn: specify if a run should be stopped. For example, this can be useful to discard
                the parameters that make the model diverge very early. It takes as input (loss, history)
                and return `True` if the run should be stopped
            only_consider_full_run: if True, the checkpoints's threshold is calculated for the runs
                that have been completed (i.e., aborted run will not be used)
        """
        self.only_consider_full_run = only_consider_full_run
        self.raise_stop_fn = raise_stop_fn
        self.min_number_of_runs = min_number_of_runs
        self.discard_if_among_worst_X_performers = discard_if_among_worst_X_performers
        self.checkpoints = checkpoints
        self.loss_fn = loss_fn
        self.store = store
        self.max_loss_by_epoch = None
        self.nb_checked_checkpoints = 0

        assert 0 < discard_if_among_worst_X_performers < 1, 'must be a fraction!'
        for c in checkpoints:
            assert 0 < c < 1, 'must be a fraction!'

    def _initialize(self, num_epochs: int) -> None:
        logger.info('initializing run analysis...')
        # beware: the eval run MUST be synchronized with the epoch checkpoints
        checkpoints_epoch = [int(f * num_epochs) for f in self.checkpoints]
        try:
            all_runs = self.store.load_all_runs()
        except RuntimeError as e:
            # no file available, not initialized!
            logger.error(f'exception opening the store={e}')
            return

        # collect loss for all runs at given checkpoints
        losses_by_step = collections.defaultdict(list)
        for run in all_runs:
            run_losses_by_step = {}
            for e in checkpoints_epoch:
                if run.history is not None and len(run.history) > e:
                    loss = self.loss_fn(run.history[e - 1])
                    if loss is not None:
                        assert isinstance(loss, Number), f'expected `float`, got={loss}'
                        run_losses_by_step[e] = loss

            if (self.only_consider_full_run and len(run_losses_by_step) == len(checkpoints_epoch)) or not self.only_consider_full_run:
                for e, v in run_losses_by_step.items():
                    losses_by_step[e].append(v)

        # for each checkpoint, sort the losses, and calculate the worst X% of the runs
        # the current run MUST be better than the threshold or it will be pruned
        max_loss_by_epoch = {}
        for e, values in losses_by_step.items():
            if len(values) < self.min_number_of_runs:
                # not enough runs to get reliable estimate, keep this run!
                max_loss_by_epoch[e] = None
                continue

            if values is not None:
                values = sorted(values)
                rank = round(len(values) * (1.0 - self.discard_if_among_worst_X_performers))
                threshold = values[rank]
            else:
                threshold = None
            max_loss_by_epoch[e] = threshold
        self.max_loss_by_epoch = max_loss_by_epoch
        logger.info(f'max_loss_by_step={max_loss_by_epoch}')

    def __call__(self, options, history: History, model, **kwargs):
        logger.info('started!')
        num_epochs = options.training_parameters.num_epochs

        if self.max_loss_by_epoch is None:
            logger.debug('initializing the checkpoints...')
            self._initialize(num_epochs)

        epoch = len(history)
        loss = self.loss_fn(history[-1])

        if self.raise_stop_fn is not None:
            # check if we are satisfying early termination criteria
            # e.g., Nan, very slow loss decrease...
            should_be_stopped, reason = self.raise_stop_fn(loss, history)
            if should_be_stopped:
                logger.info(f'epoch={epoch}, loss={loss}, early termination. Reason={reason}')
                raise ExceptionAbortRun(
                    history=history,
                    reason=f'Early termination (epoch={len(history)}). loss={loss}. Reason={reason}!')

        # return ONLY after the `raise_stop_fn` check: often,
        # the `loss` will be based on the validation (potentially mostly none)
        # while `raise_stop_fn` check will use the training.
        if loss is None:
            logger.debug('loss is None!')
            return

        if self.max_loss_by_epoch is None:
            # we can't process! No previous runs
            logger.info('self.max_loss_by_epoch is None, No previous runs!')
            return

        max_loss = self.max_loss_by_epoch.get(epoch)
        if max_loss is not None:
            self.nb_checked_checkpoints += 1
            if loss > max_loss:
                logger.info(f'epoch={epoch}, loss={loss} > {max_loss}, the run is discarded!')
                raise ExceptionAbortRun(
                    history=history,
                    reason=f'loss={loss} is too high (threshold={max_loss}, '
                           f'minimum={self.discard_if_among_worst_X_performers} of the runs')
            else:
                logger.info(f'epoch={epoch}, run passed the checkpoint. loss={loss} <= {max_loss}')

    def __del__(self):
        # since it can be tricky to setup properly (e.g., synchronize the epochs of
        # eval run with the checkpoint). These errors should help diagnose a
        # misconfiguration!
        if self.max_loss_by_epoch is None:
            logger.error('`max_loss_by_epoch` is None, no previously run store was empty?')
            return

        logger.info(f'performed {self.nb_checked_checkpoints} / {len(self.max_loss_by_epoch)} checkpoints!')
        if self.nb_checked_checkpoints == 0:
            logger.error('no checkpoints checked! Misconfiguration or first hyper-parameter run?')
