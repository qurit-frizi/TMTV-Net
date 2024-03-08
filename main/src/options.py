from typing import Optional, Any
import os
import torch
import logging
import platform
from pprint import pformat

logger = logging.getLogger(__name__)


def get_logging_root(logging_root: Optional[str] = None) -> str:
    """
    Return the data root directory
    """
    if logging_root is None:
        logging_root = os.environ.get('TRW_LOGGING_ROOT')

    if logging_root is None:
        if 'Windows' in platform.system():
            logging_root = 'c:/trw_logs/'
        else:
            logging_root = '$HOME/trw_logs/'

    logging_root = os.path.expandvars(os.path.expanduser(logging_root))
    return logging_root


class TrainingParameters:
    """
    Define here specific training parameters
    """
    def __init__(self, num_epochs: int, mixed_precision_enabled: bool = False, gradient_update_frequency: int = 1):
        self.num_epochs = num_epochs

        self.gradient_scaler = None
        if mixed_precision_enabled:
            try:
                from torch.cuda.amp import GradScaler
                self.gradient_scaler = GradScaler()

            except Exception as e:
                # mixed precision is not enabled
                logger.error(f'Mixed precision not enabled! Exception={e}')
        self.gradient_update_frequency = gradient_update_frequency

    def __repr__(self) -> str:
        return pformat(vars(self), indent=3, width=1)


class WorkflowOptions:
    """
    Define here workflow options
    """
    def __init__(self, logging_directory: str, device: torch.device):
        # expand special characters/variables
        logging_directory = os.path.expandvars(os.path.expanduser(logging_directory))
        
        self.device: torch.device = device
        self.train_split: str = 'train'
        self.logging_directory: str = logging_directory
        self.current_logging_directory: str = logging_directory
        self.trainer_run: int = 0
        self.sql_database_view_path: Optional[str] = None
        self.sql_database_path: Optional[str] = None
        self.sql_database: Optional[Any] = None

    def __repr__(self) -> str:
        return pformat(vars(self), indent=3, width=1)


class Runtime:
    """
    Define here the runtime configuration
    """
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return pformat(vars(self), indent=3, width=1)


class Options:
    """
    Create default options for the training and evaluation process.
    """
    def __init__(self,
                 logging_directory: Optional[str] = None,
                 num_epochs: int = 50,
                 device: Optional[torch.device] = None,
                 mixed_precision_enabled: bool = False,
                 gradient_update_frequency: int =1):
        """

        Args:
            logging_directory: the base directory where the logs will be exported for each trained model.
                If None and if the environment variable `LOGGING_DIRECTORY` exists, it will be used as root
                directory. Else a default folder will be used

            num_epochs: the number of epochs
            device: the device to train the model on. If `None`, we will try first any available GPU then
                revert to CPU
            mixed_precision_enabled: if `True`, enable mixed precision for the training
            gradient_update_frequency: defines how often (every `X` batches) the gradient is updated. This
                is done to simulate a larger effective batch size (e.g., batch_size = 64 and gradient_update_frequency = 3
                the effective batch size is gradient_update_frequency * batch_size)
        """
        logging_directory = get_logging_root(logging_directory)

        if device is None:
            if torch.cuda.device_count() > 0:
                env_device = os.environ.get('TRW_DEVICE')
                if env_device is not None:
                    device = torch.device(env_device)
                else:
                    device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')

        self.training_parameters: TrainingParameters = TrainingParameters(
            num_epochs=num_epochs,
            mixed_precision_enabled=mixed_precision_enabled,
            gradient_update_frequency=gradient_update_frequency
        )
        self.workflow_options: WorkflowOptions = WorkflowOptions(
            logging_directory=logging_directory,
            device=device
        )
        self.runtime: Runtime = Runtime()

    def __repr__(self) -> str:
        return pformat(vars(self), indent=3, width=1)