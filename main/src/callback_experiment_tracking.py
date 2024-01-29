import json
import os
from callbacks.callback import Callback
import logging
import subprocess


logger = logging.getLogger(__name__)


def run_command(args, cwd=None) -> str:
    try:
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=cwd
        )

        stdout, stderr = process.communicate()
        stdout = stdout.decode('utf-8').strip()
        stderr = stderr.decode('utf-8').strip()
    except Exception as e:
        return f'Command failed={args}, Exception={e}, cwd={cwd}'

    if len(stderr) == 0:
        return stdout
    else:
        return f'{stdout}\n<STDERR>\n{stderr}'


def get_git_revision():
    """
    Return the current git revision
    """
    here = os.path.abspath(os.path.dirname(__file__))
    return run_command(['git', 'rev-parse', 'HEAD'], cwd=here)


class CallbackExperimentTracking(Callback):
    """
    Record useful information to track the experiments (e.g., git hash, git status, python environment)
    so that we can easily reproduce the results
    """
    def __init__(self, output_dir_name='tracking'):
        super().__init__()
        self.output_dir_name = output_dir_name
        self.output_path = None

    def __call__(self, options, history, model, **kwargs):
        logger.info('CallbackExperimentTracking')
        if self.output_path is None:
            self.output_path = os.path.join(options.workflow_options.current_logging_directory, self.output_dir_name)
            os.makedirs(self.output_path, exist_ok=True)

        git_revision = get_git_revision()
        with open(os.path.join(self.output_path, 'git_revision.json'), 'w') as f:
            json.dump({'git_revision': git_revision}, f, indent=3)

        here = os.path.abspath(os.path.dirname(__file__))
        git_status = run_command(['git', 'status'], cwd=here)
        with open(os.path.join(self.output_path, 'git_status.txt'), 'w') as f:
            f.write(git_status)

        requirements = run_command(['pip', 'list', '--format', 'freeze'])
        with open(os.path.join(self.output_path, 'requirements.txt'), 'w') as f:
            f.write(requirements)
        logger.info('CallbackExperimentTracking done!')