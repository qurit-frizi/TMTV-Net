import logging

from bokeh.application.handlers.lifecycle import LifecycleHandler
import multiprocessing as mp
from reporting_bokeh import create_default_reporting_options
from reporting_bokeh import run_server
from .callback import Callback

logger = logging.getLogger(__name__)


class ReportingLifeCycleHandler(LifecycleHandler):
    def __init__(self):
        super().__init__()
        self.active_sessions = 0

    @staticmethod
    def _do_nothing(ignored):
        pass

    async def on_session_created(self, server_context):
        self.active_sessions += 1
        logger.info(f'session created. Current active sessions={self.active_sessions}')
        return ReportingLifeCycleHandler._do_nothing

    async def on_session_destroyed(self, server_context):
        self.active_sessions -= 1
        logger.info(f'session destroyed. Current active sessions={self.active_sessions}')
        if self.active_sessions == 0:
            exit(0)
        return ReportingLifeCycleHandler._do_nothing


class CallbackReportingStartServer(Callback):
    def __init__(
            self,
            reporting_options=create_default_reporting_options(embedded=True, config={}),
            show_app=True,
            port=0,
            keep_alive_until_client_disconnect=True):
        self.server_process = None
        self.reporting_options = reporting_options
        self.show_app = show_app
        self.keep_alive_until_client_disconnect = keep_alive_until_client_disconnect
        self.port = port

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch,
                 **kwargs):

        if self.server_process is None:
            logger.info('creating reporting server...')

            if self.keep_alive_until_client_disconnect:
                handlers = [ReportingLifeCycleHandler()]
            else:
                handlers = []
            process = mp.Process(target=run_server,
                                 args=(options.workflow_options.sql_database_path,
                                       self.reporting_options,
                                       self.show_app,
                                       handlers,
                                       self.port))
            process.daemon = True
            self.server_process = process
            self.server_process.start()
            logger.info(f'creating reporting server Done! PID={process.pid}')

    def __del__(self):
        if self.server_process is not None and not self.keep_alive_until_client_disconnect:
            logger.info(f'terminating server...')
            process = self.server_process
            self.server_process = None
            process.terminate()
            process.join()
            logger.info(f'terminating server, success!')
