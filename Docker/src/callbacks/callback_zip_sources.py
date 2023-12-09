from .callback import Callback
import logging
import os
import zipfile
import pprint
import io


logger = logging.getLogger(__name__)


def zip_sources(roots, location, extensions, exclusions):
    """
    Zip the content of `roots` folders with given extensions to a location
    """
    with zipfile.ZipFile(location, 'w') as f:
        for root in roots:
            root_for_zip = os.path.join(root, '..')
            for dirpath, dirnames, filenames in os.walk(root):
                exclude = False
                for e in exclusions:
                    if e in dirpath:
                        exclude = True
                        break
                if exclude:
                    continue

                files = []
                for filename in filenames:
                    _, extension = os.path.splitext(filename)
                    if extension not in extensions:
                        continue
                    files.append(filename)

                for file in files:
                    full_path = os.path.join(dirpath, file)
                    f.write(full_path, arcname=os.path.relpath(full_path, root_for_zip))


class CallbackZipSources(Callback):
    """
    Record important info relative to the training such as the sources & configuration info

    This is to make sure a result can always be easily reproduced. Any configuration option
    can be safely appended in options.runtime
    """
    def __init__(
            self,
            folders_to_record,
            extensions=('.py', '.sh', '.bat', '.json'),
            filename='sources.zip',
            max_width=200,
            exclusions=('.mypy_cache', '.svn', '.git', '__pycache__')):

        if not isinstance(folders_to_record, list):
            folders_to_record = [folders_to_record]

        self.folders_to_record = folders_to_record
        self.extensions = extensions
        self.filename = filename
        self.max_width = max_width
        self.exclusions = exclusions

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch,
                 **kwargs):
        logging.info(f'CallbackZipSources, folders={self.folders_to_record}')
        source_zip_path = os.path.join(options.workflow_options.current_logging_directory, self.filename)
        zip_sources(self.folders_to_record, source_zip_path, extensions=self.extensions, exclusions=self.exclusions)

        stream = io.StringIO()
        pprint.pprint(options, stream=stream, width=self.max_width)
        logger.info(f'options=\n{stream.getvalue()}')

        logger.info('CallbackZipSources successfully done!')

