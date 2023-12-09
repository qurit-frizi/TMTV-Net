import functools
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.io import output_file, show, curdoc
import os

from bokeh.layouts import column
# from bokeh.models import Div , Panel
from tornado import web
from reporting_bokeh_graph import process_data_graph
from reporting_bokeh_samples import process_data_samples, process_data_tabular
from reporting_bokeh_tabs_dynamic_data import TabsDynamicData
from reporting_bokeh_tabs_dynamic_header import TabsDynamicHeader
import sqlite3
from bokeh.server.server import Server
import json
import numpy as np

from table_sqlite import get_metadata_name, get_table_data

"""
table roles:
    - ``data_samples``: extract of data with possibly model output. Data exploration + model output correlations
    - ``data_tabular``: present the SQL table as tabular data
    - ``data_graph``: draw graphs (e.g., trend, histogram)
    - ``alias##alias_name``: specify this table should be treated as an alias for another table (``alias_name``
"""

#
# TODO:
#   - improve the custom JS to give an angle on tabular headers. Problem when we have small number of columns
#
#


class Object:
    pass


def create_default_reporting_options(embedded=True, config={}):
    """
    config must follow this scheme:
    {
       `table_name`: {
           `option1`: `value1`
       }
    }

    The different options will depend on the table role.

    - for ALL tables:
        {n


          'data' : {
              'remove_columns': ['column_name1'],
              'subsampling_factor': 1.0,
              'keep_last_n_rows': 1000
        }

    - For role `data_samples`:
       {
          'default':
              'Scatter X Axis': value,
              'Scatter Y Axis': value,
              'Color by': value,
              'Color scheme': value,
              'Binning X Axis': value,
              'Binning Y Axis': value,
              'Label with': value,
              'Display with': value,
       }
    """
    o = Object()
    o.image_size = 80
    o.font_size = 19
    o.frame_size_x = None
    o.frame_size_y = 768

    o.data_samples = Object()
    o.data_samples.display_tabular = True
    o.data_samples.display_scatter = True
    o.data_samples.max_numpy_display = 10  # if array below this size, the content will be displayed
    o.data_samples.resize_heterogeneous_numpy = True  # if True, numpy arrays of different shape will be resized to common average size
    o.db_root = None
    o.embedded = embedded
    o.style = Object()
    o.style.color_by_line_width = 1
    o.style.scatter_aspect_ratio = 1.5
    o.style.tool_window_size_x = 200
    o.style.tool_window_size_y = 500
    o.style.sorted_legend = True
    o.style.category_margin = 0.2
    o.style.scatter_continuous_factor = 10

    o.data = Object()
    o.data.refresh_time = 5.0
    o.data.unpack_numpy_arrays_with_less_than_x_columns = 15
    o.data.types_to_discard = [
        np.dtype('|S1'),        # binary string
    ]

    o.config = config
    return o


def create_tables(name, role, doc, options, connection):
    """
    Create a UI from a SQL table and return a ``Panel``
    """
    if role == 'data_samples':
        print(f'create data_samples={name}')
        data_table = TabsDynamicData(doc, options, connection, name, role, creator_fn=process_data_samples)
        panel = Panel(child=data_table.get_ui(), title=name, name=f'data_samples_{name}_main_panel')
    elif role == 'data_tabular':
        print(f'create data_tabular={name}')
        data_table = TabsDynamicData(doc, options, connection, name, role, creator_fn=process_data_tabular)
        panel = data_table.get_ui()
    elif role == 'data_graph':
        print(f'create data_graph={name}')
        data_table = TabsDynamicData(doc, options, connection, name, role, creator_fn=process_data_graph)
        panel = data_table.get_ui()
    else:
        raise NotImplementedError(f'role not implemented={role}')
    assert isinstance(panel, Panel), f'unexpected type={type(panel)}'

    # handle the table preamble here
    metadata_name = get_metadata_name(name)
    metadata = get_table_data(connection, metadata_name)
    preamble = metadata.get('table_preamble')
    if preamble is not None and len(preamble[0]) > 0:
        # we have a preamble for our table. Add a ``Div`` widget to y_axis the preamble
        child = column(Div(text=preamble[0]), panel.child, sizing_mode='stretch_both')
        panel.update(child=child)

    print(f'table={name} created!')
    return panel


def report(sql_database_path, options, doc=None):
    """
    Generate the reporting from a SQL database and configuration.

    Args:
        sql_database_path: the path to the SQLite database
        options: the options to configure the different reporting views
        doc: a possibly existing Bokeh document
    Returns:
        a populated Bokeh document
    """
    options.db_root = sql_database_path
    connection = sqlite3.connect(sql_database_path)
    root = os.path.dirname(sql_database_path)

    if not options.embedded:
        output_file(os.path.join(root, 'index.html'), title=os.path.basename(root))

    if doc is None:
        doc = curdoc()
    doc.title = os.path.basename(root)
    tabs = TabsDynamicHeader(doc, options, connection, creator_fn=functools.partial(
        create_tables,
        doc=doc,
        options=options,
        connection=connection))

    if options.embedded:
        doc.add_root(tabs.get_ui())
    else:
        show(tabs)

    return doc


def run_server(
        path_to_db,
        options=create_default_reporting_options(embedded=True),
        show_app=True,
        handlers=[],
        port=5100):
    """
    Run a server

    Note:
        the .db will be replaced by .json if a file with this name is present,
        it will be imported and used as configuration

    Args:
        path_to_db: the path to the SQLite database
        options: configuration options
        show_app: if ``True``, the application will be started in an open browser
        handlers: additional handlers for the Bokeh application
        port: the port of the server
    """
    def fn(doc):
        return report(sql_database_path=path_to_db, options=options, doc=doc)

    app_1 = Application(FunctionHandler(fn), *handlers)

    if len(options.config) == 0:
        json_config = path_to_db.replace('.db', '.json')
        if os.path.exists(json_config):
            with open(json_config, 'r') as f:
                config = json.load(f)
                options.config = config

    # here we expect to have a folder with the following structure
    # root/{app_name}/static/{...} for all the static resources (e.g., images, numpy arrays)
    app_directory = os.path.dirname(path_to_db)
    app_name = os.path.basename(app_directory)
    apps = {f'/{app_name}': app_1}
    print('Starting reporting server:', apps, ' port=', port)
    server = Server(apps, num_procs=1, port=port)

    # set up the `static` mapping
    handlers = [
        (
            f'/{app_name}/static/(.*)',
            web.StaticFileHandler,
            {'path': os.path.join(app_directory, 'static')},
        )
    ]
    server._tornado.add_handlers(r".*", handlers)

    # start the server
    server.start()
    if show_app:
        server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


if __name__ == '__main__':
    config = {
        'history': {
            'default': {
                #'Group by': 'dataset;output;not here',
                'X Axis': 'epoch',
                'Y Axis': 'value',
            }
        },
        'samples': {
            'data': {
                'remove_columns': [
                    'sex',
                    # 'images'
                ],
                'keep_last_n_rows': 60,
                'subsampling_factor': 1.0,
            },
            'default': {
                # 'Scatter X Axis': 'diagnosis',
                # 'Scatter Y Axis': 'term_classification_output_raw_1',
                # 'Color by': 'term_classification_output',
                # 'Color by': 'term_classification_output',
                # 'Display with': 'Dot'
            }
        }
    }

    options = create_default_reporting_options(config=config)
    options.image_size = 64

    # run a static HTML page (i.e., the more complicated views requiring python v=callbacks will be disabled)
    report('/path/reporting_sqlite.db', options=create_default_reporting_options(embedded=False))

