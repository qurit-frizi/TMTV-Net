import functools
import collections
import sqlite3
import logging

from bokeh.models.widgets import Tabs
from bokeh_ui import BokehUi
from normalize_data import normalize_data
from table_sqlite import get_table_data, get_table_number_of_rows, get_tables_name_and_role, table_create, \
    get_metadata_name, table_insert

def safe_lookup(dictionary, *keys, default=None):
    """
    Recursively access nested dictionaries
    Args:
        dictionary: nested dictionary
        *keys: the keys to access within the nested dictionaries
        default: the default value if dictionary is ``None`` or it doesn't contain
            the keys
    Returns:
        None if we can't access to all the keys, else dictionary[key_0][key_1][...][key_n]
    """
    if dictionary is None:
        return default

    for key in keys:
        dictionary = dictionary.get(key)
        if dictionary is None:
            return default

    return dictionary


logger = logging.getLogger(__name__)


def create_aliased_table(connection, name, aliased_table_name):
    metadata_name = get_metadata_name(name)
    alias_role = f'alias##{aliased_table_name}'

    cursor = connection.cursor()
    table_create(cursor, metadata_name, ['table_role TEXT'], primary_key=None)
    table_insert(cursor, metadata_name, names=['table_role'], values=[alias_role])

    # create an empty table
    table_create(cursor, name, ['dummy_column TEXT'], primary_key=None)


def get_data_normalize_and_alias(options, connection, name):
    """

    Retrieve data from SQL, normalize the data and resolve aliasing (

    Returns:
        tuple (data, types, type_categories, alias)
    """
    table_name_roles = get_tables_name_and_role(connection.cursor())
    table_name_roles = dict(table_name_roles)
    role = table_name_roles.get(name)
    assert role, f'table={name} doesn\'t have a role!'

    if 'alias##' in role:
        splits = role.split('##')
        assert len(splits) == 2, 'alias is not well formed. Expeced ``alias##aliasname``'
        alias = splits[1]
        # in an aliased table, use the ``name`` to pickup the correct config from the ``options``
        data, types, type_categories = normalize_data(options, get_table_data(connection, alias), table_name=name)
    else:
        alias = None
        data, types, type_categories = normalize_data(options, get_table_data(connection, name), table_name=name)

    return data, types, type_categories, alias


class TableChangedDectector:
    """
    Detect when a table has been updated by watching at the number of rows, columns and content of specific
    rows (first).
    """
    def __init__(self, connection, table_name):
        self.table_name = table_name
        self.connection = connection
        self.last_number_of_rows = None
        self.first_row = {}

    def __call__(self):
        number_of_rows = get_table_number_of_rows(self.connection.cursor(), self.table_name)
        if number_of_rows != self.last_number_of_rows:
            #print(f'Table={self.table_name} Reason=different rows({number_of_rows}/{self.last_number_of_rows})')
            self.last_number_of_rows = number_of_rows
            self.first_row = get_table_data(self.connection.cursor(), self.table_name, single_row=True)
            return True

        rows = get_table_data(self.connection.cursor(), self.table_name, single_row=True)
        if number_of_rows > 0:
            if len(rows) != len(self.first_row):
                # different number of columns
                #print(f'Table={self.table_name} Reason=different columns ({self.first_row}/{rows})')
                self.first_row = rows
                return True
            for name, value in rows.items():
                # value is different!
                value_last = self.first_row.get(name)
                if value_last is None or value_last != value:
                    #print(f'Table={self.table_name} Reason=different value.\nprevious={self.first_row}\ngot={rows})')
                    self.first_row = rows
                    return True

        # can't find differences, return no changes
        return False


class TabsDynamicData(BokehUi):
    """
    Helper class to manage updates of the underlying SQL data for a given table
    """
    def __init__(self, doc, options, connection, name, role, creator_fn):
        self.table_changed = TableChangedDectector(connection, name)
        data, types, type_categories, alias = get_data_normalize_and_alias(options, connection, name)
        tabs = creator_fn(options, name, role, data, types, type_categories)
        tabs_ui = []

        for tab in tabs:
            assert isinstance(tab, BokehUi), 'must be a ``BokehUi`` based!'
            tabs_ui.append(tab.get_ui())

        if len(tabs_ui) > 1:
            ui = Tabs(tabs=tabs_ui)
        else:
            assert len(tabs_ui) > 0
            ui = tabs_ui[0]
        super().__init__(ui=ui)

        doc.add_periodic_callback(functools.partial(self.update,
                                                    options=options,
                                                    connection=connection,
                                                    name=name,
                                                    tabs=tabs),
                                  options.data.refresh_time * 1000)

        # at this point the UI is up to date so make sure we are with the detector too
        self.table_changed()

    def update(self, options, connection, name, tabs):
        try:
            table_changed = self.table_changed()
            if table_changed:  # different number of rows, data was changed!
                # discard `0`, the table creation is not part of a
                # transaction, the table is being populated
                number_of_rows = get_table_number_of_rows(connection, name)
                if number_of_rows > 0:
                    data, types, type_categories = normalize_data(options, get_table_data(connection, name), table_name=name)
                    keep_last_n_rows = safe_lookup(options.config, name, 'data', 'keep_last_n_rows')
                    if keep_last_n_rows is not None:
                        data_trimmed = collections.OrderedDict()
                        for name, values in data.items():
                            data_trimmed[name] = values[-keep_last_n_rows:]
                        data = data_trimmed

                    for tab in tabs:
                        tab.update_data(options, name, data, types, type_categories)
        except sqlite3.OperationalError as e:
            logger.warning(f'TabsDynamicData={self} could not be updated. Exception={e}. '
                           f'``database is locked`` can be ignored if another process is '
                           f'currently populating the database')
