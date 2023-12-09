import collections
import logging


# this tag is used to specify the type of data stored on the local drive
# for example BLOB_NUMPY for numpy arrays or BLOB_IMAGE_PNG for PNG images
import sqlite3
import traceback

SQLITE_TYPE_PATTERN = '_type'


logger = logging.getLogger(__name__)


def table_create(cursor, table_name, name_type_list, primary_key):
    """
    Create a table

    Args:
        cursor: the cursor
        table_name: the name of the table
        name_type_list: a list of tuple (name, type)
        primary_key: feature index of the primary key or None
    """
    if primary_key is not None:
        assert primary_key < len(name_type_list), \
            f'primary key ({primary_key}) is outside the number of columns ({len(name_type_list)})!'
        name_type_list[primary_key] = name_type_list[primary_key] + ' PRIMARY KEY'

    table_definition = ', '.join(name_type_list)
    sql_command = f"CREATE TABLE '{table_name}' ({table_definition});"

    try:
        cursor.execute(sql_command)
    except sqlite3.OperationalError as e:
        print('``table_create`` failed. Exception=', e)
        print('SQL=\n', sql_command)


def table_insert(cursor, table_name, names, values):
    """
    Insert into an existing table

    Args:
        cursor: the cursor
        table_name: the name of the table
        names:  the names of the columns to insert
        values: the values of the columns to insert
    """
    assert len(values) > 0
    if isinstance(values[0], tuple):
        assert len(values[0]) == len(names)
        insert_fn = cursor.executemany
    else:
        assert len(values) == len(names)
        insert_fn = cursor.execute

    try:
        names_str = ','.join(names)
        values_str = ','.join(['?'] * len(names))
        sql_command = f"INSERT INTO '{table_name}' ({names_str}) VALUES ({values_str})"
        insert_fn(sql_command, values)
    except Exception as e:
        # catch the exception to add additional debug info
        logger.error(f'Table insertion failed with exception={e}. Keys={names}')
        logging.error(traceback.format_exc())
        raise e


def table_drop(cursor, table_name):
    """
    Delete a table

    Returns:
        ``True`` if table was removed, ``False`` else
    """
    sql_command = f"DROP TABLE '{table_name}';"

    try:
        cursor.execute(sql_command).fetchall()
    except sqlite3.OperationalError:
        return False
    return True


def table_truncate(cursor, table_name):
    """
    Clear all the rows of a table

    Returns:
        ``True`` if table was removed, ``False`` else
    """
    sql_command = f"DELETE FROM '{table_name}';"

    try:
        cursor.execute(sql_command).fetchall()
    except sqlite3.OperationalError:
        return False
    return True


def get_table_number_of_rows(cursor, table_name):
    """
    Return the number of rows of a table
    """
    sql_command = f"SELECT COUNT(*) FROM '{table_name}';"
    v = cursor.execute(sql_command).fetchall()
    assert len(v) == 1
    assert len(v[0]) == 1
    return v[0][0]


def get_tables_name_and_role(cursor):
    """
    Return all the table names and table role

    Args:
        cursor: the DB cursor

    Returns:
        a list of (table name, table role)
    """
    sql_command = f"SELECT name FROM sqlite_master WHERE type='table';"
    names = cursor.execute(sql_command).fetchall()

    name_roles = []
    for name in names:
        assert len(name) == 1, f'got={name}'
        name = name[0]
        if '_metadata' in name:
            continue

        name_metadata = name + '_metadata'
        sql_command = f"SELECT table_role FROM '{name_metadata}';"
        v = cursor.execute(sql_command).fetchall()
        assert len(v) == 1, f'got={v}, name={name}'
        name_roles.append((name, v[0][0]))

    return name_roles


def get_table_data(cursor, table_name, single_row=False, replace_none=False):
    """
    Extract all the data of the table

    Args:
        cursor: the DB cursor
        table_name: the name of the database
        single_row: if True, returns a single row
        replace_none: if True, replace all ``None`` values by empty string

    Returns:
        a dictionary of (name, values)
    """
    cursor = cursor.execute(f"select * from '{table_name}'")
    column_names = [column[0] for column in cursor.description]
    if single_row:
        rows = cursor.fetchone()
        if rows is None:
            return {}
        rows = [[value] for value in rows]  # must be a list!
    else:
        rows = cursor.fetchall()

    transpose = zip(*rows)
    d = collections.OrderedDict(zip(column_names, transpose))
    if replace_none:
        for column_name, column_values in d.items():
            column_values = list(column_values)
            for n, value in enumerate(column_values):
                if value is None:
                    column_values[n] = ''
            d[column_name] = column_values
    return d


def get_data_types_and_clean_data(data):
    """
    Remove the `type` columns (e.g., BLOB_NUMPY, BLOB_IMAGE_PNG) from the data and return
    column types.

    Args:
        data: a dictionary of (name, values)

    Returns:
        a dictionary of (name, type) for names that have specified type
    """
    types = {}
    type_columns = [c for c in data.keys() if SQLITE_TYPE_PATTERN in c]

    # remove the data type column, this is not useful
    for type_column in type_columns:
        types[type_column.replace(SQLITE_TYPE_PATTERN, '')] = data[type_column][0]
        del data[type_column]

    return types


def get_metadata_name(table_name):
    """
    Return the name of the table metadata for table ``table_name``
    """
    return table_name + '_metadata'


def table_add_columns(cursor, table_name, column_names):
    column_names_p = [f"'{n}'" for n in column_names]
    for c in column_names_p:
        sql_command = f"ALTER TABLE '{table_name}' ADD {c};"
        cursor.execute(sql_command)


class TableStream:
    """
    A SQLite table that can be streamed.

    Two tables will be created:

    1) in ``table_name``:
        - feature name with ``*_type`` will have SQLITE type ``type``

    2) in ``table_name``_metadata:
        - ``table_role``: the role of the table
        - ``table_preamble``: an explanation or complementary info of the table
    """

    def __init__(self, cursor, table_name, table_role, primary_key=None, table_preamble=''):
        self.name_type_list = None
        self.table_name = table_name
        self.cursor = cursor
        self.table_role = table_role
        self.primary_key = primary_key
        self.table_preamble = table_preamble

        try:
            # load the column names if the table already exists
            self.column_names = set(self.get_column_names())
        except sqlite3.OperationalError:
            # the table doesn't exist!
            self.column_names = set()

    def _create(self, table_name, name_type_list, table_role, primary_key=0):
        metadata_names = ['table_role', 'table_preamble']
        metadata_values = [table_role, self.table_preamble]
        blobs = ['table_role TEXT', 'table_preamble TEXT']
        content = []
        for id, (name, value) in enumerate(name_type_list):
            s = f'\'{name}\' {value}'  # we MUST have `'` around name so that names can be SQL keywords
            content.append(s)
            self.column_names.add(name)

        metadata_name = get_metadata_name(table_name)
        table_create(self.cursor, metadata_name, blobs, primary_key=None)
        table_insert(self.cursor, metadata_name, names=metadata_names, values=metadata_values)
        table_create(self.cursor, table_name, content, primary_key=primary_key)

        self.name_type_list = name_type_list

    def _insert(self, batch):
        names = batch.keys()
        names = [f"'{name}'" for name in names]
        values = batch.values()
        rotated_values = list(zip(*values))
        table_insert(self.cursor, self.table_name, names, rotated_values)

    def insert(self, batch):
        """
        Insert a batch of data to the table. If the table doesn't exist, it will be created.

        Args:
            batch: a dictionary like of names and values. All values must have the same length.
        """
        assert isinstance(batch, collections.Mapping), 'must be a dict like structure!'
        value_size = len(next(iter(batch.values())))
        for name, value in batch.items():
            assert hasattr(value, '__len__'), f'type={type(value)} has no __len__'
            assert len(value) == value_size, f'All values must have the same size! Got={len(value)}' \
                                             f' for name={name} expected={value_size}'

        # first we must test if the table exists already in the database
        sql_command = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.table_name}';"
        self.cursor.execute(sql_command)
        rows = self.cursor.fetchall()

        if len(rows) == 0:
            name_type_list = []
            for name, value in batch.items():
                value_type = batch.get(name + SQLITE_TYPE_PATTERN)
                if value_type is None:
                    value_type = 'TEXT'
                name_type_list.append((name, value_type))
            self._create(self.table_name, name_type_list, self.table_role, self.primary_key)
        else:
            # if it exists, make sure all the columns exist!
            missing_columns = set(batch.keys()) - self.column_names
            if len(missing_columns) > 0:
                table_add_columns(self.cursor, self.table_name, list(missing_columns))
                self.column_names = {*self.column_names, *missing_columns}

        self._insert(batch)

    def get_column_names(self):
        r = self.cursor.execute(f'select * from \'{self.table_name}\'')
        names = [n[0] for n in r.description]
        return names
