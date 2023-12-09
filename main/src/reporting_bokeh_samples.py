import warnings
import math
import functools
import collections

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, TableColumn, DataTable, HTMLTemplateFormatter, Select, \
    CategoricalColorMapper, HoverTool, LinearColorMapper, ColorBar, FixedTicker, NumberFormatter
from bokeh.models.widgets import Panel, Div
from bokeh.plotting import figure
import numpy as np
from bokeh.transform import transform
from bokeh_ui import BokehUi
from data_category import DataCategory



def len_batch(batch):
    """

    Args:
        batch: a data split or a `collections.Sequence`

    Returns:
        the number of elements within a data split
    """
    if isinstance(batch, (collections.Sequence, torch.Tensor)):
        return len(batch)

    assert isinstance(batch, collections.Mapping), 'Must be a dict-like structure! got={}'.format(type(batch))

    for name, values in batch.items():
        if isinstance(values, (list, tuple)):
            return len(values)
        if isinstance(values, torch.Tensor) and len(values.shape) != 0:
            return values.shape[0]
        if isinstance(values, np.ndarray) and len(values.shape) != 0:
            return values.shape[0]
    return 0



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


class PanelDataSamplesTabular(BokehUi):
    """
    Display tabular data

    Configuration options:
        default/with_column_title_rotation: 1 or 0
    """
    def __init__(self, options, name, data, data_types, type_categories, title='Tabular'):
        with_column_title_rotation = int(safe_lookup(
            options.config,
            name,
            'default',
            'with_column_title_rotation',
            default='1')) > 0

        ui, self.table = process_data_samples__tabular(
            options,
            name,
            data,
            data_types,
            type_categories,
            title=title,
            with_column_title_rotation=with_column_title_rotation
        )
        super().__init__(ui)

    def update_data(self, options, name, data, data_types, type_categories):
        source = self.table.source

        # make sure we do not populate very large data to a table. Remove
        # the large datasets (if not, the data is sent to the client, which
        # may result in slow rendering time)
        data = filter_large_data(options, data)

        for key in source.column_names:
            if key not in data:
                # some data is missing, just do not update the table yet
                return
        self.table.source.update(data=data)


def process_data_samples__tabular(options, name, data, data_types, type_categories, title, with_column_title_rotation=True):
    """
    Create a tabular panel of the data & types

    Args:
        options:
        data: a dictionary of (key, values)
        data_types: a dictionary of (key, type) indicating special type of ``key``
        title: the title of the panel to be displayed

    Returns:
        a panel
    """
    image_style = """
                img
                { 
                    height:%dpx; 
                    width:%dpx;
                } 
                """ % (options.image_size, options.image_size)

    template = f"""
                <div>
                <style>
                {image_style}
                </style>
                <img
                    src=<%= value %>
                ></img>
                </div>
                """

    with_images = False
    columns = []
    for key in data.keys():
        type = data_types.get(key)
        if type is not None and 'BLOB_IMAGE' in type:
            with_images = True
            c = TableColumn(field=key, title=key, formatter=HTMLTemplateFormatter(template=template))
        else:
            type_category = type_categories[key]
            table_kargs = {}
            if type_category == DataCategory.Continuous:
                table_kargs['formatter'] = NumberFormatter(format='0,0[.000]')

            c = TableColumn(field=key, title=key, **table_kargs)
        columns.append(c)

    # filter out data
    filtered_data = filter_large_data(options, data)
    data_source = ColumnDataSource(filtered_data)

    if with_column_title_rotation:
        # custom CSS to slightly rotate the column header
        # and draw text outside their respective columns
        # to improve readability. TODO That could be improved
        div = Div(text=f"""
        <style>
        .trw_reporting_table_{name} .slick-header-column {{
                background-color: transparent;
                background-image: none !important;
                transform: 
                    rotate(-10deg)
              }}
    
        .bk-root .slick-header-column.ui-state-default {{
            height: 40px;
            overflow: visible;
            vertical-align: bottom;
            line-height: 4.4;
        }}
        </style>
        """)
        div.visible = False  # hide the div to avoid position issues
    else:
        div = None

    row_height = options.font_size
    if with_images:
        row_height = options.image_size

    data_table = DataTable(
        source=data_source,
        columns=columns,
        row_height=row_height,
        #fit_columns=True,
        css_classes=[f"trw_reporting_table_{name}"])

    columns = [data_table, div] if div is not None else [data_table]

    return Panel(child=column(*columns, sizing_mode='stretch_both'), title=title), data_table


def scatter(all_data, groups, scatter_name, type_category):
    if type_category in (DataCategory.DiscreteUnordered, DataCategory.DiscreteOrdered):
        # discrete types, without ordering. Split the data into sub-groups
        # need to use global `unique_values` so that the groups are synchronized
        scatter_values = all_data[scatter_name]
        unique_values = set(scatter_values)
        if type_category == DataCategory.DiscreteOrdered:
            unique_values = sorted(unique_values)

        final_groups = []
        for group in groups:
            data = np.asarray(all_data[scatter_name])[group]

            current_groups = []
            for id, value in enumerate(unique_values):
                indices = np.where(data == value)
                current_groups.append(group[indices])
            final_groups.append(current_groups)
        return final_groups, [{'value': value, 'type': 'discrete'} for value in unique_values]
    elif type_category == DataCategory.Continuous:
        all_scatter_values = np.asarray(all_data[scatter_name])[np.concatenate(groups)]
        value_min = all_scatter_values.min()
        value_max = all_scatter_values.max()
        return [[g] for g in groups], [
            {'value_min': value_min, 'value_max': value_max, 'type': 'continuous', 'scatter_name': scatter_name}]
    else:
        raise NotImplementedError()


def group_coordinate(options, data, group, group_layout_y, group_layout_x):
    """
    Create a coordinate system for each group, independently of the other groups
    """
    if group_layout_y['type'] == 'discrete' and group_layout_x['type'] == 'discrete':
        nb_columns = math.ceil(math.sqrt(len(group)))
        coords_x = (np.arange(len(group)) % nb_columns) * options.image_size
        coords_y = (np.arange(len(group)) // nb_columns) * options.image_size
        coords = np.stack((coords_x, coords_y), axis=1)
        return coords

    if group_layout_y['type'] == 'continuous' and group_layout_x['type'] == 'discrete':
        nb_rows = len(group)
        scatter_name = group_layout_y['scatter_name']
        coords_x = np.random.rand(nb_rows) * options.image_size * 0.75
        coords_y = np.asarray(data[scatter_name])[group]
        coords = np.stack((coords_x, coords_y), axis=1)
        return coords

    if group_layout_x['type'] == 'continuous' and group_layout_y['type'] == 'discrete':
        nb_rows = len(group)
        scatter_name = group_layout_x['scatter_name']
        coords_y = np.random.rand(nb_rows) * options.image_size * 0.75
        coords_x = np.asarray(data[scatter_name])[group]
        coords = np.stack((coords_x, coords_y), axis=1)
        return coords

    if group_layout_x['type'] == 'continuous' and group_layout_y['type'] == 'continuous':
        scatter_name_x = group_layout_x['scatter_name']
        scatter_name_y = group_layout_y['scatter_name']
        coords_x = np.asarray(data[scatter_name_x])[group]
        coords_y = np.asarray(data[scatter_name_y])[group]
        coords = np.stack((coords_x, coords_y), axis=1)
        return coords

    raise NotImplementedError()


def sync_groups_coordinates(options, data_x, data_y, groups, groups_layout_y, groups_layout_x):
    """
    Up to that point, each group had sample coordinates calculated independently from
    each other. Here consolidate all the subcoordinate system into a global one
    """
    if groups_layout_x[0]['type'] == 'discrete':
        for x in range(groups.shape[1] - 1):
            groups_at_x = np.concatenate(groups[:, x])
            min_value_for_next_group = data_x[groups_at_x].max() + options.image_size + \
                                       options.style.category_margin * options.image_size

            for y in range(groups.shape[0]):
                group = groups[y, x + 1]
                if len(group) == 0:
                    # there are no sample for this group, simply skip it
                    continue
                min_group = data_x[group].min()
                shift = max(min_value_for_next_group - min_group, 0)
                data_x[group] += shift
    elif groups_layout_x[0]['type'] == 'continuous':
        pass
    else:
        raise NotImplementedError()

    if groups_layout_y[0]['type'] == 'discrete':
        for y in range(groups.shape[0] - 1):
            groups_at_y = np.concatenate(groups[y, :])
            min_value_for_next_group = data_y[groups_at_y].max() + options.image_size + \
                                       options.style.category_margin * options.image_size

            for x in range(groups.shape[1]):
                group = groups[y + 1, x]
                if len(group) == 0:
                    # there are no sample for this group, simply skip it
                    continue
                min_group = data_y[group].min()
                shift = max(min_value_for_next_group - min_group, 0)
                data_y[group] += shift
    elif groups_layout_y[0]['type'] == 'continuous':
        pass
    else:
        raise NotImplementedError()


def filter_large_data(options, data):
    """
    remove very large data, this will be communicated to the client and this is
    very slow!
    """
    filtered_data = collections.OrderedDict()
    for name, values in data.items():
        if isinstance(values[0], np.ndarray) and values[0].size > options.data_samples.max_numpy_display:
            # data is too large and take too long to display so remove it
            filtered_data[name] = ['...'] * len(values)
            warnings.warn(f'numpy array ({name}) is too large ({values[0].size}) and has been discarded')
            continue
        filtered_data[name] = values
    return filtered_data


def render_data(
        options, data, data_types, type_categories, scatter_x, scatter_y, binning_x_axis,
        binning_selection_named, binning_selection_integral, color_by, color_scheme,
        icon, label_with, layout):
    previous_figure = None

    while len(layout.children) >= 2:
        layout.children.pop()

    binning_selection_named.visible = False
    binning_selection_integral.visible = False
    if len(binning_x_axis.value) > 0 and binning_x_axis.value != 'None':
        scatter_values = data[binning_x_axis.value]
        unique_values = set(list(scatter_values))
        unique_values = list(sorted(unique_values))

        # update the binning selection tool
        binning_type = type_categories.get(binning_x_axis.value)
        selection = 'None'
        if binning_type == DataCategory.DiscreteUnordered or binning_type == DataCategory.DiscreteOrdered:
            binning_selection_named.visible = True
            binning_selection_integral.visible = False

            selection_values = ['None'] + [str(v) for v in unique_values]
            binning_selection_named.options = selection_values[::-1]  # reversed (e.g., last epoch is first choice)
            if binning_selection_named.value not in binning_selection_named.options:
                binning_selection_named.value = binning_selection_named.options[0]
            selection = binning_selection_named.value
        elif binning_type == DataCategory.Continuous:
            raise NotImplementedError()

        if selection == 'None':
            groups = []
            for value in unique_values:
                group = np.where(np.asarray(scatter_values) == value)
                groups.append((f'group={value}, ', group))
        else:
            # need to convert to original data type
            value = scatter_values[0]
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)
            selection_value = np.asarray(selection, dtype=value.dtype)
            indices = np.where(np.asarray(scatter_values) == selection_value)
            groups = [(f'group={selection}, ', indices)]
    else:
        nb_data = len_batch(data)
        groups = [('', [np.arange(nb_data)])]

    # remove very large data, this will be communicated to the client and this is
    # very slow!
    data = filter_large_data(options, data)

    x_range = None
    y_range = None
    figs = []
    for group_n, (group_name, group) in enumerate(groups):
        # copy the data: there were some display issues if not
        subdata = {name: list(np.asarray(value)[group[0]]) for name, value in data.items()}
        data_source = ColumnDataSource(subdata)
        nb_data = len(group[0])
        group = [np.arange(nb_data)]

        layout_children = render_data_frame(
            group_name, f'fig_{group_n}', options, data_source,
            subdata,
            group, data_types, type_categories, scatter_x,
            scatter_y, color_by, color_scheme, icon, label_with, previous_figure)

        if x_range is None:
            x_range = layout_children[0].children[0].x_range
            y_range = layout_children[0].children[0].y_range
        else:
            layout_children[0].children[0].x_range = x_range
            layout_children[0].children[0].y_range = y_range

        for c in layout_children:
            figs.append(c)

    layout.children.append(row(*figs, sizing_mode='stretch_both'))


def render_data_frame(fig_title, fig_name, options, data_source, data, groups, data_types, type_categories, scatter_x,
                      scatter_y, color_by, color_scheme, icon, label_with, previous_figure):
    print('render command STARTED, nb_samples=', len(groups[0]))
    # re-create a new figure each time... there are too many bugs currently in bokeh
    # to support dynamic update of the data
    fig = prepare_new_figure(options, data, data_types)
    layout_children = [row(fig, sizing_mode='stretch_both')]

    nb_data = len_batch(data)
    fig.title.text = f'{fig_title}Samples selected: {len(groups[0])}'
    fig.renderers.clear()
    fig.yaxis.visible = False
    fig.xaxis.visible = False

    data_x_name = f'{fig_name}_data_x'
    data_y_name = f'{fig_name}_data_y'

    if data_x_name not in data_source.column_names:
        data_source.add(np.zeros(nb_data, np.float32), data_x_name)
        data_source.add(np.zeros(nb_data, np.float32), data_y_name)
        data_source.add([None] * nb_data, 'color_by')

    fig.xaxis.axis_label = scatter_x.value
    fig.yaxis.axis_label = scatter_y.value

    if len(scatter_x.value) > 0 and scatter_x.value != 'None':
        scatter_x = scatter_x.value
        assert scatter_x in data, f'column not present in data! c={scatter_x}'
    else:
        scatter_x = None

    if len(scatter_y.value) > 0 and scatter_y.value != 'None':
        scatter_y = scatter_y.value
        assert scatter_y in data, f'column not present in data! c={scatter_y}'
    else:
        scatter_y = None

    if scatter_x is not None:
        groups, groups_layout_x = scatter(
            data,
            groups,
            scatter_x,
            type_category=type_categories[scatter_x])
        groups = groups[0]
    else:
        # consider all the points as x
        groups_layout_x = [{'value': '', 'type': 'discrete'}]

    if scatter_y is not None:
        groups, groups_layout_y = scatter(
            data,
            groups,
            scatter_y,
            type_category=type_categories[scatter_y])
    else:
        # consider all the points as y
        groups = [[g] for g in groups]
        groups_layout_y = [{'value': '', 'type': 'discrete'}]

    if len(color_by.value) > 0 and color_by.value != 'None':
        color_by = color_by.value
    else:
        color_by = None

    if color_by is not None and type_categories[color_by] == DataCategory.Continuous:
        color_scheme.visible = True
    else:
        color_scheme.visible = False

    # in case we have array of 3D instead of 2D+list (if samples have exactly the same in the 2 groups)
    # we need to transpose only the first and second but NOT the third
    groups = np.asarray(groups)
    groups_transpose = list(range(len(groups.shape)))
    tmp = groups_transpose[0]
    groups_transpose[0] = groups_transpose[1]
    groups_transpose[1] = tmp
    groups = groups.transpose(groups_transpose)

    data_x = np.zeros(nb_data, dtype=np.float)
    data_y = np.zeros(nb_data, dtype=np.float)
    for y in range(groups.shape[0]):
        for x in range(groups.shape[1]):
            group = groups[y, x]
            group_layout_y = groups_layout_y[y]
            group_layout_x = groups_layout_x[x]

            c = group_coordinate(options, data, group, group_layout_y, group_layout_x)
            data_x[group] = c[:, 0]
            data_y[group] = c[:, 1]

    sync_groups_coordinates(options, data_x, data_y, groups, groups_layout_y, groups_layout_x)

    data_source.data[data_x_name] = data_x
    data_source.data[data_y_name] = data_y

    if color_by is not None:
        # prepare the colors
        if 'discrete' in type_categories[color_by].value:
            colors_all_values = np.asarray(data[color_by], dtype=str)
            color_unique_values = list(set(colors_all_values))
            if options.style.sorted_legend:
                color_unique_values = sorted(color_unique_values)
            color_unique_palette_ints = np.random.randint(0, 255, [len(color_unique_values), 3])
            color_unique_palette_hexs = ['#%0.2X%0.2X%0.2X' % tuple(c) for c in color_unique_palette_ints]

            color_mapper = CategoricalColorMapper(
                factors=color_unique_values,  # must have string
                palette=color_unique_palette_hexs)

            # could not use the simple legend mechanism (was messing up the figure.
            # Instead resorting to another figure). This has the added benefit that we can
            # zoom in and out of the legend in case we have many categories
            sub_fig = figure(
                title='Labels',
                width=options.style.tool_window_size_x, height=options.style.tool_window_size_y,
                x_range=(-0.1, 1), y_range=(0, 26),
                tools='pan,reset',
                toolbar_location='above')
            y_text = 25 - np.arange(0, len(color_unique_values))
            sub_fig.circle(x=0, y=y_text, size=15, color=color_unique_palette_hexs)
            sub_fig.axis.visible = False
            sub_fig.xgrid.visible = False
            sub_fig.ygrid.visible = False
            sub_fig.text(x=0, y=y_text, text=color_unique_values, x_offset=15, y_offset=6 // 2, text_font_size='6pt')
            sub_fig.sizing_mode = 'fixed'
            layout_children.append(sub_fig)

        elif 'continuous' in type_categories[color_by].value:
            colors_all_values = np.asarray(data[color_by])
            palette = color_scheme.value
            min_value = colors_all_values.min()
            max_value = colors_all_values.max()
            color_mapper = LinearColorMapper(palette=palette, low=min_value, high=max_value)

            bar = ColorBar(
                title=color_by,
                label_standoff=10,
                title_standoff=10,
                color_mapper=color_mapper,
                location=(0, 0),
                ticker=FixedTicker(ticks=np.linspace(min_value, max_value, 5))
            )
            fig.add_layout(bar, "right")

        else:
            raise NotImplementedError()

        data_source.data['color_by'] = colors_all_values
        fill_color = transform('color_by', color_mapper)
    else:
        data_source.data['color_by'] = [None] * nb_data
        fill_color = 'blue'

    if len(icon.value) > 0 and (icon.value != 'Icon' and icon.value != 'Dot'):
        # the images do NOT support tooltips yet in bokeh, instead, overlay a transparent rectangle
        # that will display the tooltip
        units = 'data'
        if 'continuous' in [groups_layout_y[0]['type'], groups_layout_x[0]['type']]:
            # make the size of the image fixed so that when we have
            # many points close by, we can zoom in to isolate the samples
            units = 'screen'
        else:
            # keep aspect ration ONLY when we display collection of
            # images. If we have continuous axis, we do NOT want to
            # constrain the value of the axis by another axis
            fig.match_aspect = True

        fig.image_url(url=icon.value, x=data_x_name, y=data_y_name,
                      h=options.image_size, h_units=units,
                      w=options.image_size, w_units=units,
                      anchor="center", source=data_source)

        fig.rect(x=data_x_name, y=data_y_name, width=options.image_size, height=options.image_size, width_units=units,
                 height_units=units, source=data_source, fill_alpha=0, line_alpha=0)

        if color_by is not None:
            fig.rect(
                source=data_source,
                x=data_x_name,
                y=data_y_name,
                width=options.image_size - options.style.color_by_line_width,  # TODO data space mixed with screen space
                height=options.image_size - options.style.color_by_line_width,
                line_color=fill_color,
                fill_color=None,
                line_width=options.style.color_by_line_width,
                width_units=units, height_units=units,
            )

    elif icon.value == 'Icon':
        fig.oval(x=data_x_name, y=data_y_name, width=options.image_size, height=options.image_size, width_units='data',
                 height_units='data', source=data_source, fill_color=fill_color)
    elif icon.value == 'Dot':
        fig.circle(x=data_x_name, y=data_y_name, size=5, source=data_source, line_color='black', fill_color=fill_color)
    else:
        raise NotImplementedError()

    if scatter_x and groups_layout_x[0]['type'] == 'discrete':
        ticks = []
        ticks_label = []
        for x in range(groups.shape[1]):
            coords_x = data_x[np.concatenate(groups[:, x])]
            if len(coords_x) > 0:
                min_value = coords_x.min()
                max_value = coords_x.max()
                center_value = int(min_value + max_value) // 2

                ticks += [center_value]
                ticks_label += [groups_layout_x[x]['value']]

        fig.xaxis.visible = True
        fig.xaxis.ticker = ticks
        fig.xaxis.major_label_overrides = {t: str(name) for t, name in zip(ticks, ticks_label)}
        fig.xaxis.major_label_orientation = math.pi / 4
    elif scatter_x and groups_layout_x[0]['type'] == 'continuous':
        fig.xaxis.visible = True
    else:
        fig.xaxis.visible = False

    if scatter_y and groups_layout_y[0]['type'] == 'discrete':
        ticks = []
        ticks_label = []
        for y in range(groups.shape[0]):
            coords_y = data_y[np.concatenate(groups[y, :])]
            if len(coords_y) > 0:
                min_value = coords_y.min()
                max_value = coords_y.max()
                center_value = int(min_value + max_value) // 2

                ticks += [center_value]
                ticks_label += [groups_layout_y[y]['value']]

        fig.yaxis.visible = True
        fig.yaxis.ticker = ticks
        fig.yaxis.major_label_overrides = {t: str(name) for t, name in zip(ticks, ticks_label)}
        fig.yaxis.major_label_orientation = math.pi / 4
    elif scatter_y and groups_layout_y[0]['type'] == 'continuous':
        fig.yaxis.visible = True
    else:
        fig.yaxis.visible = False

    print('render command DONE')
    return layout_children


def make_custom_tooltip(options, data, data_types):
    tips = []
    for key in data.keys():
        # create a custom tooltip to display images
        t = data_types.get(key)
        if t is not None and 'BLOB_IMAGE' in t:
            tip = f"""
            <div style="opacity: 1.0;">
                <img
                    src="@{key}" width="100%"
                    border="2"
                </img>
                <span>{key}</span>
            </div>
            """
            tips.append(tip)
        else:
            tips.append(f'<div><span style="">{key}: @{key}</span></div>')

    # this style make sure that a single tooltip is displayed at once. If
    # many samples overlap, it may be really slow to render these
    # and they will be out of the screen anyway
    div_style = """
    <style>
        .bk-tooltip>div:not(:first-child) {display:none;}
    </style>
    """
    return '<div>' + div_style + '\n'.join(tips) + '</div>'


def prepare_new_figure(options, data, data_types):
    tools = 'pan,wheel_zoom,reset'
    f = figure(
        title='',
        tools=tools,
        active_scroll='wheel_zoom',
        toolbar_location='above',
        height=900,
        width=900,
        name='data_samples_fig'
    )

    hover_tool = HoverTool(tooltips=make_custom_tooltip(options, data, data_types))
    f.add_tools(hover_tool)

    f.xgrid.visible = False
    f.ygrid.visible = False
    return f


class PanelDataSamplesScatter(BokehUi):
    def __init__(self, options, name, data, data_types, type_categories):
        self.scatter_x_axis = Select(title='Scatter X Axis', name='scatter_x_axis')
        self.scatter_y_axis = Select(title='Scatter Y Axis', name='scatter_y_axis')
        self.color_by = Select(title='Color by', name='color_by')
        self.color_scheme = Select(title='Color scheme', name='color_scheme')
        self.binning_x_axis = Select(title='Binning X Axis', name='binning_x_axis')

        self.binning_selection_named = Select(title='Binning selection', name='binning_selection_named')
        self.binning_selection_named.visible = True

        self.binning_selection_integral = Select(title='Binning selection', name='binning_selection_integral')
        self.binning_selection_integral.visible = False

        self.label_with = Select(title='Label with', name='label_with')
        self.icon = Select(title='Display with', name='icon')

        controls_list = [
            self.scatter_x_axis,
            self.scatter_y_axis,
            self.color_by,
            self.color_scheme,
            self.binning_x_axis,
            self.binning_selection_named,
            self.binning_selection_integral,
            self.label_with,
            self.icon
        ]
        controls = column(controls_list, name='PanelDataSamplesScatter_controls')
        controls.sizing_mode = "fixed"

        self.update_controls(options, name, data, data_types, type_categories)

        layout = row(controls, sizing_mode='stretch_both')

        self.last_data = data
        self.last_data_types = data_types
        self.last_type_categories = type_categories

        update_partial = functools.partial(
            render_data,
            options=options,
            scatter_x=self.scatter_x_axis,
            scatter_y=self.scatter_y_axis,
            binning_x_axis=self.binning_x_axis,
            binning_selection_named=self.binning_selection_named,
            binning_selection_integral=self.binning_selection_integral,
            color_by=self.color_by,
            color_scheme=self.color_scheme,
            icon=self.icon,
            label_with=self.label_with,
            layout=layout,
        )

        for control in controls_list:
            control.on_change('value', lambda attr, old, new: self._update())

        self.refresh_view_fn = update_partial
        self._update()

        super().__init__(ui=Panel(child=layout, title='Scatter'))

    def _update(self):
        # record the last used values. If we keep
        # only the last n rows of data, we need to
        # keep these variables updated!

        self.refresh_view_fn(
            data=self.last_data,
            data_types=self.last_data_types,
            type_categories=self.last_type_categories
        )

    def update_controls(self, options, name, data, data_types, type_categories):
        data_names = list(sorted(data.keys()))
        values = ['None'] + data_names
        values_integral = ['None'] + [n for n in data_names if type_categories[n] in (
        DataCategory.DiscreteUnordered, DataCategory.DiscreteOrdered)]

        def populate_default(control, values, control_name, default='None'):
            control.options = values
            if control.value == '':
                default = safe_lookup(options.config, name, 'default', control_name, default=default)
                control.value = default

        populate_default(self.scatter_x_axis, values, 'Scatter X Axis')
        populate_default(self.scatter_y_axis, values, 'Scatter Y Axis')
        populate_default(self.color_by, values, 'Color by')
        populate_default(self.color_scheme, ['Viridis256', 'Magma256', 'Greys256', 'Turbo256'], 'Color by',
                         default='Viridis256')
        populate_default(self.binning_x_axis, values_integral, 'Binning X Axis')
        populate_default(self.label_with, values, 'Label with')
        populate_default(self.binning_selection_integral, [], 'Binning selection')
        populate_default(self.binning_selection_named, [], 'Binning selection')

        default_icon = safe_lookup(options.config, name, 'default', 'Display with', default='None')
        icon_values = [v for v, t in data_types.items() if 'BLOB_IMAGE' in t] + ['Icon', 'Dot']
        if default_icon != 'None':
            if default_icon in icon_values:
                del icon_values[icon_values.index(default_icon)]
                icon_values.insert(0, default_icon)
        if self.icon.value == '':
            self.icon.value = icon_values[0]

        self.icon.options = icon_values

    def update_data(self, options, name, data, data_types, type_categories):
        self.last_data = data
        self.last_data_types = data_types
        self.last_type_categories = type_categories
        self._update()


def process_data_samples(options, name, role, data, types, type_categories):
    tabs = []
    if options.data_samples.display_tabular:
        panel = PanelDataSamplesTabular(options, name, data, types, type_categories)
        tabs.append(panel)

    if options.data_samples.display_scatter and options.embedded:
        # here we require some python logic, so we need to have a bokeh
        # server running to y_axis this view
        panel = PanelDataSamplesScatter(options, name, data, types, type_categories)
        tabs.append(panel)

    return tabs


def process_data_tabular(options, name, role, data, types, type_categories):
    tabs = []
    panel = PanelDataSamplesTabular(options, name, data, types, type_categories, title=name)
    tabs.append(panel)
    return tabs

