import collections
import time

from bokeh.layouts import column, row, gridplot
from bokeh.models import Panel, CheckboxGroup, PreText, RadioGroup, ColumnDataSource, HoverTool
from bokeh.palettes import Category20
from bokeh.plotting import figure


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

def get_batch_n(split, nb_samples, indices, transforms, use_advanced_indexing):
    """
    Collect the split indices given and apply a series of transformations
    Args:
        nb_samples: the total number of samples of split
        split: a mapping of `np.ndarray` or `torch.Tensor`
        indices: a list of indices as numpy array
        transforms: a transformation or list of transformations or None
        use_advanced_indexing: if True, use the advanced indexing mechanism else
            use a simple list (original data is referenced)
            advanced indexing is typically faster for small objects, however for large objects (e.g., 3D data)
            the advanced indexing makes a copy of the data making it very slow.
    Returns:
        a split with the indices provided
    """
    data = {}
    for split_name, split_data in split.items():
        if isinstance(split_data, (torch.Tensor, np.ndarray)) and len(split_data) == nb_samples:
            # here we prefer [split_data[i] for i in indices] over split_data[indices]
            # this is because split_data[indices] will make a deep copy of the data which may be time consuming
            # for large data
            if use_advanced_indexing:
                split_data = split_data[indices]
            else:
                split_data = [[split_data[i]] for i in indices]
        if isinstance(split_data, list) and len(split_data) == nb_samples:
            split_data = [split_data[i] for i in indices]

        data[split_name] = split_data

    if transforms is None:
        # do nothing: there is no transform
        pass
    elif isinstance(transforms, collections.Sequence):
        # we have a list of transforms, apply each one of them
        for transform in transforms:
            data = transform(data)
    else:
        # anything else should be a functor
        data = transforms(data)

    return data

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


from bokeh_ui import BokehUi
from data_category import DataCategory
import numpy as np
from reporting_bokeh_samples import make_custom_tooltip


def process_data_graph(options, name, role, data, types, type_categories):
    tabs = []
    panel = PanelDataGraph(options, name, data, types, type_categories)
    tabs.append(panel)
    return tabs


def prepare_new_figure(options, data, data_types, xaxis_name, yaxis_name):
    tools = 'pan,wheel_zoom,reset'
    f = figure(
        tools=tools,
        active_scroll='wheel_zoom',
        toolbar_location='above',
        height=300,
        width=200,
        name='data_samples_fig'
    )

    hover_tool = HoverTool(tooltips=make_custom_tooltip(options, data, data_types))
    f.add_tools(hover_tool)

    f.xaxis.axis_label = xaxis_name
    f.yaxis.axis_label = yaxis_name
    return f


def hash_data_attributes(data, keys):
    """
    Create a hashing on the `keys` variables. Each hash uniquely identify a key value combination

    Args:
        data: the data
        keys: the keys of the data to be considered unique

    Returns:
        list of group ids
    """
    # create a hashing on the `group_by` variables
    group_by_sets = {}
    previous_size = 1
    for name in keys:
        names_set = set(data[name])

        names_value = {}
        for index, name_set in enumerate(names_set):
            names_value[name_set] = previous_size * index

        group_by_sets[name] = names_value
        previous_size *= len(names_set)

    # hash the group_by attributes
    nb_data = len_batch(data)
    groups = np.zeros([nb_data], dtype=np.uint64)
    for group_by_name in keys:
        values_orig = data[group_by_name]
        tsl_dict = group_by_sets[group_by_name]
        values_tsl = np.asarray([tsl_dict[i] for i in values_orig], dtype=np.uint64)
        groups += values_tsl
    return groups


def factorize_names(list_of_list_of_names):
    """
    Factorize the list of list of names so that we can extract
    the common parts and return the variable parts of the name

    Args:
        list_of_list_of_names: a list of [name1, name2, ..., nameN]

    Returns:

    """
    if len(list_of_list_of_names) == 0:
        return None, None

    factorized = []
    list_of_names = []

    rotated_list_of_names = list(zip(*list_of_list_of_names))
    for values in rotated_list_of_names:
        unique_values = set(values)
        if len(unique_values) == 1:
            factorized.append(values[0])
        else:
            list_of_names.append(values)

    list_of_names = list(zip(*list_of_names))
    if len(list_of_names) == 0:
        list_of_names = ['']  # we MUST have hat least 1 name even if eveything is factorized

    assert len(list_of_list_of_names) == len(list_of_names)
    return factorized, list_of_names


class PanelDataGraph(BokehUi):
    def __init__(
            self,
            options,
            name,
            data,
            data_types,
            type_categories,
    ):
        """

        Configuration options:
            - default/X Axis: name
            - default/Y Axis: name
            - default/Group by: name
            - default/Exclude: name1;name2;...;nameN
            - default/discard_axis_x: name1;name2;...;nameN
            - default/discard_axis_y: name1;name2;...;nameN
            - default/discard_group_by: name1;name2;...;nameN
            - default/discard_exclude_max_values: number. the maximum number of excluded values for a category to be shown. If an
                exclusion category has more values, no values added for this category
            - default/number_of_columns: int

        Args:
            options:
            name:
            data:
            data_types:
            type_categories:
        """
        self.discard_group_by = safe_lookup(options.config, name, 'default', 'discard_group_by', default='').split(';')
        self.discard_axis_x = safe_lookup(options.config, name, 'default', 'discard_axis_x', default='').split(';')
        self.discard_axis_y = safe_lookup(options.config, name, 'default', 'discard_axis_y', default='').split(';')
        self.discard_exclude_max_values = int(safe_lookup(options.config, name, 'default', 'discard_exclude_max_values', default='15'))

        self.group_by = CheckboxGroup()
        self.y_axis = CheckboxGroup()
        self.x_axis = RadioGroup()
        self.exclude = CheckboxGroup()

        controls_list = [
            PreText(text='Group by:'),
            self.group_by,
            PreText(text='Y Axis:'),
            self.y_axis,
            PreText(text='X Axis:'),
            self.x_axis,
            PreText(text='Exclude:'),
            self.exclude,
        ]
        controls = column(controls_list, name='PanelDataGraph_controls')
        controls.sizing_mode = "fixed"

        self.update_controls(options, name, data, data_types, type_categories)

        self.options = options
        self.layout = row(controls, sizing_mode='stretch_both')
        self.last_data = data
        self.last_data_types = data_types
        self.last_type_categories = type_categories
        self.last_figs_by_group = {}
        self.number_of_columns = int(safe_lookup(options.config, name, 'default', 'number_of_columns', default='2'))

        for control in controls_list:
            if hasattr(control, 'active'):
                control.on_change('active', lambda attr, old, new: self._update_and_clear_plots())

        self._update()

        super().__init__(ui=Panel(child=self.layout, title=name))

    def clear_graphs(self):
        self.last_figs_by_group = {}
        while len(self.layout.children) >= 2:
            # remove the previous graphs
            self.layout.children.pop()

    def render_data(self, options, data, data_types, type_categories, layout):
        time_start = time.perf_counter()

        if len(self.y_axis.active) == 0:
            # no axis selected, do not display anything
            self.clear_graphs()
            return

        # convert everything as np
        data_size = len_batch(data)
        data_np = collections.OrderedDict()
        for name, value in data.items():
            data_np[name] = np.asarray(value)

        group_by = [self.group_by.labels[active] for active in self.group_by.active]
        groups = hash_data_attributes(data_np, group_by)

        # extract the groups
        unique_groups = set(groups)
        figs = []

        figs_by_group = {}
        for y_axis_active in self.y_axis.active:
            axis_y = self.y_axis.labels[y_axis_active]
            for g in unique_groups:
                indices = np.where(groups == g)
                axis_x = self.x_axis.labels[self.x_axis.active]
                data_subset = get_batch_n(data_np, data_size, indices, transforms=None, use_advanced_indexing=True)

                # try to reuse the previous figure, instead of recreating a new one so as to
                # minimize the flickering and keep track of the zoom/pan
                fig = self.last_figs_by_group.get(g)
                if fig is None:
                    fig = prepare_new_figure(options, data, data_types, axis_x, axis_y)
                else:
                    # here we reuse a previous figure (with same hash)
                    pass

                fig = PanelDataGraph.create_graph(fig, options, data_subset, group_by, axis_x, axis_y, self.discard_group_by)
                figs.append(fig)
                figs_by_group[g] = fig

        if len(self.last_figs_by_group) != len(figs_by_group):
            # clear the graph only at the end to minimize flickering
            self.clear_graphs()

            # different groups, refresh the display in full
            self.last_figs_by_group = figs_by_group
            layout.children.append(gridplot(figs, ncols=self.number_of_columns, sizing_mode='stretch_width'))
        else:
            # reuse the figures to minimize flickering
            pass

        time_end = time.perf_counter()
        print('RENDER, time=', time_end - time_start)

    @staticmethod
    def update_figure_data(fig, data, color, group_index, line_options):
        """
        Purpose here is to directly refresh the figure as much as possible instead
        of creating new ones. Unfortunately, ``fig.renderers.clear()`` does not
        work in bokeh 2.0

        Instead, find the corresponding renderers and update its internal data.
        """
        if group_index < len(fig.renderers):
            # update existing glyphs
            fig.renderers[group_index].data_source.data.update(data)
        else:
            # create new glyphs
            fig.line(
                x='__x_values',
                y='__y_values',
                color=color,
                line_width=2,
                source=ColumnDataSource(data),
                **line_options)

    @staticmethod
    def create_graph(fig, options, data, group_by, axis_x, axis_y, discard_group_by):
        group_keys = set(data.keys()) - set(list(discard_group_by) + [axis_x, axis_y])
        groups = hash_data_attributes(data, group_keys)
        unique_groups = set(groups)

        indices_groups = []
        name_groups = []
        line_groups = []

        # prepare the group names and coordinates. We MUST do this
        # in two stages so that we can factorize the group names
        # in the figure name and leave the variable for the legend
        for g in zip(unique_groups):
            group_indices = np.where(groups == g)[0]
            if len(group_indices) == 0:
                # maybe due to excluded values, we don't have data
                continue

            name_parts = [data[k][group_indices[0]] for k in group_keys]  # group keys should have unique value (`[0]`)
            y_values = data[axis_y][group_indices]
            x_values = data[axis_x][group_indices]

            line_groups.append((y_values, x_values))
            name_groups.append(name_parts)
            indices_groups.append(group_indices)

        factorized, variables = factorize_names(name_groups)
        colors = list(Category20[20][::2]) + list(Category20[20][1::2])  # split the light and dark colors
        for index, ((y_values, x_values), name, color, indices) in enumerate(zip(line_groups, variables, colors, indices_groups)):
            line_options = {}
            if len(line_groups) != 1:
                # we do NOT need a legend: single group can always be factorized in the figure title
                line_options['legend_label'] = '/'.join(name)

            # copy the data to avoid modifying the original data
            data_cpy = collections.OrderedDict()
            for name, value in data.items():
                data_cpy[name] = value[indices]
            data_cpy['__x_values'] = x_values
            data_cpy['__y_values'] = y_values
            PanelDataGraph.update_figure_data(fig, data_cpy, color, index, line_options)

        if len(variables) <= 4:
            fig.legend.orientation = 'horizontal'
            fig.legend.spacing = 10  # add spacing, else the labels are too close!
        fig.legend.border_line_color = 'black'
        fig.legend.padding = 4

        fig.title.text = '/'.join(factorized)
        return fig

    def _update(self):
        self.render_data(
            options=self.options,
            layout=self.layout,
            data=self.last_data,
            data_types=self.last_data_types,
            type_categories=self.last_type_categories
        )

    def _update_and_clear_plots(self):
        """
        If we have large changes (e.g., new groups), clear all previous figures
        """
        self.clear_graphs()
        self._update()

    def update_controls(self, options, name, data, data_types, type_categories):
        data_names = list(sorted(data.keys()))

        values_discrete = [
            n for n in data_names if type_categories[n] in (
                DataCategory.DiscreteUnordered,
                DataCategory.DiscreteOrdered) and n not in self.discard_group_by
        ]

        values_discrete_unordered = [
            n for n in data_names if type_categories[n] == DataCategory.DiscreteUnordered and
                                     n not in self.discard_group_by
        ]

        values_axis_x = [
            n for n in data_names if type_categories[n] in (
                DataCategory.Continuous,
                DataCategory.DiscreteOrdered) and n not in self.discard_axis_x
        ]

        values_axis_y = [
            n for n in data_names if type_categories[n] in (
                DataCategory.Continuous,
                DataCategory.DiscreteOrdered) and n not in self.discard_axis_y
        ]

        def populate_default(control, values, control_name, default_active_all=False):
            default = safe_lookup(options.config, name, 'default', control_name, default='')
            default_ticked_values = default.split(';')
            active = [values.index(default_ticked) for default_ticked in default_ticked_values if default_ticked in values]
            has_active = len(active) > 0
            if len(active) == 0 and default_active_all:
                active = list(range(len(values)))

            if isinstance(control, RadioGroup):
                if len(active) == 0:
                    active = [0]
                assert len(active) == 1, 'can only have a single value!'
                active = active[0]

            if has_active:
                control.update(labels=values, active=active)
            else:
                control.update(labels=values)

        populate_default(self.group_by, values_discrete, 'Group by')
        populate_default(self.y_axis, values_axis_y, 'Y Axis')
        populate_default(self.x_axis, values_axis_x, 'X Axis')

        values_exclude = []
        for exclude_column in values_discrete_unordered:
            values = set(data[exclude_column])
            if len(values) <= self.discard_exclude_max_values:
                values_exclude += list(values)
        populate_default(self.exclude, values_exclude, 'Exclude')

    def update_data(self, options, name, data, data_types, type_categories):
        self.last_data = data
        self.last_data_types = data_types
        self.last_type_categories = type_categories
        self._update()
