import os
import logging
import copy
import matplotlib.pyplot as plt
from textwrap import wrap
import numpy as np
import math
import collections



def to_value(v):
    """
    Convert where appropriate from tensors to numpy arrays

    Args:
        v: an object. If ``torch.Tensor``, the tensor will be converted to a numpy
            array. Else returns the original ``v``

    Returns:
        ``torch.Tensor`` as numpy arrays. Any other type will be left unchanged
    """
    if isinstance(v, torch.Tensor):
        return v.cpu().data.numpy()
    return v


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
from .callback import Callback
import trainer
import utilities
import analysis_plots


logger = logging.getLogger(__name__)


class CallbackStopEpoch:
    """
    Utility callback counting the number of samples. When maximum is reached, stop the iteration
    """
    def __init__(self, nb_samples):
        self.nb_samples = nb_samples
        self.current_samples = 0

    def reset(self):
        self.current_samples = 0

    def __call__(self, dataset_name, split_name, batch):
        if self.current_samples >= self.nb_samples:
            raise StopIteration()
        nb_batch_samples = len_batch(batch)
        self.current_samples += nb_batch_samples


def plot_trend(
        export_path,
        lines_x,
        lines_y,
        title,
        xlabel,
        ylabel,
        y_scale='linear',
        x_scale='linear',
        maximum_chars_per_line=50,
        rotate_x=None,
        y_range=None,
        name_xy_markers=None):
    """
    Plot a graph defined by a list of x and y coordinates

    Args:
        export_path: folder where to export the figure
        lines_x: a list of x coordinates
        lines_y: a list of y coordinates
        title: the title of the figure
        xlabel: the label of axis x
        ylabel: the label of axis y
        y_scale: the scale of axis y
        x_scale: the scale of axis x
        maximum_chars_per_line: the maximum number of characters of the title per line
        rotate_x: if True, the rotation angle of the label of the axis x
        y_range: if not None, the (min, max) of the y-axis
        name_xy_markers: a dictionary (name, (x, y)) of markers to be displayed

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale(y_scale)
    ax.set_xscale(x_scale)
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.grid(which='both', axis='y', linestyle='--')
    ax.plot(lines_x, lines_y)
    ax.set_title('\n'.join(wrap(title, maximum_chars_per_line)), fontsize=20)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
        if rotate_x is not None:
            tick.label1.set_rotation(rotate_x)

    if name_xy_markers is not None:
        assert isinstance(name_xy_markers, collections.Mapping), 'must be a dictionary of name -> tuple (x, y)'
        xs = []
        ys = []
        for name, (x, y) in name_xy_markers.items():
            xs.append(x)
            ys.append(y)
        ax.scatter(xs, ys, color='red')

        for name, (x, y) in name_xy_markers.items():
            ax.annotate(name, (x, y))

    analysis_plots.fig_tight_layout(fig)
    analysis_plots.export_figure(export_path, title)


def default_identify_learning_rate_section(lines_x, lines_y, loss_ratio_to_discard=0.8):
    """
    Find a good section for the learning rate.

    Heuristic rules to find the best learning rate:

    1. worst loss is loss at epoch 0
    2. initially, the loss may not decrease due to small random variation, especially with small number of samples
       so tolerate that the initial LR may not be good
    3. after some epochs, the loss decrease to reach some minimum, then will increase significantly. Discard anything after this point
    4. find the LR achieving the minimum loss. This is our `optimal` LR
    """
    # first remove all the epochs after which the loss is greater than the initial loss
    starting_loss = lines_y[0]
    discard_test_check_before_epoch = 5
    loss_to_discard = loss_ratio_to_discard * starting_loss

    loss_index_too_large = np.where(lines_y > loss_to_discard)
    if len(loss_index_too_large) > 0:
        indices = loss_index_too_large[0]
        if indices[0] < discard_test_check_before_epoch:
            # we know initially the loss may decrease slowly, do not stop here! Discard
            # the beginning of this section
            relative_index = 0
            while True:
                if indices[relative_index + 1] != indices[relative_index] + 1:  # contiguous? if yes, discard!
                    break
                if relative_index + 1 >= len(indices):  # end of sequence
                    break
                relative_index += 1

            indices = indices[relative_index + 1:]

        for index in indices:
            if index > discard_test_check_before_epoch:
                lines_x = lines_x[: index + 1]
                lines_y = lines_y[: index + 1]
                break

    # look for the smallest loss
    best_loss_index = np.argmin(lines_y)
    index_to_keep = 0
    while True:
        current_loss = lines_y[best_loss_index + index_to_keep]
        if math.isnan(current_loss) or current_loss > starting_loss:
            break
        if best_loss_index + index_to_keep + 1 >= len(lines_y):
            break
        index_to_keep += 1

    lines_x = lines_x[: best_loss_index + index_to_keep + 1]
    lines_y = lines_y[: best_loss_index + index_to_keep + 1]
    return lines_x, lines_y


class CallbackLearningRateFinder(Callback):
    """
    Identify a good range for the learning rate parameter.

    See "Cyclical Learning Rates for Training Neural Networks", Leslie N. Smith. https://arxiv.org/abs/1506.01186

    Start from a small learning rate and every iteration, increase the learning rate by a factor. At the same time
    record the loss per epoch. Suitable learning rates will make the loss function decrease. We should select the
    highest learning rate which decreases the loss function.
    """
    def __init__(
            self,
            nb_samples_per_learning_rate=1000,
            learning_rate_start=1e-6,
            learning_rate_stop=1e1,
            learning_rate_mul=1.2,
            learning_rate_final_multiplier=0.8,
            dataset_name=None,
            split_name=None,
            dirname='lr_finder',
            identify_learning_rate_section=default_identify_learning_rate_section,
            set_new_learning_rate=False,
            param_maximum_loss_ratio=0.8):
        """

        Args:
            nb_samples_per_learning_rate: the number of samples used to calculate the loss for each learning rate tried
            learning_rate_start: the learning rate starting value
            learning_rate_stop: the learning rate stopping value. When the learning rate exceed this value,
                :class:`callbacks.CallbackLearningRateFinder` will be stopped
            learning_rate_mul: the learning rate multiplier for the next learning rate to be tested
            learning_rate_final_multiplier: often the best learning rate is too high for full convergence. If
                `set_new_learning_rate` is True, the final learning rate will be
                set to best_learning_rate * learning_rate_final_multiplier
            dataset_name: the dataset name to be used. If `None`, the first dataset will be used
            split_name: the split name to be used. If `None`, the default training split name will be used
            dirname: the directory where the plot will be exported
            identify_learning_rate_section: a function to identity a good region of learning rate. take as input (list of LR, list of loss)
            set_new_learning_rate: if True, the new learning calculated will replace the initial learning rate
            param_maximum_loss_ratio: if the loss reaches this peak, LR with greater loss this will be discarded
        """
        self.nb_samples_per_learning_rate = nb_samples_per_learning_rate
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.learning_rate_mul = learning_rate_mul
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.dirname = dirname
        self.identify_learning_rate_section = identify_learning_rate_section
        self.set_new_learning_rate = set_new_learning_rate
        self.param_maximum_loss_ratio = param_maximum_loss_ratio
        self.learning_rate_final_multiplier = learning_rate_final_multiplier

        assert learning_rate_start < learning_rate_stop
        assert learning_rate_start > 0
        assert learning_rate_mul > 1.0

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        """
        .. note:: The model will be deep copied so that we don't influence the training

        Args:
            **kwargs: required `optimizers_fn`
        """
        logger.info('started CallbackLearningRateFinder.__call__')
        device = options.workflow_options.device

        output_path = os.path.join(options.workflow_options.current_logging_directory, self.dirname)
        utilities.create_or_recreate_folder(output_path)

        if self.dataset_name is None:
            self.dataset_name = next(iter(datasets))

        if self.split_name is None:
            self.split_name = options.workflow_options.train_split

        logger.info('dataset={}, split={}, nb_samples={}, learning_rate_start={}, learning_rate_stop={}'.format(
            self.dataset_name,
            self.split_name,
            self.nb_samples_per_learning_rate,
            self.learning_rate_start,
            self. learning_rate_stop))

        callback_stop_epoch = CallbackStopEpoch(nb_samples=self.nb_samples_per_learning_rate)
        if callbacks_per_batch is not None:
            callbacks_per_batch = copy.copy(callbacks_per_batch)  # make sure these are only local changes!
            callbacks_per_batch.append(callback_stop_epoch)
        else:
            callbacks_per_batch = [callback_stop_epoch]

        optimizers_fn = kwargs.get('optimizers_fn')
        assert optimizers_fn is not None, '`optimizers_fn` can\'t be None!'
        split = datasets[self.dataset_name][self.split_name]

        lr_loss_list = []
        learning_rate = self.learning_rate_start
        model_copy = copy.deepcopy(model)
        while learning_rate < self.learning_rate_stop:
            # we do NOT want to modify our model or optimizer so make a copy
            # we restart from the original model to better isolate the learning rate effect
            #model_copy = copy.deepcopy(model)
            optimizers, _, _ = optimizers_fn(datasets, model_copy)
            optimizer = optimizers.get(self.dataset_name)
            assert optimizer is not None, 'optimizer can\'t be found for dataset={}'.format(self.dataset_name)
            utilities.set_optimizer_learning_rate(optimizer, learning_rate)

            callback_stop_epoch.reset()
            all_loss_terms = trainer.train_loop(
                options=options,
                device=device,
                dataset_name=self.dataset_name,
                split_name=self.split_name,
                split=split,
                optimizer=optimizer,
                per_step_scheduler=None,
                model=model_copy,
                loss_fn=losses[self.dataset_name],
                history=None,
                callbacks_per_batch=callbacks_per_batch,
                callbacks_per_batch_loss_terms=None)

            loss = 0.0
            for loss_batch in all_loss_terms:
                current_loss = to_value(loss_batch['overall_loss']['loss'])
                loss += current_loss
            loss /= len(all_loss_terms)

            lr_loss_list.append((learning_rate, loss))
            learning_rate *= self.learning_rate_mul

        lines_x = np.asarray(lr_loss_list)[:, 0]
        lines_y = np.asarray(lr_loss_list)[:, 1]

        # find the relevant section of LR
        logger.debug('Raw (LR, loss) by epoch:\n{}'.format(list(zip(lines_x, lines_y))))
        lines_x, lines_y = self.identify_learning_rate_section(lines_x, lines_y, loss_ratio_to_discard=self.param_maximum_loss_ratio)
        logger.debug('Interesting LR section (LR, loss) by epoch:\n{}'.format(list(zip(lines_x, lines_y))))

        # select the LR with the smallest loss
        min_index = np.argmin(lines_y)
        best_learning_rate = lines_x[min_index]
        best_loss = lines_y[min_index]

        # finally, export the figure
        y_range = (np.min(lines_y) * 0.9, lines_y[0] * 1.1)  # there is not point to display losses more than no training at all!
        plot_trend(
            export_path=output_path,
            lines_x=lines_x,
            lines_y=lines_y,
            title='Learning rate finder ({}, {})'.format(self.dataset_name, self.split_name),
            xlabel='learning rate',
            ylabel='overall_loss',
            y_range=y_range,
            x_scale='log',
            y_scale='log',
            name_xy_markers={'LR={}'.format(best_learning_rate): (best_learning_rate, best_loss)}
        )

        logger.info('best_learning_rate={}'.format(best_learning_rate))
        logger.info('best_loss={}'.format(best_loss))
        print('best_learning_rate=', best_learning_rate)
        print('best_loss=', best_loss)

        if self.set_new_learning_rate:
            best_learning_rate *= self.learning_rate_final_multiplier
            optimizers = kwargs.get('optimizers')
            if optimizers is not None:
                for optimizer_name, optimizer in optimizers.items():
                    logger.info('optimizer={}, changed learning rate={}'.format(optimizer_name, best_learning_rate))
                    utilities.set_optimizer_learning_rate(optimizer, best_learning_rate)
            else:
                logger.warning('No optimizers available in `kwargs`')

        logger.info('successfully finished CallbackLearningRateFinder.__call__')


