from utilities import log_and_print
from .callback import Callback


def update_best_so_far(epoch, best_so_far, dataset_name, split_name, output_name, category_name, category_value):
    """
    Update the best so far category update value
    :param epoch: the current epoch
    :param best_so_far: the store of the best so far values
    :param dataset_name: the dataset name
    :param split_name: the split name
    :param output_name: the output name
    :param category_name: the category name
    :param category_value: the value of the category
    :return: a tuple `epoch, None` if no previous data, else a tuple `epoch, value` representing the best category value
    """
    if dataset_name not in best_so_far:
        best_so_far[dataset_name] = {}
    if split_name not in best_so_far[dataset_name]:
        best_so_far[dataset_name][split_name] = {}
    if output_name not in best_so_far[dataset_name][split_name]:
        best_so_far[dataset_name][split_name][output_name] = {}
    if category_name not in best_so_far[dataset_name][split_name][output_name]:
        best_so_far[dataset_name][split_name][output_name][category_name] = (epoch, category_value)
        return epoch, category_value

    best_epoch, best_so_far_value = best_so_far[dataset_name][split_name][output_name][category_name]
    if best_so_far_value is None or category_value < best_so_far_value:
        best_so_far[dataset_name][split_name][output_name][category_name] = epoch, category_value

    return best_so_far[dataset_name][split_name][output_name][category_name]


class CallbackEpochSummary(Callback):
    """
    Summarizes the last epoch and display useful information such as metric per dataset/split
    """
    def __init__(self, logger=log_and_print, track_best_so_far=True):
        self.logger = logger
        self.track_best_so_far = track_best_so_far
        self.best_so_far = {}

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        last = history[-1]
        self.logger('epoch={}'.format(len(history) - 1))
        splits_time = []
        for dataset_name, history_dataset in last.items():
            for split_name, history_split in history_dataset.items():
                if len(history_split) == 0:
                    continue

                for output_name, outputs in history_split.items():
                    for category_name, category_value in outputs.items():
                        best_epoch, best_value = update_best_so_far(len(history) - 1, self.best_so_far, dataset_name, split_name, output_name, category_name, category_value)
                        self.logger('   {}/{}/{}, {}={} [best={}, epoch={}]'.format(dataset_name, split_name, output_name, category_name, category_value, best_value, best_epoch))

        if len(splits_time) > 0:
            times = ', '.join(splits_time)
            self.logger(f'   time: {times}')
