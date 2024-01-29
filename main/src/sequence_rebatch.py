from asyncio.log import logger
import time
from utils import len_batch
from sequence import Sequence, SequenceIterator, default_collate_list_of_dicts
import numpy as np
import torch
import collections


def split_in_2_batches(batch: collections.MutableMapping, first_batch_size: int):
    """
    Split a single batch into 2 batches. The first batch will have a fixed size.

    If there is not enough sample to split the batch, return (batch, None)

    Args:
        batch: the batch to split
        first_batch_size: the batch size of the first batch. The remaining samples will be in the second batch

    Returns:
        a tuple (first batch, second batch)
    """
    batch_size = len_batch(batch)
    if batch_size <= first_batch_size:
        return batch, None

    # actually, split!
    batch_1 = type(batch)()
    batch_2 = type(batch)()

    for name, value in batch.items():
        if isinstance(value, (np.ndarray, torch.Tensor, list)):
            # split the array/list
            batch_1[name] = value[:first_batch_size]
            batch_2[name] = value[first_batch_size:]
        else:
            # for other types, simply duplicate
            batch_1[name] = value
            batch_2[name] = value

    return batch_1, batch_2


class RebatchStatistics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.nb_batches_input = 0
        self.nb_batches_output = 0
        self.waiting_time_batches = 0
        self.waiting_time_rebatch = 0
        self.waiting_time_split = 0


class SequenceReBatch(Sequence, SequenceIterator):
    """
    This sequence will normalize the batch size of an underlying sequence

    If the underlying sequence batch is too large, it will be split in multiple batches. Conversely,
    if the size of the batch is too small, it several batches will be merged until we reach the expected batch size.
    """
    def __init__(self, source_split, batch_size, discard_batch_not_full=False, collate_fn=default_collate_list_of_dicts):
        """
        Normalize a sequence to identical batch size given an input sequence with varying batch size

        Args:
            source_split: the underlying sequence to normalize
            batch_size: the size of the batches created by this sequence
            discard_batch_not_full: if True, the last batch will be discarded if not full
            collate_fn: function to merge multiple batches
        """
        super().__init__(source_split)

        assert batch_size > 0
        assert isinstance(source_split, Sequence), '`source_split` must be a `Sequence`'
        self.source_split = source_split
        self.batch_size = batch_size
        self.discard_batch_not_full = discard_batch_not_full
        self.iter_source = None
        self.iter_overflow = None
        self.collate_fn = collate_fn
        self.statistics = RebatchStatistics()

    def subsample(self, nb_samples):
        subsampled_source = self.source_split.subsample(nb_samples)
        return SequenceReBatch(subsampled_source, batch_size=self.batch_size, discard_batch_not_full=self.discard_batch_not_full, collate_fn=self.collate_fn)

    def subsample_uids(self, uids, uids_name, new_sampler=None):
        subsampled_source = self.source_split.subsample_uids(uids, uids_name, new_sampler)
        return SequenceReBatch(subsampled_source, batch_size=self.batch_size, discard_batch_not_full=self.discard_batch_not_full, collate_fn=self.collate_fn)

    def __next__(self):
        batches = []
        total_nb_samples = 0
        try:
            while True:
                if self.iter_overflow is not None:
                    # handles the samples that had previously overflown
                    batch = self.iter_overflow
                    self.iter_overflow = None
                else:
                    # if not, get the next batch
                    time_wait_start = time.perf_counter()
                    batch = self.iter_source.__next__()
                    time_wait_end = time.perf_counter()
                    self.statistics.waiting_time_batches += time_wait_end - time_wait_start
                    self.statistics.nb_batches_input += 1

                if batch is None or len(batch) == 0:
                    # for some reason, the batch is empty
                    # get a new one!
                    continue

                nb_samples = len_batch(batch)
                if total_nb_samples + nb_samples == self.batch_size:
                    # here we are good!
                    batches.append(batch)
                    break

                if total_nb_samples + nb_samples > self.batch_size:
                    # too many samples, split the batch and keep the extra samples in the overflow
                    first_batch_size = self.batch_size - total_nb_samples

                    time_start = time.perf_counter() 
                    first_batch, overflow_batch = split_in_2_batches(batch, first_batch_size)
                    time_end = time.perf_counter()
                    self.statistics.waiting_time_split += time_end - time_start

                    batches.append(first_batch)
                    self.iter_overflow = overflow_batch
                    break

                # else keep accumulating until we have enough samples
                total_nb_samples += nb_samples
                batches.append(batch)

        except StopIteration:
            if len(batches) == 0 or (len(batches) != self.batch_size and self.discard_batch_not_full):
                # end the sequence
                overhead_time = self.statistics.waiting_time_batches + self.statistics.waiting_time_rebatch + self.statistics.waiting_time_split
                logger.info(f'SequenceReBatch={self}, overhead={overhead_time}, waiting_time_batches={self.statistics.waiting_time_batches}, waiting_time_rebatch={self.statistics.waiting_time_rebatch}, waiting_time_split={self.statistics.waiting_time_split}')
                raise StopIteration()

        if self.collate_fn is not None:
            # finally make a batch
            time_start = time.perf_counter()
            batches = self.collate_fn(batches)
            time_end = time.perf_counter()
            self.statistics.waiting_time_rebatch += time_end - time_start

        self.statistics.nb_batches_output += 1
        return batches

    def __iter__(self):
        self.statistics.reset()
        self.iter_source = self.source_split.__iter__()
        return self

    def close(self):
        if self.source_split is not None:
            self.source_split.close()

