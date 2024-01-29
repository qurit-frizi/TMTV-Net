from collate import default_collate_fn
from sequence import Sequence, SequenceIterator


class SequenceCollate(Sequence, SequenceIterator):
    """
    Group the data into a sequence of dictionary of torch.Tensor

    This can be useful to combine batches of dictionaries into a single batch with all features
    concatenated on axis 0. Often used in conjunction of :class:`SequenceAsyncReservoir`
    and :class:`SequenceMap`.
    """
    def __init__(self, source_split, collate_fn=default_collate_fn, device=None):
        """
        Group the samples into a batch.

        :param source_split: the source sequence
        :param device: the device where to send the samples
        :param collate_fn: the function to assemble a list of items. If None, return the items as they were in `source_split`
        """
        super().__init__(source_split)

        assert isinstance(source_split, Sequence), '`source_split` must be a `Sequence`'
        self.source_split = source_split
        self.collate_fn = collate_fn
        self.device = device

    def subsample(self, nb_samples):
        subsampled_source = self.source_split.subsample(nb_samples)
        return SequenceCollate(subsampled_source, collate_fn=self.collate_fn, device=self.device)

    def subsample_uids(self, uids, uids_name, new_sampler=None):
        subsampled_source = self.source_split.subsample_uids(uids, uids_name, new_sampler)
        return SequenceCollate(subsampled_source, collate_fn=self.collate_fn, device=self.device)

    def __next__(self):
        items = self.iter_source.__next__()
        items = self.collate_fn(items, device=self.device)
        return items

    def __iter__(self):
        self.iter_source = self.source_split.__iter__()
        return self

    def close(self):
        if self.source_split is not None:
            self.source_split.close()
