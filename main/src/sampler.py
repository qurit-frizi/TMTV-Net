import typing
import torch
import collections
import numpy as np



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

class Sampler(object):
    """
    Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self):
        self.data_source = None

    def initializer(self, data_source):
        """
        Initialize the sequence iteration

        Args:
            data_source: the data source to iterate
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        Returns: an iterator the return indices of the original data source
        """
        raise NotImplementedError()


class _SamplerSequentialIter:
    """
    Lazily iterate the indices of a sequential batch
    """
    def __init__(self, nb_samples, batch_size):
        self.nb_samples = nb_samples
        self.batch_size = batch_size
        self.current = 0

    def __next__(self):
        if self.current >= self.nb_samples:
            raise StopIteration()

        indices = np.arange(self.current, min(self.current + self.batch_size, self.nb_samples))
        self.current += self.batch_size
        return indices


class SamplerSequential(Sampler):
    """
    Samples elements sequentially, always in the same order.
    """
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

    def initializer(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        if self.batch_size == 1:
            return iter(range(len_batch(self.data_source)))
        else:
            return _SamplerSequentialIter(len_batch(self.data_source), self.batch_size)


class SamplerRandom(Sampler):
    """
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    """

    def __init__(self, replacement=False, nb_samples_to_generate=None, batch_size=1):
        """

        Args:
            replacement: samples are drawn with replacement if ``True``, default=``False``
            nb_samples_to_generate: number of samples to draw, default=`len(dataset)`. This argument
                is supposed to be specified only when `replacement` is ``True``.
            batch_size: the number of samples returned by each batch. If possible, use this instead of ``SequenceBatch`` for performance reasons
        """
        super().__init__()
        
        if nb_samples_to_generate is not None:
            assert replacement, 'can only specified `nb_samples_to_generate` when we sample with replacement'
            
        self.replacement = replacement
        self.nb_samples_to_generate = nb_samples_to_generate
        self.indices = None
        self.last_index = None
        self.num_samples = None
        self.batch_size = batch_size

    def initializer(self, data_source):
        self.data_source = data_source
        self.indices = None
        self.last_index = 0

        self.num_samples = len_batch(self.data_source)
        if not self.replacement and self.nb_samples_to_generate is None:
            self.nb_samples_to_generate = self.num_samples
        
    def __iter__(self):
        if self.replacement:
            self.indices = np.random.randint(0, self.num_samples, size=self.nb_samples_to_generate, dtype=np.int64)
        else:
            self.indices = np.arange(0, self.num_samples)
        np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.last_index >= len(self.indices):
            raise StopIteration
        
        next_indices = self.indices[self.last_index:self.last_index + self.batch_size]
        self.last_index += self.batch_size
        return next_indices


class SamplerSubsetRandom(Sampler):
    """
    Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def initializer(self, data_source):
        pass

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))


class SamplerSubsetRandomByListInterleaved(Sampler):
    """
    Elements from a given list of list of indices are randomly drawn without replacement,
    one element per list at a time.
    
    For sequences with different sizes, the longest of the sequences will be trimmed to
    the size of the shortest sequence. 

    This can be used for example to resample without replacement imbalanced
    classes in a classification task.

    Examples::

        >>> l1 = np.asarray([1, 2])
        >>> l2 = np.asarray([3, 4, 5])
        >>> sampler = SamplerSubsetRandomByListInterleaved([l1, l2])
        >>> sampler.initializer(None)
        >>> indices = [i for i in sampler]
        # indices could be [1, 5, 2, 4]

    Arguments:
        indices: a sequence of sequence of indices
    """

    def __init__(self, indices: typing.Sequence[typing.Sequence[int]]):
        super().__init__()
        self.indices = indices
        self.indices_interleaved = None

    def initializer(self, data_source):
        nb_elements = [len(l) for l in self.indices]
        min_element = min(nb_elements)

        # trim and randomize the list elements
        indices_trimmed = []
        for l in self.indices:
            l = np.asarray(l)
            shuffled_indices = torch.randperm(len(l))
            indices_trimmed.append(l[shuffled_indices[:min_element]])

        # interleave each element of the list
        self.indices_interleaved = np.vstack(indices_trimmed).reshape((-1,), order='F')

    def __iter__(self):
        return self.indices_interleaved.__iter__()


class SamplerClassResampling(Sampler):
    """
    Resample the samples so that `class_name` classes have equal probably of being sampled.
    
    Classification problems rarely have balanced classes so it is often required to super-sample the minority class to avoid
    penalizing the under represented classes and help the classifier to learn good features (as opposed to learn the class
    distribution).
    """

    def __init__(self, class_name, nb_samples_to_generate, reuse_class_frequencies_across_epochs=True, batch_size=1):
        """
        :param class_name: the class to be resampled. Classes must be integers
        :param nb_samples_to_generate: the number of samples to generate
        :param reuse_class_frequencies_across_epochs: if True, the class frequencies will be calculated only once then reused from epoch to epoch. This is
            because iterating through the samples to calculate the class frequencies may be time consuming and it should not change over the epochs.
        """
        super().__init__()
        self.class_name = class_name
        self.nb_samples_to_generate = nb_samples_to_generate
        self.reuse_class_frequencies_across_epochs = reuse_class_frequencies_across_epochs
        self.batch_size = batch_size
        
        self.samples_index_by_classes = None
        self.indices = None
        self.current_index = None

        self.last_data_source_samples = 0

    def initializer(self, data_source):
        assert self.class_name in data_source, 'can\'t find {} in data!'.format(self.class_name)
        self.data_source = data_source

        data_source_samples = len_batch(data_source)

        classes = to_value(data_source[self.class_name])  # we want numpy values here!
        assert len(classes.shape) == 1, 'must be a 1D vector representing a class'
        if self.samples_index_by_classes is None or \
           not self.reuse_class_frequencies_across_epochs or \
           data_source_samples != self.last_data_source_samples:  # if we don't have the same size, rebuild the cache
                self._fit(classes)

        self.last_data_source_samples = data_source_samples

        nb_classes = len(self.samples_index_by_classes)
        nb_samples_per_class = self.nb_samples_to_generate // nb_classes

        indices_by_class = []
        for class_name, indices in self.samples_index_by_classes.items():
            indices_of_indices = np.random.randint(0, len(indices), nb_samples_per_class)
            indices_by_class.append(indices[indices_of_indices])

        # concatenate the indices by class, then shuffle them
        # to make sure we don't have batches with only the same class!
        self.indices = np.concatenate(indices_by_class)
        np.random.shuffle(self.indices)

        self.current_index = 0

    def _fit(self, classes):
        d = collections.defaultdict(lambda: [])
        for index, c in enumerate(classes):
            d[c].append(index)

        self.samples_index_by_classes = {
            c: np.asarray(indexes) for c, indexes in d.items()
        }

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        next_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        return next_indices

    def __iter__(self):
        return self
