


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


from .callback_tensorboard import CallbackTensorboardBased
import utilities
import outputs as O
import functools
import collections
import torch
import numpy as np
import logging


logger = logging.getLogger(__name__)


def get_as_image(images):
    """
    Return the images as (N, C, H, W) or None if not an image

    TODO: smarter image detection!

    :param images: the object to check
    :return: None if not an image, or images with format (N, C, H, W)
    """
    if isinstance(images, (np.ndarray, torch.Tensor)):
        if len(images.shape) == 4 and (images.shape[1] == 1 or images.shape[1] == 3):
            return images
    return None


def keep_small_features(feature_name, feature_value):
    """
    Keep only the small features (e.g., len(shape) == 1) for the embedding infos

    :return: if True, keep the feature else discard it
    """
    if isinstance(feature_value, torch.Tensor) and len(feature_value.shape) > 1:
        return False
    return True


def is_batch_vector(value, batch_size):
    """
    Return true if a vector like
    :param value: the value to test
    :param batch_size: the expected size of the batch
    """
    vector_size = 0
    is_vector = False
    if isinstance(value, (torch.Tensor, np.ndarray)):
        is_vector = True
        if len(value.shape) != 0:
            vector_size = value.shape[0]
    elif isinstance(value, list):
        is_vector = True
        vector_size = len(value)

    return is_vector and vector_size == batch_size


def add_classification_strings_from_output(dataset_name, split_name, output, datasets_infos, prefix=''):
    """
    Special classification helper: add the class name (output and output_truth) as a string using the class
    mapping contained in `datasets_infos`

    :param dataset_name: the dataset name
    :param split_name: the split name
    :param output: the output
    :param datasets_infos: should contain the mapping
    :param prefix: the output and output_truth will be prefixed with `prefix`
    :return: the additional strings in a dictionary
    """
    output_dict = {}

    is_classification = False
    output_ref = output.get('output_ref')
    if output_ref is not None:
        is_classification = isinstance(output_ref, O.OutputClassification)

    if is_classification:
        # special handling of the classification node: add class names in string too so it is easier
        # to review the results, specially when we have many classes
        mapping = utilities.get_classification_mapping(datasets_infos, dataset_name, split_name, output_ref.classes_name)
        if mapping is not None:
            output_values = output.get('output')
            nb_samples = len(output_values)

            output_strs = []
            output_truth_strs = []
            for n in range(nb_samples):
                output_str = utilities.get_class_name(mapping, output_values[n])
                output_truth_values = output.get('output_truth')
                output_truth_str = utilities.get_class_name(mapping, output_truth_values[n])
                output_strs.append(output_str)
                output_truth_strs.append(output_truth_str)

            output_dict[prefix + 'output_str'] = output_strs
            output_dict[prefix + 'output_truth_str'] = output_truth_strs
    return output_dict


class CallbackTensorboardEmbedding(CallbackTensorboardBased):
    """
    This callback records the embedding to be displayed with tensorboard

    Note: we must recalculate the embedding as we need to associate a specific input (i.e., we can't store
    everything in memory so we need to collect what we need batch by batch)
    """
    def __init__(self, embedding_name, dataset_name=None, split_name=None, image_name=None, maximum_samples=2000, keep_features_fn=keep_small_features):
        """
        :param embedding_name: the name of the embedding to be used
        :param dataset_name: the name of the dataset to export the embedding. If `None`,
        we will try to find the best match
        :param split_name: the split of the dataset to export the embedding. if
        :param image_name: the image name to be used in tensorboard. If `None`, we will try to find
        an image like tensor to be used. If the `image_name` is not None but is not found in the batch,
        no image will be exported
        :param maximum_samples: the maximum number of samples to be exported for this embedding
        """
        self.embedding_name = embedding_name
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.image_name = image_name
        self.maximum_samples = maximum_samples
        self.keep_features_fn = keep_features_fn
        self.features_to_discard = ['output_ref']

    def first_time(self, datasets, options):
        self.dataset_name, self.split_name = utilities.find_default_dataset_and_split_names(
            datasets,
            default_dataset_name=self.dataset_name,
            default_split_name=self.split_name,
            train_split_name=options.workflow_options.train_split
        )

        if self.dataset_name is None:
            return

        if self.image_name is None:
            # try to find a tensor that has the shape of images
            for batch in datasets[self.dataset_name][self.split_name]:
                for feature_name, feature in batch.items():
                    as_image = get_as_image(feature)
                    if as_image is not None:
                        self.image_name = feature_name
                        break
                break
            if self.image_name is None:
                # we haven't found a suitable image for the given dataset/split
                # so use an impossible name
                self.image_name = ''

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        root = options.workflow_options.current_logging_directory
        logger.info('root={}, nb_samples={}'.format(root, self.maximum_samples))
        logger_tb = CallbackTensorboardBased.create_logger(root)
        if logger_tb is None:
            return
        if self.dataset_name is None or self.image_name is None:
            self.first_time(datasets, options)
        if self.dataset_name is None or self.image_name is None:
            logger.info('embedding can not be calculated: dataset={}, split={}'.format(self.dataset_name, self.split_name))
            return None
        if datasets.get(self.dataset_name) is None or datasets[self.dataset_name].get(self.split_name) is None:
            logger.info('embedding can not be calculated: dataset={}, split={}'.format(self.dataset_name, self.split_name))
            return
        logger.info('parameters: dataset={}, split={}, embedding={}, image_name={}'.format(self.dataset_name, self.split_name, self.embedding_name, self.image_name))

        device = options.workflow_options.device
        logger.info('collecting embeddings')

        # here collect the embeddings and images
        embedding = collections.defaultdict(list)
        nb_samples_collected = 0

        def fill_embedding(batch_size, output, prefix='', output_name=None):
            # special handling of the classification node: add class names in string too so it is easier
            # to review the results, specially when we have many classes
            additional_strings = add_classification_strings_from_output(
                self.dataset_name,
                self.split_name,
                output,
                datasets_infos,
                prefix=prefix
            )
            for name, value in additional_strings.items():
                embedding[name].append(value)

            # record the output metrics
            for feature_name, feature_values in output.items():
                if feature_name == self.image_name:
                    continue
                if not self.keep_features_fn(feature_name, feature_values):
                    continue
                if feature_name in self.features_to_discard:
                    continue

                # if we have a vector, it means it is a per-sample feature
                # else a global feature (e.g., learning rate, dropout rate...)
                full_name = prefix + feature_name
                if is_batch_vector(feature_values, batch_size):
                    embedding[full_name].append(to_value(feature_values))

        def collect_embedding(dataset_name, split_name, batch, loss_terms, embedding, embedding_name, image_name, **kwargs):
            batch_size = len_batch(batch)

            embedding_values = loss_terms.get(embedding_name)
            if embedding_values is not None:
                embedding['output'].append(to_value(embedding_values['output']))

            for output_name, output in loss_terms.items():
                if output_name == embedding_name:
                    continue

                fill_embedding(batch_size, output, prefix=output_name + '-', output_name=output_name)

            images = batch.get(image_name)
            if images is not None:
                images = get_as_image(images)
                embedding['images'].append(to_value(images))

            fill_embedding(batch_size, batch)

            nonlocal nb_samples_collected
            if nb_samples_collected >= self.maximum_samples:
                # we have exceeded the number of samples to collect, stop the loop
                raise StopIteration()
            nb_samples_collected += batch_size

        from ..train.trainer import eval_loop
        eval_loop(
            options,
            device,
            self.dataset_name,
            self.split_name,
            datasets[self.dataset_name][self.split_name],
            model,
            losses[self.dataset_name],
            history,
            callbacks_per_batch=callbacks_per_batch,
            callbacks_per_batch_loss_terms=[functools.partial(collect_embedding, embedding=embedding, embedding_name=self.embedding_name, image_name=self.image_name)]
        )

        logger.info('collecting embeddings done!')
        
        # merge the batches
        merged_embedding = {}
        for name, values in embedding.items():
            merged_embedding[name] = np.concatenate(values)

        embedding_values = merged_embedding.get('output')
        if embedding_values is None:
            logger.info('No embedding `output` could be found!')
            return

        images = merged_embedding.get('images')
        if images is not None:
            assert len(images.shape) == 4 and (images.shape[1] == 1 or images.shape[1] == 3), \
                'Expected images format (N, C, H, W), got shape={}'.format(images.shape)
            images = torch.Tensor(images)

        # export the metada
        metadata_header = []
        metadata = []
        for name, values in merged_embedding.items():
            if name != 'output' and name != 'images':
                metadata_header.append(name)

                values_str = [str(v).replace('\n', ' ').replace('\t', ' ') for v in values]
                metadata.append(values_str)
        if len(metadata_header) != 0:
            metadata = np.stack(metadata, axis=1)

        # export the embedding to the tensorboard log
        logger.info('adding embedding...')
        logger_tb.add_embedding(
            embedding_values,
            label_img=images,
            global_step=len(history) - 1,
            metadata=metadata.tolist(),
            metadata_header=metadata_header)
        logger.info('embedding successfully added!')
