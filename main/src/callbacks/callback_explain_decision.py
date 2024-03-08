import os

from utilities import postprocess_batch, transfer_batch_to_device, create_or_recreate_folder

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

from callbacks import callback
import utilities
import sample_export

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
import outputs
import guided_back_propagation
import grad_cam
import integrated_gradients
import meaningful_perturbation
from enum import Enum
import torch
import torch.nn
import numpy as np
import logging
import collections
import functools


logger = logging.getLogger(__name__)


class ExplainableAlgorithm(Enum):
    GuidedBackPropagation = guided_back_propagation.GuidedBackprop
    GradCAM = grad_cam.GradCam
    Gradient = functools.partial(guided_back_propagation.GuidedBackprop, unguided_gradient=True)
    IntegratedGradients = integrated_gradients.IntegratedGradients
    MeaningfulPerturbations = meaningful_perturbation.MeaningfulPerturbation


def default_algorithm_args():
    """
    Default algorithm arguments
    """
    return {
        ExplainableAlgorithm.MeaningfulPerturbations: {
            'mask_reduction_factor': 4,
            'iterations': 200,
            'l1_coeff': 0.1,
            'information_removal_fn': functools.partial(meaningful_perturbation.default_information_removal_smoothing, blurring_sigma=4, blurring_kernel_size=13)
        }
    }


def run_classification_explanation(
        root,
        dataset_name,
        split_name,
        model,
        batch,
        datasets_infos,
        nb_samples,
        algorithm_name,
        algorithm_fn,
        output_name,
        algorithm_kwargs=None,
        nb_explanations=1,
        epoch=None,
        average_filters=True):
    """
    Run an explanation of a classification output
    """

    # do sample by sample to simplify the export procedure
    for n in range(nb_samples):
        logger.info('sample={}'.format(n))
        batch_n = get_batch_n(
            batch,
            len_batch(batch),
            np.asarray([n]),
            transforms=None,
            use_advanced_indexing=True)

        for tensor in batch_n.values():
            if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
                # we want to back-propagate up to the inputs
                tensor.requires_grad = True

        try:
            with torch.no_grad():
                outputs = model(batch_n)
                output = outputs.get(output_name)
                assert output is not None
                output_np = to_value(output.output)[0]
                max_class_indices = (-output_np).argsort()[0:nb_explanations]
        except Exception as e:
            logger.error('exception, aborted `run_classification_explanation`=', e)
            continue

        # make sure the model is not contaminated by uncleaned hooks
        r = None
        with utilities.CleanAddedHooks(model) as context:
            algorithm_instance = algorithm_fn(model=model, **algorithm_kwargs)
            r = algorithm_instance(inputs=batch_n, target_class_name=output_name, target_class=max_class_indices[0])

        if r is None:
            # the algorithm failed, go to the next one
            return

        selected_output_name, cams_dict = r
        assert nb_explanations == 1, 'TODO handle for multiple explanations!'

        for input_name, g in cams_dict.items():
            if g is None:
                # discard this input!
                continue

            enumerate_i = 0
            c = max_class_indices[enumerate_i]  # the class output
            c_name = fill_class_name(output, c, datasets_infos, dataset_name, split_name)

            filename = 'sample-{}-output-{}-epoch-{}-rank-{}-alg-{}-explanation_for-{}'.format(n, input_name, epoch, enumerate_i, algorithm_name, c_name)
            filename = utilities.safe_filename(filename)
            export_path = os.path.join(root, filename)

            def format_image(g):
                if not isinstance(g, (np.ndarray, torch.Tensor)):
                    return g
                if average_filters and len(g.shape) >= 3:
                    return np.reshape(np.average(np.abs(g), axis=1), [g.shape[0], 1] + list(g.shape[2:]))
                return g

            with open(export_path + '.txt', 'w') as f:
                if isinstance(g, collections.Mapping):
                    # handle multiple explanation outputs
                    for name, value in g.items():
                        batch_n['explanation_{}'.format(name)] = format_image(value)
                else:
                    # default: single tensor
                    batch_n['explanation'] = format_image(g)
                batch_n['output_found'] = str(output_np)
                batch_n['output_name_found'] = c_name

                #positive, negative = guided_back_propagation.GuidedBackprop.get_positive_negative_saliency(g)
                #batch_n['explanation_positive'] = positive
                #batch_n['explanation_negative'] = negative
                #f.write('gradient average positive={}\n'.format(np.average(g[np.where(g > 0)])))
                #f.write('gradient average negative={}\n'.format(np.average(g[np.where(g < 0)])))
                sample_export.export_sample(batch_n, 0, export_path + '-', f)


def fill_class_name(output, class_index, datasets_infos, dataset_name, split_name):
    """
    Get the class name if available, if not the class index
    """

    c_name = None
    if isinstance(output, outputs.OutputClassification):
        c_names = utilities.get_classification_mapping(datasets_infos, dataset_name, split_name, output.classes_name)
        if c_names is not None:
            c_names = c_names['mappinginv']
            c_name = c_names.get(class_index)
    if c_name is None:
        c_name = class_index
    return c_name


class CallbackExplainDecision(callback.Callback):
    """
    Explain the decision of a model
    """
    def __init__(self, max_samples=10, dirname='explained', dataset_name=None, split_name=None, algorithm=(ExplainableAlgorithm.MeaningfulPerturbations, ExplainableAlgorithm.GuidedBackPropagation, ExplainableAlgorithm.GradCAM, ExplainableAlgorithm.Gradient, ExplainableAlgorithm.IntegratedGradients), output_name=None, nb_explanations=1, algorithms_kwargs=default_algorithm_args(), average_filters=True):
        """
        Args:
            max_samples: the maximum number of examples to export
            dirname: folder name where to export the explanations
            dataset_name: the name of the dataset to export. If `None`, the first dataset is chosen
            split_name: the split name to use
            algorithm: the algorithm (`ExplainableAlgorithm`) to be used to explain the model's decision or a list of `ExplainableAlgorithm`
            output_name: the output to be used as classification target. If `None`, report the first output belonging to a `OutputClassification`
            nb_explanations: the number of alternative explanations to be exported. nb_explanations = 1, explain the current guess, nb_explanations = 2,
                in addition calculate the explanation for the next best guess and so on for nb_explanations = 3
            algorithms_kwargs: additional argument (a dictionary of dictionary of algorithm argument) to be provided to the algorithm or `None`.
            average_filters: if True, the explanation will be grey value (averaged)
        """
        self.max_samples = max_samples
        self.dirname = dirname
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.algorithms_kwargs = algorithms_kwargs
        self.average_filters = average_filters

        # since all `explanation` algorithms all have drawbacks, we will
        # want to export multiple explanations and so make it a list
        if isinstance(algorithm, ExplainableAlgorithm):
            self.algorithms = [algorithm]
        else:
            assert isinstance(algorithm, collections.Sequence)
            self.algorithms = algorithm

        self.batch = None
        self.output_name = output_name
        self.nb_explanations = nb_explanations

    def first_time(self, datasets, options):

        self.dataset_name, self.split_name = utilities.find_default_dataset_and_split_names(
            datasets,
            self.dataset_name,
            self.split_name,
            train_split_name=options.workflow_options.train_split)

        if self.dataset_name is None:
            logger.error('can\'t find split={} for dataset={}'.format(self.split_name, self.dataset_name))
            return

        # record a particular batch of data so that we can have the exact same samples over multiple epochs
        self.batch = next(iter(datasets[self.dataset_name][self.split_name]))

    @staticmethod
    def find_output_name(outputs, name):
        if name is not None:
            return name

        for output_name, output in outputs.items():
            if isinstance(output, outputs.OutputClassification):
                return output_name

        return None

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('started CallbackExplainDecision.__call__')
        device = options.workflow_options.device
        model.eval()  # we are in evaluation mode!

        if self.batch is None:
            self.first_time(datasets, options)
        if self.batch is None:
            return

        root = os.path.join(options.workflow_options.current_logging_directory, self.dirname)
        if not os.path.exists(root):
            create_or_recreate_folder(root)

        batch = transfer_batch_to_device(self.batch, device=device)
        postprocess_batch(self.dataset_name, self.split_name, batch, callbacks_per_batch)

        outputs = model(batch)
        output_name = CallbackExplainDecision.find_output_name(outputs, self.output_name)
        if output_name is None:
            logger.error('can\'t find a classification output')
            return

        nb_samples = min(self.max_samples, len_batch(batch))
        for algorithm in self.algorithms:
            algorithm_kwargs = {}
            if self.algorithms_kwargs is not None and algorithm in self.algorithms_kwargs:
                algorithm_kwargs = self.algorithms_kwargs.get(algorithm)

            run_classification_explanation(
                root,
                self.dataset_name,
                self.split_name,
                model,
                batch,
                datasets_infos,
                nb_samples,
                algorithm.name,
                algorithm_fn=algorithm.value,
                output_name=output_name,
                algorithm_kwargs=algorithm_kwargs,
                nb_explanations=1,
                epoch=len(history),
                average_filters=self.average_filters)

        logger.info('successfully completed CallbackExplainDecision.__call__!')
